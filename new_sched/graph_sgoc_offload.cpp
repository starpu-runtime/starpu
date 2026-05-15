/* Shared GPU offload / eviction helpers for the SGOC graph scheduler. */

#include "graph_sgoc_internal.hpp"

#include <mutex>
#include <unordered_set>

#include <starpu_data.h>

void graph_sgoc_drain_pending_gpu_evicts(graph_sgoc_data *data, unsigned gpu_mem_node);

static bool graph_sgoc_query_valid_on_node(starpu_data_handle_t h, int memory_node)
{
    int a = 0, v = 0, loading = 0, req = 0;
    starpu_data_query_status2(h, memory_node, &a, &v, &loading, &req);
    return v != 0;
}

/**
 * Registration filters: scratch-only buffers are not offload registration targets (uses \p producer_task).
 */
static bool graph_sgoc_task_uses_handle_for_ram_read_offload(struct starpu_task *task, starpu_data_handle_t h)
{
    if (!task || !h || !task->cl)
        return false;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    for (unsigned j = 0; j < nbuf; ++j) {
        if (STARPU_TASK_GET_HANDLE(task, j) != h)
            continue;
        const unsigned mode = STARPU_TASK_GET_MODE(task, j);
        if ((mode & STARPU_SCRATCH) != 0)
            continue;
        return true;
    }
    return false;
}

/** True if this task may write \p h on the worker (conservative when buffer not found). */
static bool graph_sgoc_task_writes_handle(struct starpu_task *task, starpu_data_handle_t h)
{
    if (!task || !h || !task->cl)
        return true;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    for (unsigned j = 0; j < nbuf; ++j) {
        if (STARPU_TASK_GET_HANDLE(task, j) != h)
            continue;
        const unsigned mode = STARPU_TASK_GET_MODE(task, j);
        if ((mode & STARPU_SCRATCH) != 0)
            return false;
        if ((mode & STARPU_REDUX) != 0 || (mode & STARPU_MPI_REDUX) != 0)
            return true;
        return (mode & STARPU_W) != 0;
    }
    return true;
}

static void graph_sgoc_pending_gpu_evict_push_unique(graph_sgoc_data *data, starpu_data_handle_t h)
{
    if (!h)
        return;
    for (starpu_data_handle_t p : data->graph_pending_gpu_evict_handles) {
        if (p == h)
            return;
    }
    data->graph_pending_gpu_evict_handles.push_back(h);
}

static void graph_sgoc_sync_pending_evict_count(graph_sgoc_data *data)
{
    data->graph_pending_gpu_evict_pending_count.store(data->graph_pending_gpu_evict_handles.size(),
                                                      std::memory_order_relaxed);
}

static void graph_sgoc_execute_offload_work(graph_sgoc_data *data,
                                            std::vector<graph_sgoc_pop_offload_entry> &ram_work,
                                            std::vector<starpu_data_handle_t> &evict_only_work,
                                            unsigned gpu_mem_node)
{
    if (!data)
        return;
    const int gpu_i = static_cast<int>(gpu_mem_node);
    const int ram_i = static_cast<int>(STARPU_MAIN_RAM);
    std::vector<starpu_data_handle_t> prefetch_evict_pending;
    std::vector<starpu_data_handle_t> direct_evict_pending;
    prefetch_evict_pending.reserve(ram_work.size());
    direct_evict_pending.reserve(ram_work.size());
    for (graph_sgoc_pop_offload_entry &ent : ram_work) {
        starpu_data_handle_t h = ent.handle;
        if (!h)
            continue;
        if (!graph_sgoc_query_valid_on_node(h, gpu_i))
            continue;
        if (graph_sgoc_query_valid_on_node(h, ram_i) && !ent.producer_wrote_handle) {
            direct_evict_pending.push_back(h);
        } else {
            if (starpu_data_prefetch_on_node(h, static_cast<unsigned>(ram_i), 1u) != 0)
                continue;
            prefetch_evict_pending.push_back(h);
            if (data->graph_sgoc && data->graph_sgoc->mem_debug) {
                data->graph_sgoc->dbg_offload_ram_issue.fetch_add(1u, std::memory_order_relaxed);
                data->graph_sgoc->dbg_offload_ram_bytes.fetch_add(static_cast<std::uint64_t>(starpu_data_get_size(h)),
                                                                  std::memory_order_relaxed);
            }
        }
    }
    ram_work.clear();
    {
        std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
        for (starpu_data_handle_t h : direct_evict_pending)
            graph_sgoc_pending_gpu_evict_push_unique(data, h);
        for (starpu_data_handle_t h : prefetch_evict_pending)
            graph_sgoc_pending_gpu_evict_push_unique(data, h);
        for (starpu_data_handle_t h : evict_only_work) {
            if (!h)
                continue;
            if (!graph_sgoc_query_valid_on_node(h, gpu_i))
                continue;
            graph_sgoc_pending_gpu_evict_push_unique(data, h);
        }
        graph_sgoc_sync_pending_evict_count(data);
    }
    evict_only_work.clear();
    graph_sgoc_drain_pending_gpu_evicts(data, gpu_mem_node);
}

void graph_sgoc_register_ram_offload_for_followup_pop(graph_sgoc_data *data, struct starpu_task *followup_task,
                                                      struct starpu_task *producer_task,
                                                      const std::vector<void *> &s_offload_keys)
{
    if (!data || !followup_task || !producer_task || s_offload_keys.empty())
        return;
    std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
    auto &vec = data->graph_offload_on_followup_pop[followup_task];
    const size_t before = vec.size();
    std::unordered_set<void *> seen;
    seen.reserve(vec.size() + s_offload_keys.size());
    for (const graph_sgoc_pop_offload_entry &e : vec) {
        if (e.handle)
            seen.insert(static_cast<void *>(e.handle));
    }
    for (void *k : s_offload_keys) {
        if (!k || seen.count(k))
            continue;
        const starpu_data_handle_t h = static_cast<starpu_data_handle_t>(k);
        if (!graph_sgoc_task_uses_handle_for_ram_read_offload(producer_task, h))
            continue;
        seen.insert(k);
        vec.push_back(
            graph_sgoc_pop_offload_entry{h, graph_sgoc_task_writes_handle(producer_task, h)});
    }
    if (data->graph_sgoc && data->graph_sgoc->mm_order_trace && vec.size() > before)
        data->graph_sgoc->dbg_mm_trace_offload_regs.fetch_add(static_cast<std::uint64_t>(vec.size() - before),
                                                              std::memory_order_relaxed);
}

void graph_sgoc_register_evict_gpu_only_for_followup_pop(graph_sgoc_data *data, struct starpu_task *followup_task,
                                                          const std::vector<void *> &handles)
{
    if (!data || !followup_task || handles.empty())
        return;
    std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
    auto &vec = data->graph_evict_gpu_only_on_followup_pop[followup_task];
    std::unordered_set<void *> seen;
    seen.reserve(vec.size() + handles.size());
    for (starpu_data_handle_t h : vec)
        if (h)
            seen.insert(static_cast<void *>(h));
    for (void *k : handles) {
        if (!k || seen.count(k))
            continue;
        seen.insert(k);
        vec.push_back(static_cast<starpu_data_handle_t>(k));
    }
}

void graph_sgoc_register_ram_offload_flush_tail(graph_sgoc_data *data, struct starpu_task *producer_task,
                                                const std::vector<void *> &s_offload_keys)
{
    if (!data || !producer_task || s_offload_keys.empty())
        return;
    std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
    for (void *k : s_offload_keys) {
        if (!k)
            continue;
        const starpu_data_handle_t h = static_cast<starpu_data_handle_t>(k);
        if (!graph_sgoc_task_uses_handle_for_ram_read_offload(producer_task, h))
            continue;
        data->graph_offload_flush_tail_ram.push_back(
            graph_sgoc_pop_offload_entry{h, graph_sgoc_task_writes_handle(producer_task, h)});
    }
}

void graph_sgoc_register_evict_gpu_flush_tail(graph_sgoc_data *data, const std::vector<void *> &handles)
{
    if (!data || handles.empty())
        return;
    std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
    for (void *k : handles) {
        if (!k)
            continue;
        data->graph_offload_flush_tail_evict_gpu.push_back(static_cast<starpu_data_handle_t>(k));
    }
}

void graph_sgoc_run_followup_pop_offloads(graph_sgoc_data *data, struct starpu_task *picked_task,
                                          unsigned gpu_mem_node)
{
    if (!data || !picked_task)
        return;
    std::vector<graph_sgoc_pop_offload_entry> ram_work;
    std::vector<starpu_data_handle_t> evict_only_work;
    {
        std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
        auto it = data->graph_offload_on_followup_pop.find(picked_task);
        if (it != data->graph_offload_on_followup_pop.end()) {
            ram_work = std::move(it->second);
            data->graph_offload_on_followup_pop.erase(it);
        }
        auto it2 = data->graph_evict_gpu_only_on_followup_pop.find(picked_task);
        if (it2 != data->graph_evict_gpu_only_on_followup_pop.end()) {
            evict_only_work = std::move(it2->second);
            data->graph_evict_gpu_only_on_followup_pop.erase(it2);
        }
    }
    const bool nonempty = !ram_work.empty() || !evict_only_work.empty();
    if (nonempty && data->graph_sgoc && data->graph_sgoc->mm_order_trace)
        data->graph_sgoc->dbg_mm_trace_followup_pop_offload_tasks.fetch_add(1u, std::memory_order_relaxed);
    graph_sgoc_execute_offload_work(data, ram_work, evict_only_work, gpu_mem_node);
}

void graph_sgoc_run_flush_tail_offloads(graph_sgoc_data *data, unsigned gpu_mem_node)
{
    if (!data)
        return;
    std::vector<graph_sgoc_pop_offload_entry> ram_work;
    std::vector<starpu_data_handle_t> evict_only_work;
    {
        std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
        ram_work = std::move(data->graph_offload_flush_tail_ram);
        evict_only_work = std::move(data->graph_offload_flush_tail_evict_gpu);
    }
    if (ram_work.empty() && evict_only_work.empty())
        return;
    graph_sgoc_execute_offload_work(data, ram_work, evict_only_work, gpu_mem_node);
}

void graph_sgoc_drain_pending_gpu_evicts(graph_sgoc_data *data, unsigned gpu_mem_node)
{
    if (!data)
        return;
    if (data->graph_pending_gpu_evict_pending_count.load(std::memory_order_relaxed) == 0)
        return;
    std::vector<starpu_data_handle_t> batch;
    {
        std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
        batch.swap(data->graph_pending_gpu_evict_handles);
        graph_sgoc_sync_pending_evict_count(data);
    }
    std::vector<starpu_data_handle_t> retry;
    retry.reserve(batch.size());
    for (starpu_data_handle_t h : batch) {
        if (!h)
            continue;
        if (starpu_data_can_evict(h, gpu_mem_node, STARPU_PREFETCH)
            && starpu_data_evict_from_node(h, gpu_mem_node) == 0) {
            std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
            data->graph_pending_gpu_evict_drained++;
            if (data->graph_sgoc && data->graph_sgoc->mem_debug)
                data->graph_sgoc->dbg_evict_ok.fetch_add(1u, std::memory_order_relaxed);
            if (data->graph_sgoc) {
                void *p = static_cast<void *>(h);
                if (data->graph_sgoc->tracked_gpu_resident.erase(p))
                    data->graph_sgoc->tracked_gpu_bytes -=
                        static_cast<std::int64_t>(starpu_data_get_size(h));
            }
        } else
            retry.push_back(h);
    }
    if (retry.empty())
        return;
    std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
    for (starpu_data_handle_t h : retry) {
        bool dup = false;
        for (starpu_data_handle_t p : data->graph_pending_gpu_evict_handles) {
            if (p == h) {
                dup = true;
                break;
            }
        }
        if (!dup)
            data->graph_pending_gpu_evict_handles.push_back(h);
    }
    graph_sgoc_sync_pending_evict_count(data);
}

void graph_sgoc_clear_offload_task_registrations(graph_sgoc_data *data)
{
    if (!data)
        return;
    std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
    data->graph_offload_on_followup_pop.clear();
    data->graph_evict_gpu_only_on_followup_pop.clear();
    data->graph_offload_flush_tail_ram.clear();
    data->graph_offload_flush_tail_evict_gpu.clear();
    data->graph_pending_gpu_evict_handles.clear();
    graph_sgoc_sync_pending_evict_count(data);
}
