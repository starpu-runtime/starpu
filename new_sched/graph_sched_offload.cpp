/* Shared GPU offload / eviction helpers for the SGOC graph scheduler. */

#include "graph_sched_internal.hpp"

#include <mutex>
#include <unordered_set>

#include <starpu_data.h>

static bool graph_sched_query_valid_on_node(starpu_data_handle_t h, int memory_node)
{
    int a = 0, v = 0, loading = 0, req = 0;
    starpu_data_query_status2(h, memory_node, &a, &v, &loading, &req);
    return v != 0;
}

/**
 * starpu_data_prefetch_on_node(..., RAM) uses a read path and asserts if the handle is not initialized for read.
 * MM offload keys can include scratch-only buffers; skip those (and any handle not referenced by \p task).
 */
static bool graph_sched_task_uses_handle_for_ram_read_offload(struct starpu_task *task, starpu_data_handle_t h)
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
static bool graph_sched_task_writes_handle(struct starpu_task *task, starpu_data_handle_t h)
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

static void graph_sched_pending_gpu_evict_push_unique(graph_sched_data *data, starpu_data_handle_t h)
{
    if (!h)
        return;
    for (starpu_data_handle_t p : data->graph_pending_gpu_evict_handles) {
        if (p == h)
            return;
    }
    data->graph_pending_gpu_evict_handles.push_back(h);
}

static void graph_sched_sync_pending_evict_count(graph_sched_data *data)
{
    data->graph_pending_gpu_evict_pending_count.store(data->graph_pending_gpu_evict_handles.size(),
                                                      std::memory_order_relaxed);
}

void graph_sched_drain_deferred_ram_offload_copies(graph_sched_data *data, unsigned gpu_mem_node)
{
    if (!data)
        return;
    std::vector<starpu_data_handle_t> batch;
    {
        std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
        if (data->graph_offload_defer_ram_w_acquire.empty())
            return;
        batch.reserve(data->graph_offload_defer_ram_w_acquire.size());
        while (!data->graph_offload_defer_ram_w_acquire.empty()) {
            batch.push_back(data->graph_offload_defer_ram_w_acquire.front());
            data->graph_offload_defer_ram_w_acquire.pop_front();
        }
    }
    const int ram_i = static_cast<int>(STARPU_MAIN_RAM);
    const int gpu_i = static_cast<int>(gpu_mem_node);
    std::vector<starpu_data_handle_t> to_pending;
    to_pending.reserve(batch.size());
    for (starpu_data_handle_t h : batch) {
        if (!h)
            continue;
        if (!graph_sched_query_valid_on_node(h, gpu_i))
            continue;
        /* Coherent RAM replica already tracked by StarPU — skip W acquire and evict GPU copy. */
        if (graph_sched_query_valid_on_node(h, ram_i)) {
            to_pending.push_back(h);
            continue;
        }
        /* W-mode acquire skips read-initialization assert (see StarPU user_interactions _starpu_data_check_initialized). */
        if (starpu_data_acquire_on_node_try(h, ram_i, STARPU_W) != 0)
            continue;
        starpu_data_release_on_node(h, ram_i);
        to_pending.push_back(h);
        if (data->graph_sgoc && data->graph_sgoc->mem_debug) {
            data->graph_sgoc->dbg_offload_ram_issue.fetch_add(1u, std::memory_order_relaxed);
            data->graph_sgoc->dbg_offload_ram_bytes.fetch_add(static_cast<std::uint64_t>(starpu_data_get_size(h)),
                                                              std::memory_order_relaxed);
        }
    }
    if (to_pending.empty())
        return;
    {
        std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
        for (starpu_data_handle_t h : to_pending) {
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
        graph_sched_sync_pending_evict_count(data);
    }
    graph_sched_drain_pending_gpu_evicts(data, gpu_mem_node);
}

void graph_sched_register_evict_gpu_only_after_task(graph_sched_data *data, struct starpu_task *task,
                                                    const std::vector<void *> &handles)
{
    if (!data || !task || handles.empty())
        return;
    std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
    auto &vec = data->graph_evict_gpu_only_after_task_handles[task];
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

void graph_sched_register_offload_after_task(graph_sched_data *data, struct starpu_task *task,
                                             const std::vector<void *> &s_offload_keys)
{
    if (!data || !task || s_offload_keys.empty())
        return;
    std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
    auto &vec = data->graph_offload_after_task_handles[task];
    const size_t before = vec.size();
    std::unordered_set<void *> seen;
    seen.reserve(vec.size() + s_offload_keys.size());
    for (starpu_data_handle_t h : vec)
        if (h)
            seen.insert(static_cast<void *>(h));
    for (void *k : s_offload_keys) {
        if (!k || seen.count(k))
            continue;
        const starpu_data_handle_t h = static_cast<starpu_data_handle_t>(k);
        if (!graph_sched_task_uses_handle_for_ram_read_offload(task, h))
            continue;
        seen.insert(k);
        vec.push_back(h);
    }
    if (data->graph_sgoc && data->graph_sgoc->mm_order_trace && vec.size() > before)
        data->graph_sgoc->dbg_mm_trace_offload_regs.fetch_add(static_cast<std::uint64_t>(vec.size() - before),
                                                              std::memory_order_relaxed);
}

void graph_sched_run_post_exec_offloads(graph_sched_data *data, struct starpu_task *task, unsigned gpu_mem_node)
{
    if (!data || !task)
        return;
    std::vector<starpu_data_handle_t> work;
    {
        std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
        auto it = data->graph_offload_after_task_handles.find(task);
        if (it == data->graph_offload_after_task_handles.end())
            return;
        work = std::move(it->second);
        data->graph_offload_after_task_handles.erase(it);
    }
    if (!work.empty() && data->graph_sgoc && data->graph_sgoc->mm_order_trace)
        data->graph_sgoc->dbg_mm_trace_post_exec_offload_tasks.fetch_add(1u, std::memory_order_relaxed);
    const int gpu_i = static_cast<int>(gpu_mem_node);
    const int ram_i = static_cast<int>(STARPU_MAIN_RAM);
    std::vector<starpu_data_handle_t> defer_ram;
    defer_ram.reserve(work.size());
    std::vector<starpu_data_handle_t> direct_evict;
    direct_evict.reserve(work.size());
    for (starpu_data_handle_t h : work) {
        if (!h)
            continue;
        if (!graph_sched_task_uses_handle_for_ram_read_offload(task, h))
            continue;
        if (!graph_sched_query_valid_on_node(h, gpu_i))
            continue;
        /* Belady plan schedules offload at this post_exec (after last use before the simulated pressure point). If RAM
         * already holds a coherent replica and this task did not write the buffer, replicate is unnecessary — evict GPU
         * immediately (typical read-only parameters). */
        if (graph_sched_query_valid_on_node(h, ram_i) && !graph_sched_task_writes_handle(task, h))
            direct_evict.push_back(h);
        else
            defer_ram.push_back(h);
    }
    {
        std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
        for (starpu_data_handle_t h : defer_ram)
            data->graph_offload_defer_ram_w_acquire.push_back(h);
        for (starpu_data_handle_t h : direct_evict)
            graph_sched_pending_gpu_evict_push_unique(data, h);
        graph_sched_sync_pending_evict_count(data);
    }
    /* Deferred RAM replicate + GPU evict: graph_sched_drain_deferred_ram_offload_copies (push_task / end of flush). */
    graph_sched_drain_pending_gpu_evicts(data, gpu_mem_node);
}

void graph_sched_run_post_exec_evict_gpu_only(graph_sched_data *data, struct starpu_task *task, unsigned gpu_mem_node)
{
    if (!data || !task)
        return;
    std::vector<starpu_data_handle_t> work;
    {
        std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
        auto it = data->graph_evict_gpu_only_after_task_handles.find(task);
        if (it == data->graph_evict_gpu_only_after_task_handles.end())
            return;
        work = std::move(it->second);
        data->graph_evict_gpu_only_after_task_handles.erase(it);
    }
    const int gpu_i = static_cast<int>(gpu_mem_node);
    for (starpu_data_handle_t h : work) {
        if (!h)
            continue;
        if (!graph_sched_query_valid_on_node(h, gpu_i))
            continue;
        graph_sched_pending_gpu_evict_push_unique(data, h);
    }
    graph_sched_drain_pending_gpu_evicts(data, gpu_mem_node);
}

void graph_sched_drain_pending_gpu_evicts(graph_sched_data *data, unsigned gpu_mem_node)
{
    if (!data)
        return;
    if (data->graph_pending_gpu_evict_pending_count.load(std::memory_order_relaxed) == 0)
        return;
    std::vector<starpu_data_handle_t> batch;
    {
        std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
        batch.swap(data->graph_pending_gpu_evict_handles);
        graph_sched_sync_pending_evict_count(data);
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
    graph_sched_sync_pending_evict_count(data);
}

void graph_sched_clear_offload_task_registrations(graph_sched_data *data)
{
    if (!data)
        return;
    std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
    data->graph_offload_after_task_handles.clear();
    data->graph_evict_gpu_only_after_task_handles.clear();
    data->graph_offload_defer_ram_w_acquire.clear();
    data->graph_pending_gpu_evict_handles.clear();
    graph_sched_sync_pending_evict_count(data);
}
