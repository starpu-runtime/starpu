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

static void graph_sched_sync_pending_evict_count(graph_sched_data *data)
{
    data->graph_pending_gpu_evict_pending_count.store(data->graph_pending_gpu_evict_handles.size(),
                                                      std::memory_order_relaxed);
}

void graph_sched_register_offload_after_task(graph_sched_data *data, struct starpu_task *task,
                                             const std::vector<void *> &s_offload_keys)
{
    if (!data || !task || s_offload_keys.empty())
        return;
    std::lock_guard<std::mutex> lock(data->graph_offload_mutex);
    auto &vec = data->graph_offload_after_task_handles[task];
    std::unordered_set<void *> seen;
    seen.reserve(vec.size() + s_offload_keys.size());
    for (starpu_data_handle_t h : vec)
        if (h)
            seen.insert(static_cast<void *>(h));
    for (void *k : s_offload_keys) {
        if (!k || seen.count(k))
            continue;
        seen.insert(k);
        vec.push_back(static_cast<starpu_data_handle_t>(k));
    }
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
    const int gpu_i = static_cast<int>(gpu_mem_node);
    std::vector<starpu_data_handle_t> to_pending;
    to_pending.reserve(work.size());
    for (starpu_data_handle_t h : work) {
        if (!h)
            continue;
        if (!graph_sched_query_valid_on_node(h, gpu_i))
            continue;
        (void)starpu_data_prefetch_on_node(h, STARPU_MAIN_RAM, 1);
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
    /* Async RAM copy may not be valid yet; drain tries starpu_data_can_evict + evict and requeues until success. */
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
    data->graph_pending_gpu_evict_handles.clear();
    graph_sched_sync_pending_evict_count(data);
}
