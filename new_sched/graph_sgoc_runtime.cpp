/* SGOC runtime hooks: GPU demand fetch, deferred prefetch, clear runtime state. */

#include "graph_sgoc_internal.hpp"

#include <starpu.h>
#include <starpu_data.h>
#include <starpu_task.h>

#include <deque>
#include <iostream>

void graph_sgoc_clear_runtime(graph_sgoc_data *data)
{
    if (!data)
        return;
    data->dbg_sgoc_pop_picked_data_ready.store(0, std::memory_order_relaxed);
    data->dbg_sgoc_pop_picked_data_not_ready.store(0, std::memory_order_relaxed);
    if (!data->graph_sgoc)
        return;
    graph_sgoc_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    G.replay_task_topo_slot.clear();
    G.deferred_prefetch.clear();
    G.tracked_gpu_resident.clear();
    G.tracked_gpu_bytes = 0;
    G.flush_starpu_gpu_resident.clear();
    G.flush_starpu_ram_valid_not_gpu.clear();
    G.capture_ops.clear();
    G.capture_id_to_iter.clear();
    G.capture_next_stable_id = 1;
    G.mem_debug = 0;
    G.mm_order_trace = 0;
    G.dbg_offload_ram_issue.store(0, std::memory_order_relaxed);
    G.dbg_offload_ram_bytes.store(0, std::memory_order_relaxed);
    G.dbg_gpu_prefetch_issue.store(0, std::memory_order_relaxed);
    G.dbg_gpu_prefetch_bytes.store(0, std::memory_order_relaxed);
    G.dbg_evict_ok.store(0, std::memory_order_relaxed);
    G.dbg_mm_trace_offload_regs.store(0, std::memory_order_relaxed);
    G.dbg_mm_trace_anchor_fetch_try.store(0, std::memory_order_relaxed);
    G.dbg_mm_trace_taskbuf_fetch_try.store(0, std::memory_order_relaxed);
    G.dbg_mm_trace_followup_pop_offload_tasks.store(0, std::memory_order_relaxed);
    graph_sgoc_victim_clear_belady(data);
}

namespace {

static bool sgoc_gpu_transfer_headroom_ok(const graph_sgoc_data *data, std::int64_t sz)
{
    if (!data || !data->graph_sgoc || sz <= 0)
        return true;
    const graph_sgoc_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    const starpu_ssize_t star_avail = starpu_memory_get_available(G.gpu_mem_node);
    if (star_avail >= 0 && star_avail < sz)
        return false;
    if (G.mem_budget_bytes > 0 && G.tracked_gpu_bytes + sz > G.mem_budget_bytes)
        return false;
    return true;
}

static bool sgoc_try_demand_fetch_handle_to_gpu(graph_sgoc_data *data, starpu_data_handle_t h)
{
    if (!data || !h || !data->graph_sgoc)
        return false;
    graph_sgoc_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    if (!G.mm_execute)
        return true;
    graph_sgoc_drain_pending_gpu_evicts(data, G.gpu_mem_node);
    void *p = static_cast<void *>(h);
    if (G.tracked_gpu_resident.count(p))
        return true;
    const std::int64_t sz = static_cast<std::int64_t>(starpu_data_get_size(h));
    const int gpu_i = static_cast<int>(G.gpu_mem_node);
    int a = 0, v = 0, loading = 0, req = 0;
    starpu_data_query_status2(h, gpu_i, &a, &v, &loading, &req);
    if (v) {
        (void)G.tracked_gpu_resident.insert(p);
        G.tracked_gpu_bytes += sz;
        return true;
    }
    const int ram = static_cast<int>(STARPU_MAIN_RAM);
    starpu_data_query_status2(h, ram, &a, &v, &loading, &req);
    if (!v)
        return false;
    if (!sgoc_gpu_transfer_headroom_ok(data, sz))
    {
        starpu_memchunk_tidy(G.gpu_mem_node);
    }
    if (!sgoc_gpu_transfer_headroom_ok(data, sz))
    {
        return false;
    }
    (void)starpu_data_fetch_on_node(h, G.gpu_mem_node, 1u);
    // (void)starpu_data_prefetch_on_node(h, G.gpu_mem_node, 1u);
    if (G.mem_debug) {
        G.dbg_gpu_prefetch_issue.fetch_add(1u, std::memory_order_relaxed);
        G.dbg_gpu_prefetch_bytes.fetch_add(static_cast<std::uint64_t>(sz), std::memory_order_relaxed);
    }
    (void)G.tracked_gpu_resident.insert(p);
    G.tracked_gpu_bytes += sz;
    return true;
}

static void sgoc_drain_deferred_prefetch(graph_sgoc_data *data)
{
    if (!data || !data->graph_sgoc)
        return;
    graph_sgoc_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    graph_sgoc_drain_pending_gpu_evicts(data, G.gpu_mem_node);
    size_t guard = 0;
    while (!G.deferred_prefetch.empty() && guard++ < G.deferred_prefetch.size() + 8u) {
        starpu_data_handle_t h = G.deferred_prefetch.front();
        G.deferred_prefetch.pop_front();
        if (!sgoc_try_demand_fetch_handle_to_gpu(data, h))
            G.deferred_prefetch.push_back(h);
    }
}

} /* namespace */

void graph_sgoc_pre_exec_hook(graph_sgoc_data *data, struct starpu_task *task)
{
    if (!data || !task || !data->graph_sgoc || !data->graph_sgoc->mm_execute)
        return;
    sgoc_drain_deferred_prefetch(data);
}

void graph_sgoc_pop_prefetch_hook(graph_sgoc_data *data, struct starpu_task *task)
{
    if (!data || !task || !data->graph_sgoc || !data->graph_sgoc->mm_execute)
        return;
    graph_sgoc_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    const graph_sgoc_gpu_memory_manager &mm = data->graph_gpu_mm;

    const auto it_anchor = G.replay_task_topo_slot.find(task);
    if (it_anchor != G.replay_task_topo_slot.end()) {
        const unsigned ti = it_anchor->second;
        if (ti < mm.topo_pre_exec_prefetch_order.size()) {
            for (void *hv : mm.topo_pre_exec_prefetch_order[ti]) {
                starpu_data_handle_t h = static_cast<starpu_data_handle_t>(hv);
                if (!h)
                    continue;
                if (G.mm_order_trace)
                    G.dbg_mm_trace_anchor_fetch_try.fetch_add(1u, std::memory_order_relaxed);
                if (!sgoc_try_demand_fetch_handle_to_gpu(data, h))
                    G.deferred_prefetch.push_back(h);
            }
        }
    }

    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    for (unsigned j = 0; j < nbuf; ++j) {
        const enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, j);
        if ((mode & STARPU_SCRATCH) || (mode & STARPU_REDUX))
            continue;
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(task, j);
        if (!h)
            continue;
        if (G.mm_order_trace)
            G.dbg_mm_trace_taskbuf_fetch_try.fetch_add(1u, std::memory_order_relaxed);
        if (!sgoc_try_demand_fetch_handle_to_gpu(data, h))
            G.deferred_prefetch.push_back(h);
    }
    sgoc_drain_deferred_prefetch(data);
}

void graph_sgoc_post_exec_hook(graph_sgoc_data *data, struct starpu_task *task, unsigned gpu_mem_node)
{
    (void)gpu_mem_node;
    if (!data || !task)
        return;
    if (data->graph_sgoc && data->graph_sgoc->mm_execute)
        sgoc_drain_deferred_prefetch(data);
}

void graph_sgoc_print_memory_observations(graph_sgoc_data *data)
{
    if (!data || !data->graph_sgoc)
        return;
    const bool md = graph_sgoc_bundle::graph_sgoc_mem_debug_env() != 0;
    const bool tr = graph_sgoc_bundle::graph_sgoc_mm_order_trace_env() != 0;
    if (!md && !tr)
        return;
    graph_sgoc_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    const graph_sgoc_gpu_memory_manager &mm = data->graph_gpu_mm;
    if (md || tr) {
        std::cerr << "sgoc_mem_debug: deinit memory summary (counters as of sched teardown; call "
                     "starpu_task_wait_for_all before shutdown if you need post-replay quiescence)"
                  << " wrr_checkpoint_inserts_total=" << data->graph_total_checkpoint_inserts << std::endl;
    }
    if (md && G.mm_obs_last_flush_valid)
        graph_sgoc_bundle::graph_sgoc_log_mm_plan_advance_debug(G.mm_obs_last_topo_slots, mm);
    else if (md && !G.mm_obs_last_flush_valid)
        std::cerr << "sgoc_mem_debug: mm_plan (skipped: no non-empty flush snapshot in this session)" << std::endl;

    if (tr && G.mm_obs_last_flush_valid) {
        size_t ofs_slots = 0, ofs_refs = 0;
        for (size_t i = 0; i < mm.topo_post_exec_offload_order.size(); ++i) {
            const auto &v = mm.topo_post_exec_offload_order[i];
            if (!v.empty()) {
                ++ofs_slots;
                ofs_refs += v.size();
            }
        }
        size_t anch_slots = 0, anch_refs = 0;
        for (size_t i = 0; i < mm.topo_pre_exec_prefetch_order.size(); ++i) {
            const auto &v = mm.topo_pre_exec_prefetch_order[i];
            if (!v.empty()) {
                ++anch_slots;
                anch_refs += v.size();
            }
        }
        size_t pfb_slots = 0, pfb_refs = 0;
        for (size_t i = 0; i < mm.topo_prefetch_before_task.size(); ++i) {
            const auto &v = mm.topo_prefetch_before_task[i];
            if (!v.empty()) {
                ++pfb_slots;
                pfb_refs += v.size();
            }
        }
        std::cerr << "sgoc_mm_order_trace: plan offload_nonempty_topo_slots=" << ofs_slots << " offload_handle_refs="
                  << ofs_refs << " anchor_prefetch_nonempty_topo_slots=" << anch_slots << " anchor_prefetch_handle_refs="
                  << anch_refs << " sim_consumer_prefetch_nonempty_topo_slots=" << pfb_slots
                  << " sim_consumer_prefetch_handle_refs=" << pfb_refs << " | exec mm_execute=" << G.mm_obs_last_mm_execute
                  << " mem_offload_auto=" << G.mm_obs_last_mem_offload_auto
                  << " pin_cuda_worker=" << G.mm_obs_last_pin_worker << " replay_topo_slots=" << G.mm_obs_last_topo_slots
                  << " replay_tasks_submitted=" << G.mm_obs_last_replay_tasks_submitted
                  << " registered_offload_handle_refs=" << G.dbg_mm_trace_offload_regs.load()
                  << " followup_pop_offload_nonempty_hooks=" << G.dbg_mm_trace_followup_pop_offload_tasks.load()
                  << " pop_anchor_fetch_tries=" << G.dbg_mm_trace_anchor_fetch_try.load()
                  << " pop_taskbuf_fetch_tries=" << G.dbg_mm_trace_taskbuf_fetch_try.load()
                  << " | starpu_data_fetch_on_node_calls=" << G.dbg_gpu_prefetch_issue.load()
                  << " ram_replicate_or_skip_ok=" << G.dbg_offload_ram_issue.load()
                  << " gpu_evict_ok=" << G.dbg_evict_ok.load() << std::endl;
    }

    if (md) {
        std::cerr << "sgoc_mem_debug: replay ram_offload_starpu_calls=" << G.dbg_offload_ram_issue.load()
                  << " ram_offload_bytes=" << G.dbg_offload_ram_bytes.load()
                  << " gpu_fetch_starpu_calls=" << G.dbg_gpu_prefetch_issue.load()
                  << " gpu_fetch_bytes=" << G.dbg_gpu_prefetch_bytes.load()
                  << " gpu_evict_ok=" << G.dbg_evict_ok.load() << " mm_execute=" << G.mm_execute << std::endl;
    }
    if (md || tr) {
        std::cerr << "sgoc_mem_debug: pop_task pinned_worker data_ready_tasks=" << data->dbg_sgoc_pop_picked_data_ready.load()
                  << " data_not_ready_tasks=" << data->dbg_sgoc_pop_picked_data_not_ready.load()
                  << " (starpu_st_non_ready_buffers_size: ready => non_ready==0 && non_allocated==0)" << std::endl;
    }
}
