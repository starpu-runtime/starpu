/* SGOC — Single-GPU offload-checkpoint graph scheduler (see graph_sched_internal.hpp). */

#include "graph_sched_internal.hpp"

#include <starpu.h>
#include <starpu_graph_recorder.h>
#include <starpu_stdlib.h>
#include <starpu_task.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

#define GRAPH_SCHED_PIN_LOG_TAG "sgoc"
#include "graph_sched_pin_worker_extract.inc"
#undef GRAPH_SCHED_PIN_LOG_TAG

/** Same rule as graph_sgoc_bundle.inc graph_sched_verbose_env (read here before bundle is included). */
static int graph_sched_verbose_starpu_env()
{
    const char *e = std::getenv("STARPU_GRAPH_SCHED_VERBOSE");
    return (e && e[0]) ? std::atoi(e) : 0;
}

/** True if per-phase capture/flush timing should go to stderr (STARPU_GRAPH_SCHED_CAPTURE_TIMING or VERBOSE>=2). */
static bool graph_sched_capture_phase_report_enabled()
{
    if (graph_sched_verbose_starpu_env() >= 2)
        return true;
    const char *e = std::getenv("STARPU_GRAPH_SCHED_CAPTURE_TIMING");
    return e && e[0] && std::atoi(e) != 0;
}

struct SgocCapturePhaseTimer {
    std::chrono::steady_clock::time_point t;
    const char *const where;
    explicit SgocCapturePhaseTimer(const char *w) : t(std::chrono::steady_clock::now()), where(w) {}
    void lap(const char *label)
    {
        if (!graph_sched_capture_phase_report_enabled())
            return;
        const auto t2 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t2 - t).count();
        t = t2;
        std::cerr << "sgoc_capture_timing: " << where << " +" << std::fixed << std::setprecision(3) << ms << " ms "
                  << label << std::endl;
    }
};

static void sgoc_fill_flush_residency_sets_from_ops(graph_sched_data *data, const std::vector<GraphOp> &ops,
                                                    unsigned gpu_mem_node_i)
{
    if (!data || !data->graph_sgoc)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    G.flush_starpu_gpu_resident.clear();
    G.flush_starpu_ram_valid_not_gpu.clear();
    const int gpu_i = static_cast<int>(gpu_mem_node_i);
    const int ram_i = static_cast<int>(STARPU_MAIN_RAM);
    std::unordered_set<void *> seen;
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(op.task);
        for (unsigned i = 0; i < nbuf; ++i) {
            starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(op.task, i);
            if (!h)
                continue;
            void *p = static_cast<void *>(h);
            if (!seen.insert(p).second)
                continue;
            int ag = 0, vg = 0, lg = 0, rg = 0;
            starpu_data_query_status2(h, gpu_i, &ag, &vg, &lg, &rg);
            if (vg)
                (void)G.flush_starpu_gpu_resident.insert(p);
            else {
                int ar = 0, vr = 0, lr = 0, rr = 0;
                starpu_data_query_status2(h, ram_i, &ar, &vr, &lr, &rr);
                if (vr)
                    (void)G.flush_starpu_ram_valid_not_gpu.insert(p);
            }
        }
    }
}


namespace graph_sgoc_bundle {
static void sgoc_rebuild_handle_access_lists(graph_sched_data *data);
void graph_sgoc_rebuild_lists_and_refresh_deps(graph_sched_data *data);
#include "graph_sgoc_bundle.inc"

static void sgoc_rebuild_handle_access_lists(graph_sched_data *data)
{
    data->graph_handle_access_lists.clear();
    for (size_t i = 0; i < data->graph_handle_accesses.size(); ++i) {
        GraphHandleAccess &a = data->graph_handle_accesses[i];
        if (!a.handle)
            continue;
        void *p = static_cast<void *>(a.handle);
        GraphHandleAccessList &L = data->graph_handle_access_lists[p];
        if (a.prev_for_handle == GRAPH_ACCESS_NONE)
            L.head = i;
        if (a.next_for_handle == GRAPH_ACCESS_NONE)
            L.tail = i;
    }
}

void graph_sgoc_rebuild_lists_and_refresh_deps(graph_sched_data *data)
{
    sgoc_rebuild_handle_access_lists(data);
    graph_sched_refresh_op_dependencies(data);
}

void graph_sgoc_finalize_outermost_capture(graph_sched_data *data, std::vector<GraphOp> &&replay,
                                           std::vector<GraphHandleAccess> &&replay_ha,
                                           unsigned added_invalidate_submit, unsigned sched_ctx_id)
{
    /* starpu_task_wait_for_all() runs at outermost recording_begin (quiesce before capture) and recording_end
     * (before linearize) so work is quiesced before this path. */

    SgocCapturePhaseTimer timer("finalize");
    graph_sched_captured_handle_groups parsed{};
    const int v = graph_sched_verbose_env();
    graph_sched_parse_captured_data_handles(replay, parsed, v);
    timer.lap("parse_captured_data_handles");
    bool has_batch = false;
    std::uint32_t batch_val = 0;
    if (!graph_sched_infer_batch_capture_context(replay, &has_batch, &batch_val))
        has_batch = false;
    timer.lap("infer_batch_capture_context");

    ::graph_sched_sgoc_release_outermost_capture(data, std::move(replay), std::move(replay_ha), parsed, has_batch,
                                                 batch_val, v, sched_ctx_id);
    timer.lap("release_outermost_capture");

    std::lock_guard<std::mutex> lock(data->policy_mutex);
    data->graph_captured_handle_groups = std::move(parsed);
    data->graph_total_synthetic_invalidate_inserts += added_invalidate_submit;
    timer.lap("store_captured_handle_groups");
}

} /* namespace graph_sgoc_bundle */

void graph_sched_sgoc_clear_runtime(graph_sched_data *data)
{
    if (!data)
        return;
    data->dbg_sgoc_pop_picked_data_ready.store(0, std::memory_order_relaxed);
    data->dbg_sgoc_pop_picked_data_not_ready.store(0, std::memory_order_relaxed);
    if (!data->graph_sgoc)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    G.pre_exec_prefetch.clear();
    G.post_exec_prefetch.clear();
    G.post_exec_offload_order.clear();
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
    G.dbg_mm_trace_post_exec_offload_tasks.store(0, std::memory_order_relaxed);
    graph_sched_sgoc_victim_clear_belady(data);
}

static size_t sgoc_count_task_ops(const std::vector<GraphOp> &ops)
{
    size_t n = 0;
    for (const GraphOp &op : ops) {
        if (op.kind == GraphOp::TASK && op.task)
            n++;
    }
    return n;
}

/** True if StarPU-reported free GPU bytes (when known) and our planner budget allow starting a transfer of \p sz. */
static bool sgoc_gpu_transfer_headroom_ok(const graph_sched_data *data, std::int64_t sz)
{
    if (!data || !data->graph_sgoc || sz <= 0)
        return true;
    const graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    const starpu_ssize_t star_avail = starpu_memory_get_available(G.gpu_mem_node);
    if (star_avail >= 0 && star_avail < sz)
        return false;
    if (G.mem_budget_bytes > 0 && G.tracked_gpu_bytes + sz > G.mem_budget_bytes)
        return false;
    return true;
}

/** Demand path: issue starpu_data_fetch_on_node (not speculative prefetch) when RAM holds a valid replica. */
static bool sgoc_try_demand_fetch_handle_to_gpu(graph_sched_data *data, starpu_data_handle_t h)
{
    if (!data || !h || !data->graph_sgoc)
        return false;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    if (!G.mm_execute)
        return true;
    graph_sched_drain_pending_gpu_evicts(data, G.gpu_mem_node);
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
        return false;
    (void)starpu_data_fetch_on_node(h, G.gpu_mem_node, 1u);
    if (G.mem_debug) {
        G.dbg_gpu_prefetch_issue.fetch_add(1u, std::memory_order_relaxed);
        G.dbg_gpu_prefetch_bytes.fetch_add(static_cast<std::uint64_t>(sz), std::memory_order_relaxed);
    }
    (void)G.tracked_gpu_resident.insert(p);
    G.tracked_gpu_bytes += sz;
    return true;
}

static void sgoc_drain_deferred_prefetch(graph_sched_data *data)
{
    if (!data || !data->graph_sgoc)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    graph_sched_drain_pending_gpu_evicts(data, G.gpu_mem_node);
    size_t guard = 0;
    while (!G.deferred_prefetch.empty() && guard++ < G.deferred_prefetch.size() + 8u) {
        starpu_data_handle_t h = G.deferred_prefetch.front();
        G.deferred_prefetch.pop_front();
        if (!sgoc_try_demand_fetch_handle_to_gpu(data, h))
            G.deferred_prefetch.push_back(h);
    }
}

void graph_sched_sgoc_pre_exec_hook(graph_sched_data *data, struct starpu_task *task)
{
    if (!data || !task || !data->graph_sgoc || !data->graph_sgoc->mm_execute)
        return;
    sgoc_drain_deferred_prefetch(data);
}

void graph_sched_sgoc_pop_prefetch_hook(graph_sched_data *data, struct starpu_task *task)
{
    if (!data || !task || !data->graph_sgoc || !data->graph_sgoc->mm_execute)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    const graph_sched_gpu_memory_manager &mm = data->graph_gpu_mm;

    /* MM plan: GPU demand-fetch handles assigned to this anchor topo slot (may be several slots before consumer). */
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

void graph_sched_sgoc_post_exec_hook(graph_sched_data *data, struct starpu_task *task, unsigned gpu_mem_node)
{
    if (!data || !task)
        return;
    graph_sched_run_post_exec_offloads(data, task, gpu_mem_node);
    graph_sched_run_post_exec_evict_gpu_only(data, task, gpu_mem_node);
    if (data->graph_sgoc && data->graph_sgoc->mm_execute)
        sgoc_drain_deferred_prefetch(data);
}

void graph_sched_sgoc_print_memory_observations(graph_sched_data *data)
{
    if (!data || !data->graph_sgoc)
        return;
    const bool md = graph_sgoc_bundle::graph_sched_sgoc_mem_debug_env() != 0;
    const bool tr = graph_sgoc_bundle::graph_sched_mm_order_trace_env() != 0;
    if (!md && !tr)
        return;
    /* Do not call starpu_task_wait_for_all() here: this runs from sched deinit inside starpu_shutdown(), when
     * StarPU barrier/wait machinery may already be torn down (pthread mutex invalid). */
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    const graph_sched_gpu_memory_manager &mm = data->graph_gpu_mm;
    if (md || tr)
        std::cerr << "sgoc_mem_debug: deinit memory summary (counters as of sched teardown; call "
                     "starpu_task_wait_for_all before shutdown if you need post-replay quiescence)" << std::endl;
    if (md && G.mm_obs_last_flush_valid)
        graph_sgoc_bundle::graph_sched_sgoc_log_mm_plan_advance_debug(G.mm_obs_last_topo_slots, mm);
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
                  << " post_exec_offload_nonempty_hooks=" << G.dbg_mm_trace_post_exec_offload_tasks.load()
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

void graph_sched_sgoc_release_outermost_capture(graph_sched_data *data, std::vector<GraphOp> replay,
                                                std::vector<GraphHandleAccess> replay_ha,
                                                graph_sched_captured_handle_groups &parsed, bool has_batch,
                                                std::uint32_t batch_val, int vb, unsigned sched_ctx_id)
{
    /* Flush pipeline (post recording_end linearize): replay holds dense graph_ops + handle_accesses with
     * capture-time synthetic pre-W invalidates already materialized as INVALIDATE ops.
     *  (1) Install ops/HA, rebuild handle lists, refresh pred/succ (idempotent if linearize already refreshed).
     *  (2) WRR checkpoints (STARPU_GRAPH_SCHED_CHECKPOINT_MAX): invalidate + cloned producer before backward read.
     *  (3) StarPU residency snapshot for optional ready-VRAM topo tie-break.
     *  (4) Lex topo + memory-peak sim on that order (diagnostics / legacy greedy trigger).
     *  (5) Exec topo: ready-set greedy VRAM order (default) or lex + greedy-memory fallback.
     *  (6) MM Belady linear plan + runtime victim Belady — both consume post-checkpoint \a graph_ops and \a topo_order
     *      (extra TASK slots from rematerialization clones affect appearances / greedy choices). */
    SgocCapturePhaseTimer flush_timer("flush");
    (void)has_batch;
    (void)batch_val;
    (void)sched_ctx_id;
    if (!data)
        return;
    if (sgoc_count_task_ops(replay) == 0) {
        flush_timer.lap("early_exit_empty_task_ops");
        if (vb >= 2)
            std::cerr << "sgoc: outermost_capture_skip_empty_graph (no TASK ops)\n";
        if (graph_sgoc_bundle::graph_sched_sgoc_mem_debug_env())
            std::cerr << "sgoc_mem_debug: skipped flush (capture has no TASK ops); MM logs only run after non-empty "
                         "graph capture\n"
                      << std::flush;
        if (data->graph_sgoc)
            data->graph_sgoc->mm_obs_last_flush_valid = false;
        return;
    }

    data->graph_mem_offload_plan.valid = false;
    flush_timer.lap("init_offload_plan_invalid");

    if (!data->graph_sgoc)
        data->graph_sgoc = std::make_unique<graph_sched_data::graph_sgoc_runtime>();
    else
        graph_sched_sgoc_clear_runtime(data);
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;

    {
        std::lock_guard<std::mutex> lk(data->policy_mutex);
        data->graph_ops = std::move(replay);
        data->graph_handle_accesses = std::move(replay_ha);
        graph_sgoc_bundle::sgoc_rebuild_handle_access_lists(data);
        graph_sgoc_bundle::graph_sched_refresh_op_dependencies(data);
    }
    flush_timer.lap("move_ops_rebuild_access_deps");

    graph_sgoc_bundle::graph_sgoc_apply_wrr_checkpoints_before_topo(data, parsed, vb);
    flush_timer.lap("wrr_checkpoints_before_topo");

    const int pin_worker = data->graph_pinned_worker_id;
    std::vector<void *> s_offload_active;

    if (pin_worker >= 0) {
        const unsigned gpu_node_fill = starpu_worker_get_memory_node(static_cast<unsigned>(pin_worker));
        sgoc_fill_flush_residency_sets_from_ops(data, data->graph_ops, gpu_node_fill);
    } else {
        G.flush_starpu_gpu_resident.clear();
        G.flush_starpu_ram_valid_not_gpu.clear();
    }
    flush_timer.lap("residency_snapshot_starpu");

    const std::unordered_set<void *> *const starpu_truth =
        (pin_worker >= 0) ? &G.flush_starpu_gpu_resident : nullptr;

    std::vector<size_t> topo_lex;
    graph_sgoc_bundle::graph_sched_compute_topological_order(data->graph_ops, topo_lex);
    flush_timer.lap("compute_topo_lex");
    size_t mem_peak_topo_i = 0;
    std::int64_t mem_peak_bytes = 0;
    std::int64_t mem_initial_bytes = 0;
    size_t mem_initial_live_handles = 0;
    graph_sgoc_bundle::graph_sched_compute_memory_after_ops(data->graph_ops, data->graph_handle_accesses, topo_lex,
                                                            &mem_peak_topo_i, &mem_peak_bytes, &mem_initial_bytes,
                                                            &mem_initial_live_handles, vb >= 6);
    flush_timer.lap("compute_memory_peak_simulation");

    std::int64_t mem_budget = data->graph_pinned_worker_max_allowed_memory_bytes;
    const std::int64_t forced = graph_sgoc_bundle::graph_sched_force_mem_budget_bytes_env();
    if (forced >= 0)
        mem_budget = forced;
    if (mem_budget > 0)
        mem_budget = static_cast<std::int64_t>(static_cast<double>(mem_budget)
                                                 * graph_sgoc_bundle::graph_sched_mem_budget_fraction_env());
    {
        const std::int64_t sgoc_b = graph_sgoc_bundle::graph_sched_sgoc_budget_bytes_env();
        if (sgoc_b >= 0)
            mem_budget = sgoc_b;
    }
    flush_timer.lap("derive_mem_budget");

    std::vector<size_t> topo_order;
    if (graph_sgoc_bundle::graph_sgoc_ready_vram_topo_enabled()) {
        graph_sgoc_bundle::graph_sgoc_compute_ready_set_greedy_vram_topological_order(
            data->graph_ops, data->graph_handle_accesses, starpu_truth, topo_order, vb);
        if (vb >= 2)
            std::cerr << "sgoc: ready_set_stateful_vram_topo len=" << topo_order.size() << " mem_lex_peak_bytes="
                      << mem_peak_bytes << " budget=" << mem_budget << std::endl;
    } else {
        topo_order = topo_lex;
        if (mem_budget > 0 && mem_peak_bytes > mem_budget && graph_sgoc_bundle::graph_sched_linear_replay_greedy_enabled()) {
            if (vb >= 2)
                std::cerr << "sgoc: lexicographic topo peak_bytes=" << mem_peak_bytes << " > budget=" << mem_budget
                          << " - switching to greedy memory topological order\n";
            graph_sgoc_bundle::graph_sched_compute_greedy_memory_topological_order(data->graph_ops, topo_order, nullptr,
                                                                                   nullptr, nullptr, nullptr);
        }
    }
    flush_timer.lap("compute_exec_topo_order");

    const int mm_offload_auto = graph_sgoc_bundle::graph_sched_mem_offload_auto_env();
    if (mm_offload_auto && pin_worker >= 0)
        graph_sgoc_bundle::graph_sched_apply_gpu_mm_plan_from_capture(
            data->graph_ops, data->graph_handle_accesses, topo_order, data, &parsed, pin_worker, vb, false,
            true, s_offload_active, starpu_truth);
    flush_timer.lap("gpu_mm_plan_from_capture");

    G.mm_execute = graph_sgoc_bundle::graph_sched_mm_execute_hints_env();
    G.mem_debug = graph_sgoc_bundle::graph_sched_sgoc_mem_debug_env();
    G.mm_order_trace = graph_sgoc_bundle::graph_sched_mm_order_trace_env();
    if (pin_worker >= 0)
        G.gpu_mem_node = starpu_worker_get_memory_node(static_cast<unsigned>(pin_worker));
    G.mem_budget_bytes = mem_budget;
    graph_sched_sgoc_victim_rebuild_belady(data, topo_order);
    if (G.mm_execute && pin_worker >= 0) {
        const starpu_ssize_t used = starpu_memory_get_used(G.gpu_mem_node);
        if (used >= 0)
            G.tracked_gpu_bytes = static_cast<std::int64_t>(used);
        else
            G.tracked_gpu_bytes = mem_initial_bytes;
    }
    flush_timer.lap("runtime_mm_init_victim_tracked_bytes");

    const graph_sched_gpu_memory_manager &mm = data->graph_gpu_mm;

    G.replay_task_topo_slot.clear();
    for (size_t ti = 0; ti < topo_order.size(); ++ti) {
        const size_t opi = topo_order[ti];
        if (opi >= data->graph_ops.size())
            continue;
        const GraphOp &op = data->graph_ops[opi];
        if (op.kind == GraphOp::TASK && op.task)
            G.replay_task_topo_slot[op.task] = static_cast<unsigned>(ti);
    }

    const graph_sched_replay_accounting_scope replay_scope{data};
    _starpu_graph_recorder_set_flushing(1);

    std::unordered_map<const struct starpu_codelet *, bool> pin_cl_runnable;
    std::unordered_map<const struct starpu_codelet *, bool> *pin_cl_cache = nullptr;
    if (pin_worker >= 0) {
        size_t n_task_ops = 0;
        for (const GraphOp &o : data->graph_ops) {
            if (o.kind == GraphOp::TASK)
                n_task_ops++;
        }
        pin_cl_runnable.reserve(std::max<size_t>(16, n_task_ops / 4));
        pin_cl_cache = &pin_cl_runnable;
    }
    flush_timer.lap("before_replay_submit_loop");

    size_t n_task_submitted = 0;
    for (size_t ti = 0; ti < topo_order.size(); ++ti) {
        const size_t op_idx = topo_order[ti];
        GraphOp &op = data->graph_ops[op_idx];
        switch (op.kind) {
        case GraphOp::TASK:
            if (pin_worker >= 0)
                graph_sgoc_bundle::graph_sched_apply_replay_worker_pin(op.task, pin_worker, vb, pin_cl_cache);
            _starpu_task_insert_submit_built_task(op.task);
            n_task_submitted++;
            if (G.mm_execute && pin_worker >= 0 && ti < mm.topo_post_exec_offload_order.size()
                && !mm.topo_post_exec_offload_order[ti].empty() && op.task)
                graph_sched_register_offload_after_task(data, op.task, mm.topo_post_exec_offload_order[ti]);
            if (G.mm_execute && pin_worker >= 0 && ti < mm.topo_post_exec_evict_gpu_only_order.size()
                && !mm.topo_post_exec_evict_gpu_only_order[ti].empty() && op.task)
                graph_sched_register_evict_gpu_only_after_task(data, op.task,
                                                               mm.topo_post_exec_evict_gpu_only_order[ti]);
            break;
        case GraphOp::INVALIDATE:
            _starpu_data_invalidate_submit_impl(op.handle);
            break;
        }
    }
    flush_timer.lap(("replay_submit n_task=" + std::to_string(n_task_submitted) + " n_topo_slots=" + std::to_string(topo_order.size())).c_str());

    if (pin_worker >= 0)
        graph_sched_drain_deferred_ram_offload_copies(data, G.gpu_mem_node);

    G.mm_obs_last_flush_valid = true;
    G.mm_obs_last_topo_slots = topo_order.size();
    G.mm_obs_last_replay_tasks_submitted = n_task_submitted;
    G.mm_obs_last_mem_offload_auto = mm_offload_auto;
    G.mm_obs_last_pin_worker = pin_worker;
    G.mm_obs_last_mm_execute = G.mm_execute;

    _starpu_graph_recorder_set_flushing(0);

    {
        std::lock_guard<std::mutex> lk(data->policy_mutex);
        data->graph_ops.clear();
        data->graph_handle_accesses.clear();
        data->graph_handle_access_lists.clear();
    }
    flush_timer.lap("clear_ops_under_policy_mutex");
}

namespace {

static int sgoc_capture_task_hook(struct starpu_task *task, void *arg)
{
    auto *d = static_cast<graph_sched_data *>(arg);
    std::lock_guard<std::mutex> lock(d->policy_mutex);
    if (d->graph_record_nested == 0)
        return -1;
    graph_sgoc_bundle::graph_sched_append_captured_task(d, task);
    return 0;
}

static int sgoc_capture_invalidate_hook(starpu_data_handle_t handle, void *arg)
{
    auto *d = static_cast<graph_sched_data *>(arg);
    std::lock_guard<std::mutex> lock(d->policy_mutex);
    if (d->graph_record_nested == 0)
        return -1;
    graph_sgoc_bundle::graph_sched_append_capture_explicit_invalidate(d, handle);
    return 0;
}

} /* namespace */

void graph_sched_sgoc_register(graph_sched_data *data)
{
    _starpu_graph_recorder_register(sgoc_capture_task_hook, sgoc_capture_invalidate_hook, nullptr, data);
}

void graph_sched_sgoc_deinit(graph_sched_data *data, unsigned sched_ctx_id)
{
    for (;;) {
        SgocCapturePhaseTimer td("deinit_flush");
        std::vector<GraphOp> replay;
        std::vector<GraphHandleAccess> replay_handle_accesses;
        unsigned added_invalidate_submit = 0;
        bool moved_capture = false;
        {
            std::unique_lock<std::mutex> lock(data->policy_mutex);
            td.lap("policy_mutex_locked");
            if (data->graph_record_nested == 0)
                break;
            data->graph_record_nested--;
            if (data->graph_record_nested == 0) {
                lock.unlock();
                td.lap("policy_unlock_before_wait_for_all");
                starpu_task_wait_for_all();
                td.lap("starpu_task_wait_for_all");
                lock.lock();
                td.lap("policy_relock_after_wait");
                graph_sched_account_outermost_capture_end(data);
                td.lap("account_capture_wall_time");
                moved_capture = true;
                added_invalidate_submit = data->graph_added_invalidate_submit;
                graph_sgoc_bundle::graph_sgoc_linearize_capture_to_ops(data);
                td.lap("after_linearize_see_linearize_line");
                replay = std::move(data->graph_ops);
                replay_handle_accesses = std::move(data->graph_handle_accesses);
                data->graph_handle_accesses.clear();
                data->graph_handle_access_lists.clear();
            }
        }
        td.lap("policy_mutex_scope_done");
        if (moved_capture) {
            graph_sgoc_bundle::graph_sgoc_finalize_outermost_capture(data, std::move(replay),
                                                                     std::move(replay_handle_accesses),
                                                                     added_invalidate_submit, sched_ctx_id);
            td.lap("finalize_outermost_capture_done");
        }
        _starpu_graph_recording_pop();
        td.lap("graph_recording_pop");
    }
    _starpu_graph_recorder_unregister(data);
}

void graph_sched_account_outermost_capture_end(graph_sched_data *data)
{
    const auto t_end = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t_end - data->graph_capture_wall_start).count();
    const std::uint64_t ns = static_cast<std::uint64_t>(sec * 1e9);
    data->graph_sched_graph_capture_wall_time_ns.fetch_add(ns, std::memory_order_relaxed);
    data->graph_sched_graph_capture_sessions.fetch_add(1u, std::memory_order_relaxed);
    if (graph_sgoc_bundle::graph_sched_verbose_env() >= 2) {
        const std::ios::fmtflags ff = std::cerr.flags();
        std::cerr << std::fixed << std::setprecision(6) << (data->policy_log_name ? data->policy_log_name : "graph")
                  << ": graph_capture_wall_sec=" << sec
                  << " (outermost: wall clock after begin wait at recording_begin through account at recording_end "
                     "after end wait; excludes both waits and flush replay/planning)" << std::endl;
        std::cerr.flags(ff);
    }
}

extern "C" {

void starpu_graph_sched_graph_recording_begin(unsigned sched_ctx_id)
{
    graph_sched_data *data = graph_sched_graph_policy_data(sched_ctx_id);
    if (!data)
        return;

    _starpu_graph_recording_push();

    std::unique_lock<std::mutex> lock(data->policy_mutex);
    if (data->graph_record_nested == 0) {
        lock.unlock();
        const auto w0 = std::chrono::steady_clock::now();
        starpu_task_wait_for_all();
        const auto w1 = std::chrono::steady_clock::now();
        const double wait_ms = std::chrono::duration<double, std::milli>(w1 - w0).count();
        if (graph_sched_capture_phase_report_enabled()) {
            std::cerr << "sgoc_capture_timing: recording_begin starpu_task_wait_for_all +" << std::fixed
                      << std::setprecision(3) << wait_ms << " ms" << std::endl;
        }
        lock.lock();
        data->graph_capture_wall_start = std::chrono::steady_clock::now();
        data->graph_ops.clear();
        data->graph_handle_accesses.clear();
        data->graph_handle_access_lists.clear();
        data->graph_added_invalidate_submit = 0;
        data->graph_idempotent_tasks_sorted.clear();
        data->graph_captured_handle_groups = {};
        graph_sched_clear_offload_task_registrations(data);
        if (!data->graph_sgoc)
            data->graph_sgoc = std::make_unique<graph_sched_data::graph_sgoc_runtime>();
        else
            graph_sched_sgoc_clear_runtime(data);
    }
    data->graph_record_nested++;
}

void starpu_graph_sched_graph_recording_end(unsigned sched_ctx_id)
{
    graph_sched_data *data = graph_sched_graph_policy_data(sched_ctx_id);
    if (!data)
        return;

    std::vector<GraphOp> replay;
    std::vector<GraphHandleAccess> replay_handle_accesses;
    bool outermost_end = false;
    unsigned added_invalidate_submit = 0;
    {
        std::unique_lock<std::mutex> lock(data->policy_mutex);
        if (data->graph_record_nested == 0)
            return;

        SgocCapturePhaseTimer top("recording_end");
        top.lap("policy_mutex_locked");

        data->graph_record_nested--;
        if (data->graph_record_nested == 0) {
            /* Quiesce before linearizing capture or querying StarPU (plan §1); do not hold policy_mutex across wait. */
            lock.unlock();
            top.lap("policy_unlock_before_wait_for_all");
            starpu_task_wait_for_all();
            top.lap("starpu_task_wait_for_all");
            lock.lock();
            top.lap("policy_relock_after_wait");
            graph_sched_account_outermost_capture_end(data);
            top.lap("account_capture_wall_time");
            added_invalidate_submit = data->graph_added_invalidate_submit;
            graph_sgoc_bundle::graph_sgoc_linearize_capture_to_ops(data);
            top.lap("after_linearize_see_linearize_line");
            replay = std::move(data->graph_ops);
            replay_handle_accesses = std::move(data->graph_handle_accesses);
            data->graph_handle_accesses.clear();
            data->graph_handle_access_lists.clear();
            outermost_end = true;
        }
        top.lap("before_policy_mutex_release");
    }

    SgocCapturePhaseTimer tail("recording_end_tail");
    tail.lap("policy_mutex_released");
    if (outermost_end) {
        graph_sgoc_bundle::graph_sgoc_finalize_outermost_capture(data, std::move(replay),
                                                                 std::move(replay_handle_accesses),
                                                                 added_invalidate_submit, sched_ctx_id);
        tail.lap("finalize_outermost_capture_done");
    }

    _starpu_graph_recording_pop();
    tail.lap("graph_recording_pop");
}

} /* extern "C" */
