/* Flush replay: residency snapshot, topo/MM, submit; finalize outermost capture. */
#include "graph_sgoc_common.hpp"
#include "graph_sgoc_bundle_dag.hpp"
#include "graph_sgoc_bundle_mm.hpp"
#include "graph_sgoc_bundle_parse.hpp"
#include "graph_sgoc_bundle_topo.hpp"

namespace graph_sgoc_bundle {

static void sgoc_rebuild_handle_access_lists(graph_sgoc_data *data)
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

void graph_sgoc_rebuild_lists_and_refresh_deps(graph_sgoc_data *data)
{
    sgoc_rebuild_handle_access_lists(data);
    graph_sgoc_refresh_op_dependencies(data);
}

} /* namespace graph_sgoc_bundle */

static size_t sgoc_count_task_ops(const std::vector<GraphOp> &ops)
{
    size_t n = 0;
    for (const GraphOp &op : ops) {
        if (op.kind == GraphOp::TASK && op.task)
            n++;
    }
    return n;
}

/** Next TASK in replay topo after slot \p ti_producer (INVALIDATE slots skipped). */
static struct starpu_task *sgoc_next_task_after_topo_slot(const graph_sgoc_data *data,
                                                         const std::vector<size_t> &topo_order, size_t ti_producer)
{
    if (!data)
        return nullptr;
    for (size_t t = ti_producer + 1; t < topo_order.size(); ++t) {
        const size_t opi = topo_order[t];
        if (opi >= data->graph_ops.size())
            continue;
        const GraphOp &o = data->graph_ops[opi];
        if (o.kind == GraphOp::TASK && o.task)
            return o.task;
    }
    return nullptr;
}

static void sgoc_fill_flush_residency_sets_from_ops(graph_sgoc_data *data, const std::vector<GraphOp> &ops,
                                                    unsigned gpu_mem_node_i)
{
    if (!data || !data->graph_sgoc)
        return;
    graph_sgoc_data::graph_sgoc_runtime &G = *data->graph_sgoc;
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

void graph_sgoc_release_outermost_capture(graph_sgoc_data *data, std::vector<GraphOp> replay,
                                                std::vector<GraphHandleAccess> replay_ha,
                                                graph_sgoc_captured_handle_groups &parsed, bool has_batch,
                                                std::uint32_t batch_val, int vb, unsigned sched_ctx_id)
{
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
        if (graph_sgoc_bundle::graph_sgoc_mem_debug_env())
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
        data->graph_sgoc = std::make_unique<graph_sgoc_data::graph_sgoc_runtime>();
    else
        graph_sgoc_clear_runtime(data);
    graph_sgoc_data::graph_sgoc_runtime &G = *data->graph_sgoc;

    {
        std::lock_guard<std::mutex> lk(data->policy_mutex);
        data->graph_ops = std::move(replay);
        data->graph_handle_accesses = std::move(replay_ha);
        graph_sgoc_bundle::graph_sgoc_rebuild_lists_and_refresh_deps(data);
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
    graph_sgoc_bundle::graph_sgoc_compute_topological_order(data->graph_ops, topo_lex);
    flush_timer.lap("compute_topo_lex");
    size_t mem_peak_topo_i = 0;
    std::int64_t mem_peak_bytes = 0;
    std::int64_t mem_initial_bytes = 0;
    size_t mem_initial_live_handles = 0;
    graph_sgoc_bundle::graph_sgoc_compute_memory_after_ops(data->graph_ops, data->graph_handle_accesses, topo_lex,
                                                            &mem_peak_topo_i, &mem_peak_bytes, &mem_initial_bytes,
                                                            &mem_initial_live_handles, vb >= 6);
    flush_timer.lap("compute_memory_peak_simulation");

    std::int64_t mem_budget = data->graph_pinned_worker_max_allowed_memory_bytes;
    const std::int64_t forced = graph_sgoc_bundle::graph_sgoc_force_mem_budget_bytes_env();
    if (forced >= 0)
        mem_budget = forced;
    if (mem_budget > 0)
        mem_budget = static_cast<std::int64_t>(static_cast<double>(mem_budget)
                                               * graph_sgoc_bundle::graph_sgoc_mem_budget_fraction_env());
    {
        const std::int64_t sgoc_b = graph_sgoc_bundle::graph_sgoc_budget_bytes_env();
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
        if (mem_budget > 0 && mem_peak_bytes > mem_budget && graph_sgoc_bundle::graph_sgoc_linear_replay_greedy_enabled()) {
            if (vb >= 2)
                std::cerr << "sgoc: lexicographic topo peak_bytes=" << mem_peak_bytes << " > budget=" << mem_budget
                          << " - switching to greedy memory topological order\n";
            graph_sgoc_bundle::graph_sgoc_compute_greedy_memory_topological_order(
                data->graph_ops, topo_order, nullptr, nullptr, nullptr, nullptr, nullptr);
        }
    }
    flush_timer.lap("compute_exec_topo_order");

    const int mm_offload_auto = graph_sgoc_bundle::graph_sgoc_mem_offload_auto_env();
    if (mm_offload_auto && pin_worker >= 0)
        graph_sgoc_bundle::graph_sgoc_apply_gpu_mm_plan_from_capture(
            data->graph_ops, data->graph_handle_accesses, topo_order, data, &parsed, pin_worker, vb, false, true,
            s_offload_active, starpu_truth);
    flush_timer.lap("gpu_mm_plan_from_capture");

    G.mm_execute = graph_sgoc_bundle::graph_sgoc_mm_execute_hints_env();
    G.mem_debug = graph_sgoc_bundle::graph_sgoc_mem_debug_env();
    G.mm_order_trace = graph_sgoc_bundle::graph_sgoc_mm_order_trace_env();
    if (pin_worker >= 0)
        G.gpu_mem_node = starpu_worker_get_memory_node(static_cast<unsigned>(pin_worker));
    G.mem_budget_bytes = mem_budget;
    graph_sgoc_victim_rebuild_belady(data, topo_order);
    if (G.mm_execute && pin_worker >= 0) {
        const starpu_ssize_t used = starpu_memory_get_used(G.gpu_mem_node);
        if (used >= 0)
            G.tracked_gpu_bytes = static_cast<std::int64_t>(used);
        else
            G.tracked_gpu_bytes = mem_initial_bytes;
    }
    flush_timer.lap("runtime_mm_init_victim_tracked_bytes");

    const graph_sgoc_gpu_memory_manager &mm = data->graph_gpu_mm;

    G.replay_task_topo_slot.clear();
    for (size_t ti = 0; ti < topo_order.size(); ++ti) {
        const size_t opi = topo_order[ti];
        if (opi >= data->graph_ops.size())
            continue;
        const GraphOp &op = data->graph_ops[opi];
        if (op.kind == GraphOp::TASK && op.task)
            G.replay_task_topo_slot[op.task] = static_cast<unsigned>(ti);
    }

    const graph_sgoc_replay_accounting_scope replay_scope{data};
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
                graph_sgoc_bundle::graph_sgoc_apply_replay_worker_pin(op.task, pin_worker, vb, pin_cl_cache);
            _starpu_task_insert_submit_built_task(op.task);
            n_task_submitted++;
            if (G.mm_execute && pin_worker >= 0 && ti < mm.topo_post_exec_offload_order.size()
                && !mm.topo_post_exec_offload_order[ti].empty() && op.task) {
                struct starpu_task *const followup = sgoc_next_task_after_topo_slot(data, topo_order, ti);
                if (followup)
                    graph_sgoc_register_ram_offload_for_followup_pop(data, followup, op.task,
                                                                     mm.topo_post_exec_offload_order[ti]);
                else
                    graph_sgoc_register_ram_offload_flush_tail(data, op.task, mm.topo_post_exec_offload_order[ti]);
            }
            if (G.mm_execute && pin_worker >= 0 && ti < mm.topo_post_exec_evict_gpu_only_order.size()
                && !mm.topo_post_exec_evict_gpu_only_order[ti].empty() && op.task) {
                struct starpu_task *const followup_ev = sgoc_next_task_after_topo_slot(data, topo_order, ti);
                if (followup_ev)
                    graph_sgoc_register_evict_gpu_only_for_followup_pop(data, followup_ev,
                                                                        mm.topo_post_exec_evict_gpu_only_order[ti]);
                else
                    graph_sgoc_register_evict_gpu_flush_tail(data, mm.topo_post_exec_evict_gpu_only_order[ti]);
            }
            break;
        case GraphOp::INVALIDATE:
            _starpu_data_invalidate_submit_impl(op.handle);
            break;
        }
    }
    flush_timer.lap(("replay_submit n_task=" + std::to_string(n_task_submitted) + " n_topo_slots=" + std::to_string(topo_order.size())).c_str());

    if (pin_worker >= 0)
        graph_sgoc_run_flush_tail_offloads(data, G.gpu_mem_node);

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

namespace graph_sgoc_bundle {

void graph_sgoc_finalize_outermost_capture(graph_sgoc_data *data, std::vector<GraphOp> &&replay,
                                           std::vector<GraphHandleAccess> &&replay_ha,
                                           unsigned added_invalidate_submit, unsigned sched_ctx_id)
{
    SgocCapturePhaseTimer timer("finalize");
    graph_sgoc_captured_handle_groups parsed{};
    const int v = graph_sgoc_verbose_env();
    graph_sgoc_parse_captured_data_handles(replay, parsed, v);
    timer.lap("parse_captured_data_handles");
    bool has_batch = false;
    std::uint32_t batch_val = 0;
    if (!graph_sgoc_infer_batch_capture_context(replay, &has_batch, &batch_val))
        has_batch = false;
    timer.lap("infer_batch_capture_context");

    ::graph_sgoc_release_outermost_capture(data, std::move(replay), std::move(replay_ha), parsed, has_batch,
                                                 batch_val, v, sched_ctx_id);
    timer.lap("release_outermost_capture");

    std::lock_guard<std::mutex> lock(data->policy_mutex);
    data->graph_captured_handle_groups = std::move(parsed);
    data->graph_total_synthetic_invalidate_inserts += added_invalidate_submit;
    timer.lap("store_captured_handle_groups");
}

} /* namespace graph_sgoc_bundle */
