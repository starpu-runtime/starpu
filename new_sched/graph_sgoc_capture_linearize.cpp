/* Capture list: append TASK/INVALIDATE; linearize to dense graph_ops. */
#include "graph_sgoc_common.hpp"
#include "graph_sgoc_bundle_dag.hpp"
#include "graph_sgoc_bundle_flush.hpp"

namespace graph_sgoc_bundle {

void graph_sched_append_captured_task(graph_sched_data *data, struct starpu_task *task)
{
    graph_sched_insert_missing_pre_write_invalidates(data, task);

    GraphOp op{};
    op.kind = GraphOp::TASK;
    op.task = task;
    op.handle = nullptr;
    graph_sched_graph_op_set_stage_from_sched_ctx(op, task->sched_ctx, task);
    op.predicted_exec_time =
        graph_sched_predicted_exec_time_us_for_pinned_worker(task, data->graph_pinned_worker_id, task->sched_ctx);

    if (data->graph_sgoc) {
        graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
        const size_t sid = G.capture_next_stable_id++;
        op.capture_stable_id = sid;
        G.capture_ops.push_back(std::move(op));
        G.capture_id_to_iter[sid] = std::prev(G.capture_ops.end());
        GraphOp &op_ref = G.capture_ops.back();
        graph_sched_register_task_accesses_op(data, sid, task, op_ref);
        graph_sgoc_capture_add_edges_for_op(data, op_ref);
        return;
    }

    data->graph_ops.push_back(std::move(op));
    graph_sched_register_task_accesses(data, data->graph_ops.size() - 1, task);
    graph_sched_refresh_op_dependencies(data);
}
void graph_sgoc_linearize_capture_to_ops(graph_sched_data *data)
{
    if (!data || !data->graph_sgoc)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    const bool cap_tim = graph_sched_capture_phase_report_enabled();
    const auto t_lin0 = std::chrono::steady_clock::now();
    const size_t n_listed = G.capture_ops.size();
    if (G.capture_ops.empty()) {
        if (cap_tim) {
            const auto t_lin1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t_lin1 - t_lin0).count();
            std::cerr << "sgoc_capture_timing: linearize +" << std::fixed << std::setprecision(3) << ms
                      << " ms list_ops=0 (no-op)" << std::endl;
        }
        return;
    }

    /* One O(total accesses) pass; incremental capture no longer calls this per append. */
    graph_sched_validate_invalidate_then_pure_write_windows(data);

    std::unordered_map<size_t, size_t> sid_to_idx;
    sid_to_idx.reserve(G.capture_ops.size() * 2u);
    std::vector<GraphOp> out;
    out.reserve(G.capture_ops.size());

    for (auto it = G.capture_ops.begin(); it != G.capture_ops.end(); ++it) {
        const size_t sid = it->capture_stable_id;
        sid_to_idx[sid] = out.size();
        GraphOp op = std::move(*it);
        op.capture_stable_id = 0;
        out.push_back(std::move(op));
    }
    G.capture_ops.clear();
    G.capture_id_to_iter.clear();
    G.capture_next_stable_id = 1;

    const auto remap_sid = [&](size_t x) -> size_t {
        if (x == GRAPH_ACCESS_NONE)
            return GRAPH_ACCESS_NONE;
        const auto j = sid_to_idx.find(x);
        return j == sid_to_idx.end() ? GRAPH_ACCESS_NONE : j->second;
    };

    for (GraphOp &op : out) {
        std::vector<size_t> pnew;
        pnew.reserve(op.predecessors.size());
        for (size_t p : op.predecessors) {
            const size_t r = remap_sid(p);
            if (r != GRAPH_ACCESS_NONE)
                pnew.push_back(r);
        }
        op.predecessors.swap(pnew);
        std::vector<size_t> snew;
        snew.reserve(op.successors.size());
        for (size_t s : op.successors) {
            const size_t r = remap_sid(s);
            if (r != GRAPH_ACCESS_NONE)
                snew.push_back(r);
        }
        op.successors.swap(snew);
    }

    for (GraphHandleAccess &a : data->graph_handle_accesses) {
        if (a.op_idx != GRAPH_ACCESS_NONE)
            a.op_idx = remap_sid(a.op_idx);
    }

    data->graph_ops = std::move(out);
    /* Dense op_idx remap is done; rebuild handle list heads/tails and pred/succ from chains so the graph matches
     * what checkpoint insertion and Belady/MM will see (capture-time synthetic pre-W invalidates are already ops). */
    graph_sgoc_rebuild_lists_and_refresh_deps(data);
    if (cap_tim) {
        const auto t_lin1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t_lin1 - t_lin0).count();
        std::cerr << "sgoc_capture_timing: linearize +" << std::fixed << std::setprecision(3) << ms << " ms list_ops="
                  << n_listed << " graph_ops_out=" << data->graph_ops.size() << std::endl;
    }
}

void graph_sched_append_capture_explicit_invalidate(graph_sched_data *data, starpu_data_handle_t handle)
{
    if (!data->graph_sgoc)
        return;
    graph_sched_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    GraphOp op{};
    op.kind = GraphOp::INVALIDATE;
    op.task = nullptr;
    op.handle = handle;
    graph_sched_graph_op_set_stage_from_sched_ctx(op, starpu_sched_ctx_get_context(), nullptr);
    const size_t sid = G.capture_next_stable_id++;
    op.capture_stable_id = sid;
    G.capture_ops.push_back(std::move(op));
    G.capture_id_to_iter[sid] = std::prev(G.capture_ops.end());
    graph_sched_register_invalidate_access_op(data, G.capture_ops.back(), sid, handle);
    graph_sgoc_capture_add_edges_for_op(data, G.capture_ops.back());
}

} /* namespace graph_sgoc_bundle */
