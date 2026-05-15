/* WRR checkpoint orchestration: feasible producers, insert loop, remat-speed ranking, apply before topo. */
#include "graph_sgoc_common.hpp"
#include "graph_sgoc_bundle_checkpoint.hpp"
#include "graph_sgoc_bundle_dag.hpp"
#include "graph_sgoc_bundle_flush.hpp"

namespace graph_sgoc_bundle {

void graph_sgoc_append_unique_op_idx(std::vector<size_t> &op_indices, size_t op_idx)
{
    if (op_idx == GRAPH_ACCESS_NONE)
        return;
    for (size_t existing : op_indices) {
        if (existing == op_idx)
            return;
    }
    op_indices.push_back(op_idx);
}

void graph_sgoc_collect_task_ops_for_handle(const std::vector<GraphHandleAccess> &handle_accesses,
                                                    size_t access_idx, std::vector<size_t> &op_indices_out)
{
    if (access_idx >= handle_accesses.size())
        return;

    size_t idx = access_idx;
    while (idx != GRAPH_ACCESS_NONE) {
        if (idx >= handle_accesses.size())
            return;
        idx = handle_accesses[idx].prev_for_handle;
    }

    idx = access_idx;
    while (handle_accesses[idx].prev_for_handle != GRAPH_ACCESS_NONE)
        idx = handle_accesses[idx].prev_for_handle;

    while (idx != GRAPH_ACCESS_NONE) {
        if (idx >= handle_accesses.size())
            return;
        const GraphHandleAccess &access = handle_accesses[idx];
        if (access.task != nullptr)
            graph_sgoc_append_unique_op_idx(op_indices_out, access.op_idx);
        idx = access.next_for_handle;
    }
}

void graph_sgoc_collect_checkpoint_affected_ops(const std::vector<GraphOp> &ops,
                                                        const std::vector<GraphHandleAccess> &handle_accesses,
                                                        size_t checkpoint_op_idx,
                                                        std::vector<size_t> &affected_op_indices_out)
{
    affected_op_indices_out.clear();
    if (checkpoint_op_idx >= ops.size())
        return;

    graph_sgoc_append_unique_op_idx(affected_op_indices_out, checkpoint_op_idx);
    const GraphOp &checkpoint_op = ops[checkpoint_op_idx];
    for (const GraphOpHandleAccessRef &ref : checkpoint_op.handle_accesses) {
        if (ref.access_idx >= handle_accesses.size())
            continue;
        graph_sgoc_collect_task_ops_for_handle(handle_accesses, ref.access_idx, affected_op_indices_out);
    }
}

void graph_sgoc_update_checkpoint_state_for_op(
    std::vector<GraphOp> &ops, const std::vector<GraphHandleAccess> &handle_accesses, size_t op_idx,
    const std::vector<SgocWrrCheckpointTemplate> *checkpoint_templates)
{
    (void)handle_accesses;
    if (op_idx >= ops.size())
        return;

    GraphOp &op = ops[op_idx];
    if (!checkpoint_templates || checkpoint_templates->empty()) {
        op.checkpoint_idempotent = false;
        op.checkpoint_wrr = false;
        return;
    }

    const bool m = graph_sgoc_op_matches_wrr_checkpoint_templates(op, *checkpoint_templates);
    op.checkpoint_idempotent = m;
    op.checkpoint_wrr = m;
}

void graph_sgoc_collect_checkpoint_eligible_count(const std::vector<GraphOp> &ops, size_t &eligible_tasks_out)
{
    eligible_tasks_out = 0;
    for (const GraphOp &op : ops) {
        if (op.checkpoint_wrr)
            eligible_tasks_out++;
    }
}

void graph_sgoc_log_checkpoint_candidate_skip(const std::vector<GraphOp> &ops, size_t op_idx,
                                                      const graph_sgoc_data *policy_data, const char *reason,
                                                      unsigned consecutive_reads,
                                                      const GraphOpHandleAccessRef *write_ref)
{
    if (op_idx >= ops.size())
        return;
    const GraphOp &op = ops[op_idx];
    struct starpu_task *t = op.task;
    const char *cl_name = (t && t->cl && t->cl->name) ? t->cl->name : "?";
    const unsigned long task_id = t ? starpu_task_get_job_id(t) : 0;
    const void *wh = (write_ref && write_ref->handle) ? static_cast<const void *>(write_ref->handle) : nullptr;
    const char *ordering = "capture_order (first op with checkpoint_wrr)";
    if (policy_data && !policy_data->graph_idempotent_tasks_sorted.empty())
        ordering = "rematerialization_speed_desc (graph_idempotent_tasks_sorted; one row per producer GraphOp index — "
                   "task structs may be pooled/reused across minibatches)";

    std::cerr << "sgoc: checkpoint insert skip — candidates are TASK ops that (1) appear as checkpointable "
                 "activation producers in graph_sgoc_parse_captured_data_handles and (2) have idempotent shape "
                 "(one pure-W, other buffers R/scratch). Ordering: " << ordering << std::endl;
    std::cerr << "sgoc:   op_idx=" << op_idx << " task_id=" << task_id << " cl=" << cl_name;
    if (op.graph_stage_subiteration_valid)
        std::cerr << " graph_subiter=" << op.graph_stage_subiteration;
    std::cerr << " written_handle=" << wh
              << " consecutive_pure_read_TASKs_on_global_chain_after_W=" << consecutive_reads << std::endl;
    std::cerr << "sgoc:   chain rule: after W, need a forward read (odd training graph_subiter) then a backward read "
                 "(even); invalidate immediately before the backward read (see "
                 "graph_sgoc_checkpoint_wrr_chain_resolve). Reason: " << (reason ? reason : "?") << std::endl;
}

void graph_sgoc_build_feasible_checkpoint_task_order(const std::vector<GraphOp> &ops,
                                                             const std::vector<GraphHandleAccess> &handle_accesses,
                                                             const graph_sgoc_data *policy_data,
                                                             const std::vector<SgocWrrCheckpointTemplate> *checkpoint_templates,
                                                             const graph_sgoc_wrr_activation_sub_policy *activation_sub_policy,
                                                             std::vector<size_t> &out_producer_op_indices,
                                                             std::vector<size_t> &chain_scratch)
{
    out_producer_op_indices.clear();
    if (!checkpoint_templates || checkpoint_templates->empty())
        return;

    auto chain_ok = [&](const GraphOp &op) -> bool {
        const GraphOpHandleAccessRef *wr = nullptr;
        const char *fr = nullptr;
        unsigned n = 0;
        return graph_sgoc_checkpoint_wrr_chain_resolve(op, ops, handle_accesses, &wr, chain_scratch, &fr, &n,
                                                        activation_sub_policy);
    };

    auto try_add_op_index = [&](size_t oi) {
        if (oi >= ops.size())
            return;
        const GraphOp &op = ops[oi];
        if (op.kind != GraphOp::TASK || !op.checkpoint_wrr)
            return;
        const GraphOpHandleAccessRef *wr = graph_op_find_single_pure_write_access(op);
        if (!wr || !wr->handle)
            return;
        if (chain_ok(op))
            out_producer_op_indices.push_back(oi);
    };

    if (policy_data && !policy_data->graph_idempotent_tasks_sorted.empty()) {
        for (const GraphIdempotentTaskPredicted &row : policy_data->graph_idempotent_tasks_sorted) {
            if (row.producer_op_idx != GRAPH_ACCESS_NONE)
                try_add_op_index(row.producer_op_idx);
        }
    } else {
        for (size_t i = 0; i < ops.size(); ++i)
            try_add_op_index(i);
    }
}

unsigned graph_sgoc_insert_checkpoints(std::vector<GraphOp> &ops, std::vector<GraphHandleAccess> &handle_accesses,
                                               int pin_worker, graph_sgoc_data *policy_data,
                                               const std::vector<SgocWrrCheckpointTemplate> *checkpoint_templates,
                                               bool repeat_previous_batch_flush,
                                               const graph_sgoc_wrr_activation_sub_policy *activation_sub_policy)
{
    const unsigned checkpoint_max = graph_sgoc_checkpoint_max_env();
    const int sched_verbose = graph_sgoc_verbose_env();
    unsigned inserted = 0;
    if (!checkpoint_templates || checkpoint_templates->empty())
        return 0;

    std::vector<size_t> chain_scratch;
    std::vector<size_t> feasible_producer_op_indices;
    graph_sgoc_build_feasible_checkpoint_task_order(ops, handle_accesses, policy_data, checkpoint_templates,
                                                     activation_sub_policy, feasible_producer_op_indices, chain_scratch);

    if (sched_verbose >= 3 && !repeat_previous_batch_flush) {
        std::cerr << "sgoc: checkpoint insert plan: chain_feasible_producers=" << feasible_producer_op_indices.size()
                  << " (only these are attempted; activation producers that fail the chain rule are not scanned)"
                  << std::endl;
    }

    for (size_t fi = 0; fi < feasible_producer_op_indices.size(); ++fi) {
        if (inserted >= checkpoint_max)
            break;

        size_t op_idx = feasible_producer_op_indices[fi];
        if (op_idx >= ops.size() || ops[op_idx].kind != GraphOp::TASK || !ops[op_idx].checkpoint_wrr)
            continue;

        struct starpu_task *checkpointed_task = ops[op_idx].task;

        const GraphOpHandleAccessRef *wr_precheck = nullptr;
        const char *chain_reason = nullptr;
        unsigned n_chain_reads = 0;
        if (!graph_sgoc_checkpoint_wrr_chain_resolve(ops[op_idx], ops, handle_accesses, &wr_precheck, chain_scratch,
                                                       &chain_reason, &n_chain_reads, activation_sub_policy)) {
            if (sched_verbose >= 2 && !repeat_previous_batch_flush)
                graph_sgoc_log_checkpoint_candidate_skip(ops, op_idx, policy_data, chain_reason, n_chain_reads,
                                                          wr_precheck);
            if (op_idx < ops.size()) {
                ops[op_idx].checkpoint_idempotent = false;
                ops[op_idx].checkpoint_wrr = false;
            }
            continue;
        }

        struct starpu_task *checkpoint_task = graph_sgoc_clone_task_for_checkpoint(ops[op_idx].task);
        if (!checkpoint_task) {
            if (graph_sgoc_verbose_env() >= 2)
                std::cerr << "sgoc: failed to allocate checkpoint task clone" << std::endl;
            break;
        }
        size_t inv_insert_pos = 0;
        if (!graph_sgoc_insert_checkpoint_for_wrr_task(ops, handle_accesses, op_idx, checkpoint_task, pin_worker,
                                                      activation_sub_policy, &inv_insert_pos)) {
            graph_sgoc_destroy_checkpoint_task(checkpoint_task);
            if (sched_verbose >= 2)
                std::cerr << "sgoc: checkpoint insert failed after chain precheck (bug?) op_idx=" << op_idx << std::endl;
            if (op_idx < ops.size()) {
                ops[op_idx].checkpoint_idempotent = false;
                ops[op_idx].checkpoint_wrr = false;
            }
            continue;
        }
        for (size_t j = fi + 1; j < feasible_producer_op_indices.size(); ++j) {
            if (feasible_producer_op_indices[j] >= inv_insert_pos)
                feasible_producer_op_indices[j] += 2;
        }
        std::vector<size_t> affected_op_indices;
        graph_sgoc_collect_checkpoint_affected_ops(ops, handle_accesses, op_idx, affected_op_indices);
        for (size_t affected_op_idx : affected_op_indices)
            graph_sgoc_update_checkpoint_state_for_op(ops, handle_accesses, affected_op_idx, checkpoint_templates);

        if (op_idx < ops.size()) {
            ops[op_idx].checkpoint_idempotent = false;
            ops[op_idx].checkpoint_wrr = false;
        }

        if (sched_verbose >= 3 && !repeat_previous_batch_flush) {
            const char *cl_name = (checkpointed_task && checkpointed_task->cl && checkpointed_task->cl->name)
                                      ? checkpointed_task->cl->name
                                      : "unknown";
            const unsigned long task_id = checkpointed_task ? starpu_task_get_job_id(checkpointed_task) : 0;
            const void *wh =
                (wr_precheck && wr_precheck->handle) ? static_cast<const void *>(wr_precheck->handle) : nullptr;
            const std::int64_t rbytes = graph_sgoc_op_intrinsic_memory_delta(ops[op_idx]);
            std::cerr << "sgoc: checkpoint inserted [" << (inserted + 1) << "/" << checkpoint_max
                      << "] producer_op_idx=" << op_idx << " cl=" << cl_name << " producer_job_id=" << task_id
                      << " written_handle=" << wh << " clone_task=" << static_cast<const void *>(checkpoint_task)
                      << " remat_bytes=" << rbytes << std::endl;
        }
        inserted++;
    }

    return inserted;
}

void graph_sgoc_checkpoint_pool_stats(const std::vector<GraphOp> &ops,
                                              const std::vector<GraphHandleAccess> &handle_accesses,
                                              const graph_sgoc_wrr_activation_sub_policy *activation_sub_policy,
                                              size_t *checkpoint_activation_producers_out,
                                              size_t *checkpoint_chain_insertable_out)
{
    graph_sgoc_collect_checkpoint_eligible_count(ops, *checkpoint_activation_producers_out);
    *checkpoint_chain_insertable_out = 0;
    std::vector<size_t> chain_feasible_scratch;
    for (const GraphOp &op : ops) {
        if (!op.checkpoint_wrr)
            continue;
        const GraphOpHandleAccessRef *wact = graph_op_find_single_pure_write_access(op);
        if (!wact || !wact->handle)
            continue;
        const GraphOpHandleAccessRef *wr = nullptr;
        const char *fr = nullptr;
        unsigned n = 0;
        if (graph_sgoc_checkpoint_wrr_chain_resolve(op, ops, handle_accesses, &wr, chain_feasible_scratch, &fr, &n,
                                                     activation_sub_policy))
            (*checkpoint_chain_insertable_out)++;
    }
}

/**
 * Requires template marks (\c checkpoint_wrr) on \p ops.
 * Fills policy_data->graph_idempotent_tasks_sorted with activation producers, descending by rematerialization_speed_bps.
 */
void graph_sgoc_fill_wrr_checkpoint_order_by_remat_speed(graph_sgoc_data *policy_data,
                                                                 const std::vector<GraphOp> &ops, int pin_worker,
                                                                 bool repeat_previous_batch_flush)
{
    if (!policy_data)
        return;
    policy_data->graph_idempotent_tasks_sorted.clear();

    std::vector<GraphIdempotentTaskPredicted> rows;
    for (size_t oi = 0; oi < ops.size(); ++oi) {
        const GraphOp &op = ops[oi];
        if (op.kind != GraphOp::TASK || !op.checkpoint_wrr || !op.task)
            continue;
        const GraphOpHandleAccessRef *wr = graph_op_find_single_pure_write_access(op);
        if (!wr || !wr->handle)
            continue;
        const double raw =
            graph_sgoc_predicted_exec_time_us_for_pinned_worker(op.task, pin_worker, op.task->sched_ctx);
        const double time_us = graph_sgoc_effective_predicted_us(raw);
        const std::int64_t bytes = graph_sgoc_op_intrinsic_memory_delta(op);
        double bps = 0;
        if (bytes > 0 && std::isfinite(time_us) && time_us > 0.0)
            bps = static_cast<double>(bytes) * 1e6 / time_us;
        GraphIdempotentTaskPredicted row{};
        row.task = op.task;
        row.producer_op_idx = oi;
        row.predicted_exec_time_us = time_us;
        row.rematerialization_bytes = bytes;
        row.rematerialization_speed_bps = bps;
        rows.push_back(row);
    }
    std::sort(rows.begin(), rows.end(), [](const GraphIdempotentTaskPredicted &a, const GraphIdempotentTaskPredicted &b) {
        if (a.rematerialization_speed_bps != b.rematerialization_speed_bps)
            return a.rematerialization_speed_bps > b.rematerialization_speed_bps;
        if (a.predicted_exec_time_us != b.predicted_exec_time_us)
            return a.predicted_exec_time_us < b.predicted_exec_time_us;
        if (a.task != b.task)
            return a.task < b.task;
        return a.producer_op_idx < b.producer_op_idx;
    });
    policy_data->graph_idempotent_tasks_sorted = std::move(rows);

    if (graph_sgoc_verbose_env() >= 3 && !repeat_previous_batch_flush) {
        std::cerr << "sgoc: checkpoint-eligible activation producers by rematerialization speed (descending B/s, "
                     "fastest first; worker_id="
                  << pin_worker << " count=" << policy_data->graph_idempotent_tasks_sorted.size() << "):" << std::endl;
        for (const GraphIdempotentTaskPredicted &row : policy_data->graph_idempotent_tasks_sorted) {
            const char *cln =
                (row.task && row.task->cl && row.task->cl->name) ? row.task->cl->name : "?";
            const unsigned long jid = row.task ? starpu_task_get_job_id(row.task) : 0;
            std::cerr << "  task_id=" << jid << " cl=" << cln << " remat_bytes=" << row.rematerialization_bytes
                      << " predicted_us=";
            if (std::isfinite(row.predicted_exec_time_us))
                std::cerr << row.predicted_exec_time_us;
            else
                std::cerr << "inf";
            std::cerr << " remat_B_per_s=" << row.rematerialization_speed_bps << std::endl;
        }
    }
}
void graph_sgoc_apply_wrr_checkpoints_before_topo(graph_sgoc_data *data, const graph_sgoc_captured_handle_groups &parsed,
                                                  int vb)
{
    if (!data)
        return;
    const unsigned ck_max = graph_sgoc_checkpoint_max_env();
    if (ck_max == 0 || parsed.activations.empty())
        return;

    std::unordered_set<void *> keys;
    keys.reserve(parsed.activations.size());
    for (starpu_data_handle_t h : parsed.activations) {
        if (h)
            keys.insert(static_cast<void *>(h));
    }
    if (keys.empty())
        return;

    std::vector<SgocWrrCheckpointTemplate> ckpt_templates;
    graph_sgoc_collect_wrr_checkpoint_templates(data->graph_ops, parsed, keys, ckpt_templates);
    graph_sgoc_apply_wrr_checkpoint_templates(data->graph_ops, ckpt_templates);

    graph_sgoc_wrr_activation_sub_policy sub_policy{};
    if (parsed.activation_checkpoint_min_pair_valid && parsed.activation_forward_backward_delta_sub > 0u) {
        sub_policy.use_producer_relative_pair = true;
        sub_policy.forward_backward_delta_sub = parsed.activation_forward_backward_delta_sub;
    }

    graph_sgoc_fill_wrr_checkpoint_order_by_remat_speed(data, data->graph_ops, data->graph_pinned_worker_id, false);

    if (vb >= 3) {
        size_t activation_producers = 0;
        size_t chain_insertable = 0;
        graph_sgoc_checkpoint_pool_stats(data->graph_ops, data->graph_handle_accesses, &sub_policy,
                                          &activation_producers, &chain_insertable);
        std::cerr << "sgoc: checkpoint pass: activation_producers=" << activation_producers
                  << " chain_insertable=" << chain_insertable << " checkpoint_max=" << ck_max << std::endl;
    }

    const unsigned inserted = graph_sgoc_insert_checkpoints(
        data->graph_ops, data->graph_handle_accesses, data->graph_pinned_worker_id, data, &ckpt_templates, false,
        &sub_policy);
    data->graph_total_checkpoint_inserts += inserted;

    if (vb >= 3)
        std::cerr << "sgoc: checkpoint pass: inserted_checkpoints=" << inserted << std::endl;

    graph_sgoc_rebuild_lists_and_refresh_deps(data);
}

} /* namespace graph_sgoc_bundle */
