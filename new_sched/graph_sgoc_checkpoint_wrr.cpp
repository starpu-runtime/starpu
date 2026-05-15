/* WRR checkpoint graph surgery: templates, chain resolution, clone, single-task insert. */
#include "graph_sgoc_common.hpp"
#include "graph_sgoc_bundle_dag.hpp"
#include "graph_sgoc_bundle_flush.hpp"
#include "graph_sgoc_bundle_parse.hpp"

namespace graph_sgoc_bundle {

const GraphOpHandleAccessRef *graph_op_find_single_pure_write_access(const GraphOp &op);
bool graph_op_is_checkpoint_idempotent(const GraphOp &op);
bool graph_sgoc_task_codelet_excluded_from_wrr_checkpoint(const struct starpu_task *task);

static void graph_sgoc_append_task_structure_sig_from_op(const GraphOp &op,
                                                          std::vector<GraphBatchTaskStructureSig> &out)
{
    if (op.kind != GraphOp::TASK || !op.task)
        return;
    struct starpu_task *t = op.task;
    GraphBatchTaskStructureSig s;
    if (t->cl && t->cl->name)
        s.codelet_name = t->cl->name;
    if (t->cl) {
        const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(t);
        s.buffer_sizes.reserve(nbuf);
        for (unsigned i = 0; i < nbuf; ++i) {
            starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(t, i);
            s.buffer_sizes.push_back(h ? starpu_data_get_size(h) : 0u);
        }
    }
    out.push_back(std::move(s));
}

/** Same as graph_sgoc_append_task_structure_sig_from_op but records per-buffer modes (batch-0 consistency vs STARPU_RW). */
static void graph_sgoc_append_task_structure_sig_from_op_with_modes(const GraphOp &op,
                                                                     std::vector<GraphBatchTaskStructureSig> &out)
{
    if (op.kind != GraphOp::TASK || !op.task)
        return;
    struct starpu_task *t = op.task;
    GraphBatchTaskStructureSig s;
    if (t->cl && t->cl->name)
        s.codelet_name = t->cl->name;
    if (t->cl) {
        const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(t);
        s.buffer_sizes.reserve(nbuf);
        s.buffer_modes.reserve(nbuf);
        for (unsigned i = 0; i < nbuf; ++i) {
            starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(t, i);
            s.buffer_sizes.push_back(h ? starpu_data_get_size(h) : 0u);
            s.buffer_modes.push_back(static_cast<unsigned>(STARPU_TASK_GET_MODE(t, i)));
        }
    }
    out.push_back(std::move(s));
}

static bool graph_sgoc_task_structure_sig_equal(const GraphBatchTaskStructureSig &a,
                                                 const GraphBatchTaskStructureSig &b)
{
    return a.codelet_name == b.codelet_name && a.buffer_sizes == b.buffer_sizes && a.buffer_modes == b.buffer_modes;
}

/** Fill \p out with codelet name, per-buffer sizes and StarPU access modes (same layout as captured buffers). */
static void graph_sgoc_build_task_structure_sig_with_modes_from_op(const GraphOp &op, GraphBatchTaskStructureSig &out)
{
    out = GraphBatchTaskStructureSig{};
    if (op.kind != GraphOp::TASK || !op.task)
        return;
    struct starpu_task *t = op.task;
    if (t->cl && t->cl->name)
        out.codelet_name = t->cl->name;
    if (!t->cl)
        return;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(t);
    out.buffer_sizes.reserve(nbuf);
    out.buffer_modes.reserve(nbuf);
    for (unsigned i = 0; i < nbuf; ++i) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(t, i);
        out.buffer_sizes.push_back(h ? starpu_data_get_size(h) : 0u);
        out.buffer_modes.push_back(static_cast<unsigned>(STARPU_TASK_GET_MODE(t, i)));
    }
}

unsigned graph_sgoc_pure_write_buffer_index(const GraphOp &op)
{
    const GraphOpHandleAccessRef *w = graph_op_find_single_pure_write_access(op);
    if (!w)
        return std::numeric_limits<unsigned>::max();
    for (unsigned i = 0; i < op.handle_accesses.size(); ++i) {
        if (&op.handle_accesses[i] == w)
            return i;
    }
    return std::numeric_limits<unsigned>::max();
}

void graph_sgoc_collect_wrr_checkpoint_templates(const std::vector<GraphOp> &ops,
                                                         const graph_sgoc_captured_handle_groups &parsed,
                                                         const std::unordered_set<void *> &checkpointable_activation_keys,
                                                         std::vector<SgocWrrCheckpointTemplate> &out)
{
    out.clear();
    if (!parsed.activation_checkpoint_min_pair_valid || checkpointable_activation_keys.empty())
        return;
    const std::uint32_t min_f = parsed.activation_checkpoint_min_forward_sub;
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        if (!op.graph_stage_subiteration_valid || op.graph_stage_subiteration != min_f)
            continue;
        if (graph_sgoc_task_codelet_excluded_from_wrr_checkpoint(op.task))
            continue;
        if (!graph_op_is_checkpoint_idempotent(op))
            continue;
        const GraphOpHandleAccessRef *wr = graph_op_find_single_pure_write_access(op);
        if (!wr || !wr->handle)
            continue;
        if (!checkpointable_activation_keys.count(static_cast<void *>(wr->handle)))
            continue;
        const unsigned wbi = graph_sgoc_pure_write_buffer_index(op);
        if (wbi == std::numeric_limits<unsigned>::max())
            continue;
        SgocWrrCheckpointTemplate row;
        graph_sgoc_build_task_structure_sig_with_modes_from_op(op, row.sig);
        row.pure_write_buffer_index = wbi;
        bool dup = false;
        for (const SgocWrrCheckpointTemplate &ex : out) {
            if (ex.pure_write_buffer_index != row.pure_write_buffer_index)
                continue;
            if (graph_sgoc_task_structure_sig_equal(ex.sig, row.sig)) {
                dup = true;
                break;
            }
        }
        if (!dup)
            out.push_back(std::move(row));
    }
}

bool graph_sgoc_op_matches_wrr_checkpoint_templates(const GraphOp &op,
                                                            const std::vector<SgocWrrCheckpointTemplate> &templates)
{
    if (op.kind != GraphOp::TASK || !op.task)
        return false;
    if (graph_sgoc_task_codelet_excluded_from_wrr_checkpoint(op.task))
        return false;
    if (!graph_op_is_checkpoint_idempotent(op))
        return false;
    if (!op.graph_stage_subiteration_valid || !graph_sgoc_graph_subiter_is_forward_stage(op.graph_stage_subiteration))
        return false;
    const unsigned wbi = graph_sgoc_pure_write_buffer_index(op);
    if (wbi == std::numeric_limits<unsigned>::max())
        return false;
    GraphBatchTaskStructureSig sig;
    graph_sgoc_build_task_structure_sig_with_modes_from_op(op, sig);
    for (const SgocWrrCheckpointTemplate &tpl : templates) {
        if (tpl.pure_write_buffer_index != wbi)
            continue;
        if (graph_sgoc_task_structure_sig_equal(sig, tpl.sig))
            return true;
    }
    return false;
}

void graph_sgoc_apply_wrr_checkpoint_templates(std::vector<GraphOp> &ops,
                                                       const std::vector<SgocWrrCheckpointTemplate> &templates)
{
    for (GraphOp &op : ops) {
        op.checkpoint_wrr = false;
        op.checkpoint_idempotent = false;
        if (templates.empty())
            continue;
        if (!graph_sgoc_op_matches_wrr_checkpoint_templates(op, templates)) {
            continue;
        }
        op.checkpoint_wrr = true;
        op.checkpoint_idempotent = true;
    }
}

const GraphOpHandleAccessRef *graph_op_find_single_pure_write_access(const GraphOp &op)
{
    const GraphOpHandleAccessRef *write_ref = nullptr;

    for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
        if (!graph_access_mode_is_pure_write(ref.mode))
            continue;
        if (write_ref != nullptr)
            return nullptr;
        write_ref = &ref;
    }

    return write_ref;
}

bool graph_op_is_checkpoint_idempotent(const GraphOp &op)
{
    if (op.kind != GraphOp::TASK)
        return false;

    unsigned pure_write_count = 0;
    for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
        if (graph_access_mode_is_pure_write(ref.mode)) {
            pure_write_count++;
            continue;
        }
        if (graph_access_mode_is_pure_read(ref.mode) || graph_access_mode_is_pure_scratch(ref.mode))
            continue;
        return false;
    }

    return pure_write_count == 1;
}

/** NNTile wrappers: GEMM / explicit copies are omitted from activation WRR checkpoint candidate sets. */
bool graph_sgoc_task_codelet_excluded_from_wrr_checkpoint(const struct starpu_task *task)
{
    if (!task || !task->cl || !task->cl->name)
        return false;
    const char *const n = task->cl->name;
    return !std::strcmp(n, "nntile_copy") || !std::strcmp(n, "nntile_gemm");
}

void graph_sgoc_collect_consecutive_pure_read_task_accesses(const std::vector<GraphHandleAccess> &handle_accesses,
                                                                    size_t write_access_idx,
                                                                    std::vector<size_t> &read_accesses_out)
{
    read_accesses_out.clear();
    if (write_access_idx >= handle_accesses.size())
        return;

    size_t next_idx = handle_accesses[write_access_idx].next_for_handle;
    while (next_idx != GRAPH_ACCESS_NONE) {
        if (next_idx >= handle_accesses.size())
            break;

        const GraphHandleAccess &access = handle_accesses[next_idx];
        if (graph_access_mode_is_invalidate(access.mode) || graph_access_mode_is_writer(access.mode))
            break;

        if (access.task != nullptr) {
            if (!graph_access_mode_is_pure_read(access.mode))
                break;
            read_accesses_out.push_back(next_idx);
        }

        next_idx = access.next_for_handle;
    }
}

bool graph_sgoc_access_op_graph_subiter(const std::vector<GraphOp> &ops,
                                                const std::vector<GraphHandleAccess> &handle_accesses, size_t access_idx,
                                                bool *valid_out, std::uint32_t *sub_out)
{
    if (access_idx >= handle_accesses.size())
        return false;
    const size_t oi = handle_accesses[access_idx].op_idx;
    if (oi >= ops.size())
        return false;
    const GraphOp &rop = ops[oi];
    *valid_out = rop.graph_stage_subiteration_valid;
    *sub_out = rop.graph_stage_subiteration;
    return true;
}

/** WRR chain: pair reads at producer forward sub \e f and sub \e f + \c forward_backward_delta_sub (NNTile delta=1). */

/**
 * Checkpoint insertion: on the handle chain after W, require a pure-read TASK in a forward training stage and a
 * later pure-read TASK in a backward training stage. Invalidate + clone are scheduled immediately before the backward
 * read. With multiple inner minibatches, several forward reads may appear before the matching backward; we try each
 * forward anchor and pair it with the first later backward read, then prefer the pair whose forward/backward ops match
 * the producer's \c graph_stage_batch_iteration when that tag is valid on the producer.
 */
bool graph_sgoc_checkpoint_wrr_chain_resolve(const GraphOp &producer_op, const std::vector<GraphOp> &ops,
                                                     const std::vector<GraphHandleAccess> &handle_accesses,
                                                     const GraphOpHandleAccessRef **write_ref_out,
                                                     std::vector<size_t> &read_accesses_out, const char **failure_reason_out,
                                                     unsigned *consecutive_pure_read_tasks_out,
                                                     const graph_sgoc_wrr_activation_sub_policy *activation_sub_policy)
{
    if (failure_reason_out)
        *failure_reason_out = nullptr;
    if (consecutive_pure_read_tasks_out)
        *consecutive_pure_read_tasks_out = 0;
    *write_ref_out = nullptr;
    read_accesses_out.clear();

    if (producer_op.kind != GraphOp::TASK || !producer_op.task) {
        if (failure_reason_out)
            *failure_reason_out = "not a TASK op or null task pointer";
        return false;
    }

    const GraphOpHandleAccessRef *write_ref = graph_op_find_single_pure_write_access(producer_op);
    *write_ref_out = write_ref;
    if (!write_ref) {
        if (failure_reason_out)
            *failure_reason_out =
                "no exactly-one pure-STARPU_W access (insert needs a single producer write ref on the checkpointed handle)";
        return false;
    }
    if (write_ref->access_idx >= handle_accesses.size()) {
        if (failure_reason_out)
            *failure_reason_out = "pure-W access_idx is out of range for graph_handle_accesses (stale graph?)";
        return false;
    }

    std::vector<size_t> all_reads;
    graph_sgoc_collect_consecutive_pure_read_task_accesses(handle_accesses, write_ref->access_idx, all_reads);
    if (consecutive_pure_read_tasks_out)
        *consecutive_pure_read_tasks_out = static_cast<unsigned>(all_reads.size());

    const bool use_delta =
        activation_sub_policy && activation_sub_policy->use_producer_relative_pair;
    std::uint32_t req_f = 0u;
    std::uint32_t req_b = 0u;
    if (use_delta) {
        if (!producer_op.graph_stage_subiteration_valid) {
            if (failure_reason_out)
                *failure_reason_out = "producer TASK has no valid graph_stage_subiteration for chain pairing";
            return false;
        }
        req_f = producer_op.graph_stage_subiteration;
        const unsigned long long sum =
            static_cast<unsigned long long>(req_f)
            + static_cast<unsigned long long>(activation_sub_policy->forward_backward_delta_sub);
        if (sum > static_cast<unsigned long long>(std::numeric_limits<std::uint32_t>::max())) {
            if (failure_reason_out)
                *failure_reason_out = "forward/backward sub pair overflows uint32_t";
            return false;
        }
        req_b = static_cast<std::uint32_t>(sum);
    }

    std::vector<std::pair<size_t, size_t>> fwd_bwd_pairs;
    fwd_bwd_pairs.reserve(8u);
    for (size_t k = 0; k < all_reads.size(); ++k) {
        bool vf = false;
        std::uint32_t sf = 0;
        if (!graph_sgoc_access_op_graph_subiter(ops, handle_accesses, all_reads[k], &vf, &sf))
            continue;
        if (!vf)
            continue;
        if (use_delta) {
            if (sf != req_f)
                continue;
        } else if (!graph_sgoc_graph_subiter_is_forward_stage(sf)) {
            continue;
        }
        const size_t r_fwd_access = all_reads[k];
        for (size_t j = k + 1; j < all_reads.size(); ++j) {
            bool vb = false;
            std::uint32_t sb = 0;
            if (!graph_sgoc_access_op_graph_subiter(ops, handle_accesses, all_reads[j], &vb, &sb))
                continue;
            if (!vb)
                continue;
            if (use_delta) {
                if (sb != req_b)
                    continue;
            } else if (!graph_sgoc_graph_subiter_is_backward_stage(sb)) {
                continue;
            }
            fwd_bwd_pairs.push_back({r_fwd_access, all_reads[j]});
            break;
        }
    }

    if (fwd_bwd_pairs.empty()) {
        if (failure_reason_out)
            *failure_reason_out = use_delta
                                      ? "after W on the handle chain, need a pure-read TASK at the producer's forward "
                                        "graph_subiter and a later pure-read TASK at graph_subiter+fwd_bwd_delta; "
                                        "invalidate is placed before that backward read"
                                      : "after W on the handle chain, need a pure-read TASK with valid forward-stage "
                                        "graph_subiter (odd; not 0/optimizer) and a later pure-read TASK with "
                                        "backward-stage graph_subiter (even); invalidate is placed before that "
                                        "backward read";
        return false;
    }

    size_t chosen = 0;
    if (producer_op.graph_stage_batch_iteration_valid && fwd_bwd_pairs.size() > 1) {
        const std::uint32_t pb = producer_op.graph_stage_batch_iteration;
        int best_score = -1;
        for (size_t pi = 0; pi < fwd_bwd_pairs.size(); ++pi) {
            const size_t fidx = fwd_bwd_pairs[pi].first;
            const size_t bidx = fwd_bwd_pairs[pi].second;
            if (fidx >= handle_accesses.size() || bidx >= handle_accesses.size())
                continue;
            const GraphOp &opf = ops[handle_accesses[fidx].op_idx];
            const GraphOp &opb = ops[handle_accesses[bidx].op_idx];
            int score = 0;
            if (opb.graph_stage_batch_iteration_valid && opb.graph_stage_batch_iteration == pb)
                score += 2;
            if (opf.graph_stage_batch_iteration_valid && opf.graph_stage_batch_iteration == pb)
                score += 1;
            if (score > best_score
                || (score == best_score && fwd_bwd_pairs[pi].first < fwd_bwd_pairs[chosen].first)) {
                best_score = score;
                chosen = pi;
            }
        }
    }

    const size_t r_fwd_access = fwd_bwd_pairs[chosen].first;
    const size_t r_bwd_access = fwd_bwd_pairs[chosen].second;
    read_accesses_out.push_back(r_fwd_access);
    read_accesses_out.push_back(r_bwd_access);
    return true;
}

static size_t graph_sgoc_find_prev_handle_producer_op_idx(const std::vector<GraphHandleAccess> &handle_accesses,
                                                           size_t access_idx)
{
    size_t prev_idx = access_idx;
    while (prev_idx != GRAPH_ACCESS_NONE) {
        if (prev_idx >= handle_accesses.size())
            return GRAPH_ACCESS_NONE;
        const GraphHandleAccess &access = handle_accesses[prev_idx];
        if (graph_access_is_handle_producer_for_deps(access))
            return access.op_idx;
        prev_idx = access.prev_for_handle;
    }
    return GRAPH_ACCESS_NONE;
}

static size_t graph_sgoc_find_next_handle_producer_op_idx(const std::vector<GraphHandleAccess> &handle_accesses,
                                                           size_t access_idx)
{
    size_t next_idx = access_idx;
    while (next_idx != GRAPH_ACCESS_NONE) {
        if (next_idx >= handle_accesses.size())
            return GRAPH_ACCESS_NONE;
        const GraphHandleAccess &access = handle_accesses[next_idx];
        if (graph_access_is_handle_producer_for_deps(access))
            return access.op_idx;
        next_idx = access.next_for_handle;
    }
    return GRAPH_ACCESS_NONE;
}

void graph_sgoc_add_handle_prefix_task_dependencies(std::vector<GraphOp> &ops,
                                                            const std::vector<GraphHandleAccess> &handle_accesses,
                                                            size_t first_access_idx, size_t last_access_idx,
                                                            size_t target_op_idx)
{
    if (target_op_idx >= ops.size() || first_access_idx >= handle_accesses.size() || last_access_idx >= handle_accesses.size())
        return;

    size_t idx = first_access_idx;
    while (idx != GRAPH_ACCESS_NONE) {
        if (idx >= handle_accesses.size())
            return;
        const GraphHandleAccess &access = handle_accesses[idx];
        if (access.task != nullptr)
            graph_op_add_edge(ops, target_op_idx, access.op_idx);
        if (idx == last_access_idx)
            break;
        idx = access.next_for_handle;
    }
}

void graph_sgoc_add_handle_suffix_read_dependencies(std::vector<GraphOp> &ops,
                                                            const std::vector<GraphHandleAccess> &handle_accesses,
                                                            size_t first_access_idx, size_t producer_op_idx)
{
    if (producer_op_idx >= ops.size() || first_access_idx >= handle_accesses.size())
        return;

    size_t idx = first_access_idx;
    while (idx != GRAPH_ACCESS_NONE) {
        if (idx >= handle_accesses.size())
            return;
        const GraphHandleAccess &access = handle_accesses[idx];
        if (graph_access_mode_is_invalidate(access.mode) || graph_access_mode_is_writer(access.mode))
            break;
        if (access.task != nullptr && graph_access_mode_is_pure_read(access.mode))
            graph_op_add_edge(ops, access.op_idx, producer_op_idx);
        idx = access.next_for_handle;
    }
}

void graph_sgoc_destroy_checkpoint_task(struct starpu_task *task);

struct starpu_task *graph_sgoc_clone_task_for_checkpoint(const struct starpu_task *task)
{
    if (!task)
        return nullptr;

    struct starpu_task *clone = starpu_task_create();
    if (!clone)
        return nullptr;

    clone->cl = task->cl;
    clone->sched_ctx = task->sched_ctx;
    clone->name = task->name;
    clone->file = task->file;
    clone->line = task->line;

    const unsigned nbuf = task->cl ? STARPU_TASK_GET_NBUFFERS(task) : 0;
    clone->nbuffers = task->nbuffers;

    if (task->cl_arg && task->cl_arg_size > 0) {
        void *cl_arg_copy = std::malloc(task->cl_arg_size);
        if (!cl_arg_copy) {
            graph_sgoc_destroy_checkpoint_task(clone);
            return nullptr;
        }
        std::memcpy(cl_arg_copy, task->cl_arg, task->cl_arg_size);
        clone->cl_arg = cl_arg_copy;
        clone->cl_arg_size = task->cl_arg_size;
        clone->cl_arg_free = 1;
    }

    if (task->dyn_handles && nbuf > 0) {
        size_t bytes = nbuf * sizeof(starpu_data_handle_t);
        clone->dyn_handles = static_cast<starpu_data_handle_t *>(std::malloc(bytes));
        if (!clone->dyn_handles) {
            graph_sgoc_destroy_checkpoint_task(clone);
            return nullptr;
        }
        std::memcpy(clone->dyn_handles, task->dyn_handles, bytes);
    } else if (nbuf > 0) {
        std::memcpy(clone->handles, task->handles, nbuf * sizeof(starpu_data_handle_t));
    }

    if ((task->cl && task->cl->nbuffers == STARPU_VARIABLE_NBUFFERS) || task->dyn_modes) {
        size_t bytes = nbuf * sizeof(enum starpu_data_access_mode);
        clone->dyn_modes = static_cast<enum starpu_data_access_mode *>(std::malloc(bytes));
        if (!clone->dyn_modes) {
            graph_sgoc_destroy_checkpoint_task(clone);
            return nullptr;
        }
        if (task->dyn_modes)
            std::memcpy(clone->dyn_modes, task->dyn_modes, bytes);
        else
            std::memcpy(clone->dyn_modes, task->modes, bytes);
    } else if (nbuf > 0) {
        std::memcpy(clone->modes, task->modes, nbuf * sizeof(enum starpu_data_access_mode));
    }

    return clone;
}

void graph_sgoc_destroy_checkpoint_task(struct starpu_task *task)
{
    if (!task)
        return;
    if (task->dyn_modes)
        std::free(task->dyn_modes);
    if (task->dyn_handles)
        std::free(task->dyn_handles);
    if (task->cl_arg_free && task->cl_arg)
        std::free(task->cl_arg);
    std::free(task);
}

size_t graph_sgoc_insert_handle_access_after(std::vector<GraphHandleAccess> &handle_accesses, size_t prev_idx,
                                                     size_t op_idx, starpu_data_handle_t handle, unsigned mode,
                                                     struct starpu_task *task)
{
    GraphHandleAccess access{};
    access.handle = handle;
    access.mode = mode;
    access.task = task;
    access.op_idx = op_idx;
    access.prev_for_handle = prev_idx;
    access.next_for_handle =
        (prev_idx < handle_accesses.size()) ? handle_accesses[prev_idx].next_for_handle : GRAPH_ACCESS_NONE;

    const size_t access_idx = handle_accesses.size();
    handle_accesses.push_back(access);

    if (prev_idx < handle_accesses.size())
        handle_accesses[prev_idx].next_for_handle = access_idx;
    if (access.next_for_handle != GRAPH_ACCESS_NONE && access.next_for_handle < handle_accesses.size())
        handle_accesses[access.next_for_handle].prev_for_handle = access_idx;

    return access_idx;
}

/** \p keys comes from parsed checkpointable activations (handles only). */
bool graph_sgoc_checkpoint_activation_write_matches(starpu_data_handle_t write_handle,
                                                            const std::unordered_set<void *> *keys)
{
    return write_handle && keys && keys->count(static_cast<void *>(write_handle)) != 0;
}

bool graph_sgoc_insert_checkpoint_for_wrr_task(std::vector<GraphOp> &ops,
                                                       std::vector<GraphHandleAccess> &handle_accesses, size_t op_idx,
                                                       struct starpu_task *checkpoint_task, int pin_worker,
                                                       const graph_sgoc_wrr_activation_sub_policy *activation_sub_policy,
                                                       size_t *invalidate_insert_pos_out = nullptr)
{
    if (op_idx >= ops.size())
        return false;

    const GraphOp op = ops[op_idx];
    const GraphOpHandleAccessRef *write_ref = nullptr;
    std::vector<size_t> read_accesses;
    if (!graph_sgoc_checkpoint_wrr_chain_resolve(op, ops, handle_accesses, &write_ref, read_accesses, nullptr, nullptr,
                                                  activation_sub_policy))
        return false;

    const GraphOpHandleAccessRef write_ref_copy = *write_ref;
    const size_t r_fwd_access_idx = read_accesses[0];
    const size_t r_bwd_access_idx = read_accesses[1];
    if (r_fwd_access_idx >= handle_accesses.size() || r_bwd_access_idx >= handle_accesses.size())
        return false;
    (void)r_fwd_access_idx;

    const size_t prev_before_bwd = handle_accesses[r_bwd_access_idx].prev_for_handle;
    if (prev_before_bwd == GRAPH_ACCESS_NONE || prev_before_bwd >= handle_accesses.size())
        return false;

    const size_t r_bwd_op_idx = handle_accesses[r_bwd_access_idx].op_idx;
    std::vector<GraphOpHandleAccessRef> read_refs;
    read_refs.reserve(op.handle_accesses.size());
    for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
        if (graph_access_mode_is_pure_read(ref.mode))
            read_refs.push_back(ref);
    }

    const size_t inv_insert_pos = r_bwd_op_idx;
    if (invalidate_insert_pos_out)
        *invalidate_insert_pos_out = inv_insert_pos;

    GraphOp inv_op{};
    inv_op.kind = GraphOp::INVALIDATE;
    inv_op.task = nullptr;
    inv_op.handle = write_ref_copy.handle;

    ops.insert(ops.begin() + inv_insert_pos, inv_op);
    graph_sgoc_bump_op_graph_indices_after_insert(ops, inv_insert_pos);
    graph_sgoc_bump_handle_access_op_indices_after_insert(handle_accesses, inv_insert_pos);

    const size_t inv_op_idx = inv_insert_pos;
    const size_t inv_access_idx =
        graph_sgoc_insert_handle_access_after(handle_accesses, prev_before_bwd, inv_op_idx, write_ref_copy.handle,
                                               GRAPH_ACCESS_INVALIDATE_RAW, nullptr);
    ops[inv_op_idx].handle_accesses.push_back({write_ref_copy.handle, GRAPH_ACCESS_INVALIDATE_RAW, inv_access_idx});

    const size_t checkpoint_insert_pos = inv_insert_pos + 1;

    GraphOp checkpoint_op{};
    checkpoint_op.kind = GraphOp::TASK;
    checkpoint_op.task = checkpoint_task;
    checkpoint_op.graph_stage_subiteration_valid = op.graph_stage_subiteration_valid;
    checkpoint_op.graph_stage_subiteration = op.graph_stage_subiteration;
    checkpoint_op.graph_stage_batch_iteration_valid = op.graph_stage_batch_iteration_valid;
    checkpoint_op.graph_stage_batch_iteration = op.graph_stage_batch_iteration;

    ops.insert(ops.begin() + checkpoint_insert_pos, checkpoint_op);
    graph_sgoc_bump_op_graph_indices_after_insert(ops, checkpoint_insert_pos);
    graph_sgoc_bump_handle_access_op_indices_after_insert(handle_accesses, checkpoint_insert_pos);

    const size_t checkpoint_op_idx = checkpoint_insert_pos;

    GraphOp &w2 = ops[checkpoint_op_idx];
    const size_t primary_access_idx =
        graph_sgoc_insert_handle_access_after(handle_accesses, inv_access_idx, checkpoint_op_idx, write_ref_copy.handle,
                                               write_ref_copy.mode, w2.task);
    w2.handle_accesses.push_back({write_ref_copy.handle, write_ref_copy.mode, primary_access_idx});

    if (pin_worker >= 0)
        w2.predicted_exec_time =
            graph_sgoc_predicted_exec_time_us_for_pinned_worker(w2.task, pin_worker, w2.task->sched_ctx);

    for (const GraphOpHandleAccessRef &ref : read_refs) {
        if (ref.access_idx >= handle_accesses.size())
            continue;

        const size_t inserted_idx =
            graph_sgoc_insert_handle_access_after(handle_accesses, ref.access_idx, checkpoint_op_idx, ref.handle,
                                                   ref.mode, w2.task);
        w2.handle_accesses.push_back({ref.handle, ref.mode, inserted_idx});

        const size_t prev_producer_op_idx =
            graph_sgoc_find_prev_handle_producer_op_idx(handle_accesses, handle_accesses[inserted_idx].prev_for_handle);
        graph_op_add_edge(ops, checkpoint_op_idx, prev_producer_op_idx);

        const size_t next_producer_op_idx =
            graph_sgoc_find_next_handle_producer_op_idx(handle_accesses, handle_accesses[inserted_idx].next_for_handle);
        if (next_producer_op_idx != GRAPH_ACCESS_NONE && next_producer_op_idx < ops.size())
            graph_op_add_edge(ops, next_producer_op_idx, checkpoint_op_idx);
    }

    graph_sgoc_add_handle_prefix_task_dependencies(ops, handle_accesses, write_ref_copy.access_idx, prev_before_bwd,
                                                    inv_op_idx);
    graph_op_add_edge(ops, checkpoint_op_idx, inv_op_idx);
    graph_sgoc_add_handle_suffix_read_dependencies(ops, handle_accesses, r_bwd_access_idx, checkpoint_op_idx);

    return true;
}

} /* namespace graph_sgoc_bundle */
