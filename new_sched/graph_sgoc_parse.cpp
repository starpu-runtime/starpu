/* Parse captured ops into P/S/G/A (checkpoint/offload) groups. */
#include "graph_sgoc_common.hpp"
#include "graph_sgoc_bundle_dag.hpp"
#include "graph_sgoc_bundle_parse.hpp"

namespace graph_sgoc_bundle {

/** Bump one of pure_r / pure_w / rw / scratch / other for activation-pattern accounting. */
void graph_sgoc_activation_stat_bump(unsigned mode, unsigned &pure_r, unsigned &pure_w, unsigned &rw,
                                             unsigned &scratch, unsigned &other)
{
    if (graph_access_mode_is_invalidate(mode))
        return;
    if (graph_access_mode_is_pure_read(mode)) {
        pure_r++;
        return;
    }
    if (graph_access_mode_is_pure_write(mode)) {
        pure_w++;
        return;
    }
    if (graph_access_mode_is_read_write(mode)) {
        rw++;
        return;
    }
    if (graph_access_mode_is_pure_scratch(mode)) {
        scratch++;
        return;
    }
    other++;
}

/** Classify pipeline stage from the inner \c graph_stage_subiteration (StarPU nested \c iteration slot).
 * NNTile counts minibatch steps: odd = forward (1,3,5,…), even = backward (2,4,6,…); 0 = prep/clear; \c UINT32_MAX = optimizer.
 * Pipelines that tag exactly 1/2 for the first minibatch only remain compatible (1 odd, 2 even). */
bool graph_sgoc_graph_subiter_is_training_stage(std::uint32_t sub)
{
    constexpr std::uint32_t k_optimizer = std::numeric_limits<std::uint32_t>::max();
    return sub != 0u && sub != k_optimizer;
}

bool graph_sgoc_graph_subiter_is_forward_stage(std::uint32_t sub)
{
    if (!graph_sgoc_graph_subiter_is_training_stage(sub))
        return false;
    return (sub % 2u) == 1u;
}

bool graph_sgoc_graph_subiter_is_backward_stage(std::uint32_t sub)
{
    if (!graph_sgoc_graph_subiter_is_training_stage(sub))
        return false;
    return (sub % 2u) == 0u;
}

/** Per-handle access counts for forward vs backward training stages; see graph_sgoc_parse_captured_data_handles. */
struct FirstMinibatchActivationAgg {
    unsigned f_r = 0, f_w = 0, f_rw = 0, f_sc = 0, f_other = 0;
    unsigned b_r = 0, b_w = 0, b_rw = 0, b_sc = 0, b_other = 0;
};

static bool graph_sgoc_minibatch12_backward_activation_ok(const FirstMinibatchActivationAgg &ag)
{
    return ag.b_r >= 1u && ag.b_w == 0u && ag.b_rw == 0u && ag.b_sc == 0u && ag.b_other == 0u;
}

static bool graph_sgoc_minibatch12_forward_checkpointable(const FirstMinibatchActivationAgg &ag)
{
    return ag.f_w == 1u && ag.f_r >= 1u && ag.f_rw == 0u && ag.f_sc == 0u && ag.f_other == 0u;
}

/** Forward rule for offloadable: same as checkpointable, or at least one ::STARPU_RW (no scratch / other). */
static bool graph_sgoc_minibatch12_forward_offloadable(const FirstMinibatchActivationAgg &ag)
{
    if (ag.f_sc != 0u || ag.f_other != 0u)
        return false;
    if (graph_sgoc_minibatch12_forward_checkpointable(ag))
        return true;
    return ag.f_rw >= 1u;
}

std::int64_t graph_sgoc_sum_handle_vector_bytes(const std::vector<starpu_data_handle_t> &handles)
{
    std::int64_t sum = 0;
    for (starpu_data_handle_t h : handles) {
        if (h)
            sum += static_cast<std::int64_t>(starpu_data_get_size(h));
    }
    return sum;
}

/**
 * Classify captured handles after recording (runs before replay).
 * 1) Optimizer step (subiteration UINT32_MAX): pure ::STARPU_R → \p out.gradients; pure ::STARPU_W / ::STARPU_RW
 *    by pass → parameter/state candidates (see below).
 * 2) Any W/RW candidate that also appears on a task in a \b forward training stage (odd inner subiteration; see
 *    graph_sgoc_graph_subiter_is_forward_stage) becomes \p out.parameters;
 *    remaining W/RW candidates are \p out.states (optimizer-only buffers such as Adam m/v).
 * 3) \p out.activations (checkpointable) and \p out.offloadable_activations: forward vs backward stages **scoped to the
 *    smallest forward and smallest backward training subiterations observed** (typically first inner microbatch: 1+2
 *    in NNTile; **batch / outer-iteration tags are ignored**). This matches legacy subiter-1/2 behavior when those are
 *    1 and 2, and avoids merging all later odd/even subs (which would accumulate multiple forward writes per handle).
 *    Same backward rule (pure R only); forward strict W+R for checkpointable, or ::STARPU_RW for offloadable.
 *    Order: first appearance on a task with the minimum forward subiteration.
 */
void graph_sgoc_parse_captured_data_handles(const std::vector<GraphOp> &ops,
                                                    graph_sgoc_captured_handle_groups &out, int verbose)
{
    out.parameters.clear();
    out.gradients.clear();
    out.states.clear();
    out.activations.clear();
    out.offloadable_activations.clear();
    out.activation_checkpoint_per_inner_batch = false;
    out.checkpointable_activation_slots.clear();

    constexpr std::uint32_t k_optimizer_subiter = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t min_fwd_train = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t min_bwd_train = std::numeric_limits<std::uint32_t>::max();
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task || !op.graph_stage_subiteration_valid)
            continue;
        const std::uint32_t sub = op.graph_stage_subiteration;
        if (graph_sgoc_graph_subiter_is_forward_stage(sub))
            min_fwd_train = std::min(min_fwd_train, sub);
        if (graph_sgoc_graph_subiter_is_backward_stage(sub))
            min_bwd_train = std::min(min_bwd_train, sub);
    }
    out.activation_checkpoint_min_pair_valid =
        (min_fwd_train != std::numeric_limits<std::uint32_t>::max()
         && min_bwd_train != std::numeric_limits<std::uint32_t>::max());
    if (out.activation_checkpoint_min_pair_valid) {
        out.activation_checkpoint_min_forward_sub = min_fwd_train;
        out.activation_checkpoint_min_backward_sub = min_bwd_train;
        out.activation_forward_backward_delta_sub = min_bwd_train - min_fwd_train;
    } else {
        out.activation_checkpoint_min_forward_sub = 0u;
        out.activation_checkpoint_min_backward_sub = 0u;
        out.activation_forward_backward_delta_sub = 0u;
    }

    bool in_optimizer_region = false;
    unsigned optimizer_pass = 0;
    std::unordered_set<void *> optimizer_candidate_keys;
    std::vector<starpu_data_handle_t> optimizer_candidates_ordered;
    std::unordered_set<void *> gradient_read_keys;

    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;

        const bool valid = op.graph_stage_subiteration_valid;
        const std::uint32_t sub = op.graph_stage_subiteration;
        const bool is_optimizer = valid && sub == k_optimizer_subiter;

        if (valid && !is_optimizer)
            in_optimizer_region = false;

        if (is_optimizer) {
            if (!in_optimizer_region) {
                optimizer_pass++;
                in_optimizer_region = true;
            }
            for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
                if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                    continue;
                const unsigned mode = ref.mode;
                void *key = static_cast<void *>(ref.handle);
                if (graph_access_mode_is_pure_read(mode)) {
                    if (gradient_read_keys.insert(key).second)
                        out.gradients.push_back(ref.handle);
                    continue;
                }
                bool take = false;
                if (optimizer_pass == 1u)
                    take = graph_access_mode_is_pure_write(mode) || graph_access_mode_is_read_write(mode);
                else
                    take = graph_access_mode_is_read_write(mode);
                if (!take)
                    continue;
                if (optimizer_candidate_keys.insert(key).second)
                    optimizer_candidates_ordered.push_back(ref.handle);
            }
        }
    }

    std::unordered_set<void *> first_forward_handle_keys;
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        if (!out.activation_checkpoint_min_pair_valid || !op.graph_stage_subiteration_valid)
            continue;
        if (op.graph_stage_subiteration != out.activation_checkpoint_min_forward_sub)
            continue;
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                continue;
            first_forward_handle_keys.insert(static_cast<void *>(ref.handle));
        }
    }

    out.parameters.reserve(optimizer_candidates_ordered.size());
    out.states.reserve(optimizer_candidates_ordered.size());
    for (starpu_data_handle_t h : optimizer_candidates_ordered) {
        if (first_forward_handle_keys.count(static_cast<void *>(h)))
            out.parameters.push_back(h);
        else
            out.states.push_back(h);
    }

    std::unordered_map<void *, FirstMinibatchActivationAgg> activation_agg;

    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        if (!op.graph_stage_subiteration_valid || !out.activation_checkpoint_min_pair_valid)
            continue;
        const std::uint32_t sub = op.graph_stage_subiteration;
        if (sub != out.activation_checkpoint_min_forward_sub && sub != out.activation_checkpoint_min_backward_sub)
            continue;
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                continue;
            void *key = static_cast<void *>(ref.handle);
            FirstMinibatchActivationAgg &ag = activation_agg[key];
            if (sub == out.activation_checkpoint_min_forward_sub)
                graph_sgoc_activation_stat_bump(ref.mode, ag.f_r, ag.f_w, ag.f_rw, ag.f_sc, ag.f_other);
            else
                graph_sgoc_activation_stat_bump(ref.mode, ag.b_r, ag.b_w, ag.b_rw, ag.b_sc, ag.b_other);
        }
    }

    if (verbose >= 3) {
        size_t fw_gt1 = 0;
        for (const auto &e : activation_agg) {
            if (e.second.f_w > 1u)
                fw_gt1++;
        }
        std::cerr << "sgoc: handle_parser: activation_agg_unique_handles=" << activation_agg.size()
                  << " forward_pure_W_gt1=" << fw_gt1 << " activation_min_fwd_sub=" << out.activation_checkpoint_min_forward_sub
                  << " min_bwd_sub=" << out.activation_checkpoint_min_backward_sub
                  << " (first forward/backward pair; batch tags ignored)" << std::endl;
    }

    std::unordered_set<void *> checkpointable_activation_keys;
    std::unordered_set<void *> offloadable_activation_keys;
    for (const auto &entry : activation_agg) {
        const FirstMinibatchActivationAgg &ag = entry.second;
        if (!graph_sgoc_minibatch12_backward_activation_ok(ag))
            continue;
        if (graph_sgoc_minibatch12_forward_offloadable(ag))
            offloadable_activation_keys.insert(entry.first);
        if (graph_sgoc_minibatch12_forward_checkpointable(ag))
            checkpointable_activation_keys.insert(entry.first);
    }

    std::unordered_set<void *> activation_ordered_seen;
    std::unordered_set<void *> offloadable_ordered_seen;
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        if (!out.activation_checkpoint_min_pair_valid || !op.graph_stage_subiteration_valid)
            continue;
        if (op.graph_stage_subiteration != out.activation_checkpoint_min_forward_sub)
            continue;
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                continue;
            void *key = static_cast<void *>(ref.handle);
            if (checkpointable_activation_keys.count(key) && activation_ordered_seen.insert(key).second)
                out.activations.push_back(ref.handle);
            if (offloadable_activation_keys.count(key) && offloadable_ordered_seen.insert(key).second)
                out.offloadable_activations.push_back(ref.handle);
        }
    }

    if (verbose >= 3) {
        const std::int64_t bp = graph_sgoc_sum_handle_vector_bytes(out.parameters);
        const std::int64_t bg = graph_sgoc_sum_handle_vector_bytes(out.gradients);
        const std::int64_t bs = graph_sgoc_sum_handle_vector_bytes(out.states);
        const std::int64_t ba = graph_sgoc_sum_handle_vector_bytes(out.activations);
        const std::int64_t boo = graph_sgoc_sum_handle_vector_bytes(out.offloadable_activations);
        const std::int64_t btot = bp + bg + bs + ba + boo;
        std::cerr << "sgoc: handle_parser: parameters n=" << out.parameters.size() << " bytes=" << bp
                  << " gradients n=" << out.gradients.size() << " bytes=" << bg
                  << " states n=" << out.states.size() << " bytes=" << bs
                  << " activations_cp n=" << out.activations.size() << " bytes=" << ba
                  << " activations_off n=" << out.offloadable_activations.size() << " bytes=" << boo
                  << " total_bytes=" << btot << std::endl;
    }
}
bool graph_sgoc_infer_batch_capture_context(const std::vector<GraphOp> &ops, bool *has_batch_tags_out,
                                                    std::uint32_t *batch_value_out)
{
    if (has_batch_tags_out)
        *has_batch_tags_out = false;
    bool saw_task = false;
    bool any_valid = false;
    bool any_invalid = false;
    std::uint32_t v = 0;
    bool v_set = false;
    for (const GraphOp &op : ops) {
        if (op.kind != GraphOp::TASK || !op.task)
            continue;
        saw_task = true;
        if (op.graph_stage_batch_iteration_valid) {
            any_valid = true;
            if (!v_set) {
                v = op.graph_stage_batch_iteration;
                v_set = true;
            } else if (op.graph_stage_batch_iteration != v)
                return false;
        } else {
            any_invalid = true;
        }
    }
    if (!saw_task)
        return true;
    if (any_valid && any_invalid)
        return false;
    if (any_valid && has_batch_tags_out && batch_value_out) {
        *has_batch_tags_out = true;
        *batch_value_out = v;
    }
    return true;
}


} /* namespace graph_sgoc_bundle */
