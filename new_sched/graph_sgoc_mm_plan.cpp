/* Linear GPU memory planner (Belady-style offload / prefetch hints). */
#include "graph_sgoc_common.hpp"
#include "graph_sgoc_bundle_dag.hpp"
#include "graph_sgoc_bundle_mm.hpp"
#include "graph_sgoc_bundle_topo.hpp"

namespace graph_sgoc_bundle {

std::int64_t graph_sched_sum_unique_handle_bytes(const std::vector<starpu_data_handle_t> &handles)
{
    std::unordered_set<void *> seen;
    std::int64_t sum = 0;
    for (starpu_data_handle_t h : handles) {
        if (!h)
            continue;
        void *k = static_cast<void *>(h);
        if (!seen.insert(k).second)
            continue;
        sum += static_cast<std::int64_t>(starpu_data_get_size(h));
    }
    return sum;
}

/** Default 1: enable automatic offload-before-task planning (linear topo simulation vs GPU budget). */
int graph_sched_mem_offload_auto_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_MEM_OFFLOAD_AUTO");
    if (!e || !e[0])
        return 1;
    return atoi(e) != 0;
}

/** Fraction of graph_pinned_worker_max_allowed_memory_bytes for MM planning (SGOC default 1.0; use env to trim). */
double graph_sched_mem_budget_fraction_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_MEM_BUDGET_FRACTION");
    if (!e || !e[0])
        return 1.0;
    const double x = std::strtod(e, nullptr);
    return (x > 0.0 && x <= 1.0) ? x : 1.0;
}

/** If > 0, overrides policy budget for planning/debug (bytes). */
std::int64_t graph_sched_force_mem_budget_bytes_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_FORCE_MEM_BUDGET_BYTES");
    if (!e || !e[0])
        return -1;
    char *end = nullptr;
    const long long v = std::strtoll(e, &end, 10);
    if (end == e || v < 0)
        return -1;
    return static_cast<std::int64_t>(v);
}

/**
 * SGOC-only: if set to a non-negative decimal, overrides the planner GPU memory budget (bytes) after fraction scaling
 * of the pinned-node allowance would otherwise apply. Unset or invalid leaves the computed budget unchanged.
 */
std::int64_t graph_sched_sgoc_budget_bytes_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_SGOC_BUDGET_BYTES");
    if (!e || !e[0])
        return -1;
    char *end = nullptr;
    const long long v = std::strtoll(e, &end, 10);
    if (end == e || v < 0)
        return -1;
    return static_cast<std::int64_t>(v);
}

/**
 * Default 0: when 1, replay may emit StarPU prefetch/offload actions from the GPU MM plan built at flush.
 */
int graph_sched_mm_execute_hints_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_MM_EXECUTE_HINTS");
    if (!e || !e[0])
        return 0;
    return atoi(e) != 0;
}

/** Non-zero: stderr MM prefetch/offload advance + replay counters (STARPU_GRAPH_SCHED_SGOC_MEM_DEBUG). */
int graph_sched_sgoc_mem_debug_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_SGOC_MEM_DEBUG");
    if (!e || !e[0])
        return 0;
    return atoi(e) != 0;
}

/** Non-zero: one stderr summary per SGOC flush comparing MM plan lists to replay hook activity. */
int graph_sched_mm_order_trace_env(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_MM_ORDER_TRACE");
    if (!e || !e[0])
        return 0;
    return atoi(e) != 0;
}

/**
 * Projected GPU footprint immediately before executing \p op (TASK): same rules as graph_sched_op_memory_delta /
 * graph_sched_op_apply_memory_effect — no handle classification by P/G/A/S.
 */
std::int64_t graph_sched_gpu_mm_projected_bytes_before_task(std::int64_t base_bytes,
                                                                   const std::unordered_set<void *> &resident,
                                                                   const std::unordered_set<void *> &offloaded,
                                                                   const GraphOp &op)
{
    if (op.kind != GraphOp::TASK)
        return base_bytes;
    std::int64_t proj = base_bytes;
    std::unordered_set<void *> bump_once;
    bump_once.reserve(op.handle_accesses.size() * 2 + 8u);
    for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
        if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
            continue;
        void *p = static_cast<void *>(ref.handle);
        if (!bump_once.insert(p).second)
            continue;
        if (offloaded.count(p) != 0) {
            proj += static_cast<std::int64_t>(starpu_data_get_size(ref.handle));
            continue;
        }
        if (resident.count(p) != 0)
            continue;
        proj += static_cast<std::int64_t>(starpu_data_get_size(ref.handle));
    }
    return proj;
}

/**
 * True when the next graph operation that touches \p key at the handle level, after topo slot \p from_ti, is a
 * standalone INVALIDATE on \p key, with no TASK in between that references \p key (non-invalidate). Then a RAM offload
 * before the pressure point is unnecessary: dropping the GPU copy matches simulation (invalidate will discard the
 * version anyway).
 */
bool graph_sched_gpu_mm_next_graph_touch_is_invalidate_without_intervening_task(
    const std::vector<GraphOp> &ops, const std::vector<size_t> &topo_order, void *key, size_t from_ti)
{
    for (size_t tj = from_ti + 1; tj < topo_order.size(); ++tj) {
        const size_t opi = topo_order[tj];
        if (opi >= ops.size())
            continue;
        const GraphOp &oj = ops[opi];
        if (oj.kind == GraphOp::INVALIDATE) {
            if (oj.handle && static_cast<void *>(oj.handle) == key)
                return true;
            continue;
        }
        if (oj.kind == GraphOp::TASK) {
            for (const GraphOpHandleAccessRef &ref : oj.handle_accesses) {
                if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                    continue;
                if (static_cast<void *>(ref.handle) == key)
                    return false;
            }
        }
    }
    return false;
}

void graph_sched_gpu_mm_build_task_topo_appearances(const std::vector<GraphOp> &ops,
                                                           const std::vector<size_t> &topo_order,
                                                           std::unordered_map<void *, std::vector<size_t>> &out)
{
    out.clear();
    for (size_t ti = 0; ti < topo_order.size(); ++ti) {
        const size_t opi = topo_order[ti];
        if (opi >= ops.size())
            continue;
        const GraphOp &op = ops[opi];
        if (op.kind != GraphOp::TASK)
            continue;
        std::unordered_set<void *> seen;
        seen.reserve(op.handle_accesses.size() + 4u);
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                continue;
            void *p = static_cast<void *>(ref.handle);
            if (!seen.insert(p).second)
                continue;
            out[p].push_back(ti);
        }
    }
}

/**
 * Earliest topo index at which this handle is needed **after** \p ti (Belady-style distance).
 * If there is no further appearance in the current batch pass, the next need is the first appearance in a
 * virtually repeated batch: \p num_topo_tasks + v.front(). Handles that never appear in any TASK have no bound
 * on next need here — return UINT64_MAX so they rank as furthest (best offload victims).
 */
std::uint64_t graph_sched_gpu_mm_next_need_topo_index(const std::vector<size_t> &v, size_t ti,
                                                             size_t num_topo_tasks)
{
    const auto ub = std::upper_bound(v.begin(), v.end(), ti);
    if (ub != v.end())
        return static_cast<std::uint64_t>(*ub);
    if (!v.empty())
        return static_cast<std::uint64_t>(num_topo_tasks) + static_cast<std::uint64_t>(v.front());
    return std::numeric_limits<std::uint64_t>::max();
}

/**
 * From simulated per-topo-slot handle lists (offload-before-task or evict-only-before-task), derive ordered post_exec
 * lists: each handle is scheduled after the TASK where it is last used before that slot (reverse topo walk; optional
 * wrap passes without ingesting new marks until \p pending clears).
 */
void graph_sched_gpu_mm_derive_post_exec_offload_order(const std::vector<GraphOp> &ops,
                                                              const std::vector<size_t> &topo_order,
                                                              const std::vector<std::vector<void *>> &offload_before_topo,
                                                              std::vector<std::vector<void *>> &post_exec_out)
{
    const size_t T = topo_order.size();
    post_exec_out.assign(T, {});
    std::unordered_set<void *> pending;
    pending.reserve(64u);

    auto match_task = [&](size_t ti) {
        const size_t opi = topo_order[ti];
        if (opi >= ops.size())
            return;
        const GraphOp &op = ops[opi];
        if (op.kind != GraphOp::TASK)
            return;
        std::unordered_set<void *> seen;
        seen.reserve(op.handle_accesses.size() + 4u);
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                continue;
            void *p = static_cast<void *>(ref.handle);
            if (!seen.insert(p).second)
                continue;
            if (pending.erase(p))
                post_exec_out[ti].push_back(p);
        }
    };

    auto reverse_pass = [&](bool add_offload_marks) {
        for (ssize_t sti = static_cast<ssize_t>(T) - 1; sti >= 0; --sti) {
            const size_t ti = static_cast<size_t>(sti);
            const size_t opi = topo_order[ti];
            if (opi >= ops.size())
                continue;
            const GraphOp &op = ops[opi];
            if (add_offload_marks && op.kind == GraphOp::TASK && ti < offload_before_topo.size()) {
                for (void *h : offload_before_topo[ti]) {
                    if (h)
                        pending.insert(h);
                }
            }
            match_task(ti);
        }
    };

    reverse_pass(true);
    while (!pending.empty()) {
        const size_t before = pending.size();
        reverse_pass(false);
        if (pending.size() == before)
            break;
    }
}

/**
 * From simulated prefetch-before-consumer lists, assign each prefetch to the earliest prior topo TASK slot where
 * modeled GPU bytes after that slot plus the handle size still fit \p budget_bytes (simulated headroom). Falls back
 * to the consumer slot when no earlier TASK qualifies. Preserves global order (increasing consumer topo, then access
 * order). Runtime issues these at \c pop_task of the anchor TASK.
 */
void graph_sched_gpu_mm_derive_anchor_pop_prefetch_order(
    const std::vector<GraphOp> &ops, const std::vector<size_t> &topo_order,
    const std::vector<std::vector<void *>> &prefetch_before_topo,
    const std::vector<std::int64_t> &bytes_after_each_topo_slot,
    const std::unordered_map<void *, size_t> &first_offload_topo_slot, std::int64_t budget_bytes,
    std::vector<std::vector<void *>> &pop_prefetch_out)
{
    const size_t T = topo_order.size();
    pop_prefetch_out.assign(T, {});
    if (prefetch_before_topo.size() != T || bytes_after_each_topo_slot.size() != T)
        return;
    for (size_t ti = 0; ti < T; ++ti) {
        if (ti >= prefetch_before_topo.size())
            break;
        for (void *h_void : prefetch_before_topo[ti]) {
            if (!h_void)
                continue;
            starpu_data_handle_t h = static_cast<starpu_data_handle_t>(h_void);
            const std::int64_t sz = static_cast<std::int64_t>(starpu_data_get_size(h));
            size_t t_start = 0;
            const auto fo = first_offload_topo_slot.find(h_void);
            if (fo != first_offload_topo_slot.end())
                t_start = fo->second;
            size_t anchor = ti;
            if (budget_bytes > 0 && sz > 0) {
                for (size_t tj = t_start; tj < ti; ++tj) {
                    const size_t opi = topo_order[tj];
                    if (opi >= ops.size() || ops[opi].kind != GraphOp::TASK)
                        continue;
                    if (bytes_after_each_topo_slot[tj] + sz <= budget_bytes) {
                        anchor = tj;
                        break;
                    }
                }
            }
            pop_prefetch_out[anchor].push_back(h_void);
        }
    }
}

static size_t sgoc_mm_debug_min_anchor_topo(const std::vector<std::vector<void *>> &per_topo, size_t T, void *p)
{
    size_t best = static_cast<size_t>(-1);
    for (size_t aj = 0; aj < T && aj < per_topo.size(); ++aj) {
        for (void *k : per_topo[aj]) {
            if (k == p) {
                if (best == static_cast<size_t>(-1) || aj < best)
                    best = aj;
                break;
            }
        }
    }
    return best;
}

void graph_sched_sgoc_log_mm_plan_advance_debug(size_t topo_slots, const graph_sched_gpu_memory_manager &mm)
{
    const size_t T = topo_slots;
    std::uint64_t pf_n = 0, pf_early = 0, pf_slot_sum = 0, pf_early_bytes = 0, pf_all_bytes = 0;
    for (size_t ti = 0; ti < T; ++ti) {
        if (ti >= mm.topo_prefetch_before_task.size())
            break;
        for (void *p : mm.topo_prefetch_before_task[ti]) {
            if (!p)
                continue;
            const size_t anchor = sgoc_mm_debug_min_anchor_topo(mm.topo_pre_exec_prefetch_order, T, p);
            const size_t use_anchor = (anchor == static_cast<size_t>(-1)) ? ti : anchor;
            const std::int64_t sz = starpu_data_get_size(static_cast<starpu_data_handle_t>(p));
            pf_n++;
            pf_all_bytes += static_cast<std::uint64_t>(sz);
            if (use_anchor < ti) {
                pf_early++;
                pf_slot_sum += (ti - use_anchor);
                pf_early_bytes += static_cast<std::uint64_t>(sz);
            }
        }
    }

    std::uint64_t of_n = 0, of_early = 0, of_slot_sum = 0, of_early_bytes = 0, of_all_bytes = 0;
    for (size_t ti = 0; ti < T; ++ti) {
        if (ti >= mm.topo_offload_before_task.size())
            break;
        for (void *p : mm.topo_offload_before_task[ti]) {
            if (!p)
                continue;
            const size_t anchor = sgoc_mm_debug_min_anchor_topo(mm.topo_post_exec_offload_order, T, p);
            const size_t use_anchor = (anchor == static_cast<size_t>(-1)) ? ti : anchor;
            const std::int64_t sz = starpu_data_get_size(static_cast<starpu_data_handle_t>(p));
            of_n++;
            of_all_bytes += static_cast<std::uint64_t>(sz);
            if (use_anchor < ti) {
                of_early++;
                of_slot_sum += (ti - use_anchor);
                of_early_bytes += static_cast<std::uint64_t>(sz);
            }
        }
    }

    const double pf_avg = (pf_early > 0) ? static_cast<double>(pf_slot_sum) / static_cast<double>(pf_early) : 0.0;
    const double of_avg = (of_early > 0) ? static_cast<double>(of_slot_sum) / static_cast<double>(of_early) : 0.0;
    std::cerr << "sgoc_mem_debug: mm_plan topo_slots=" << T << " prefetch_consumer_handles=" << pf_n
              << " prefetch_early_anchor=" << pf_early << " prefetch_early_bytes=" << pf_early_bytes
              << " prefetch_total_bytes=" << pf_all_bytes << " prefetch_avg_topo_advance_early=" << std::fixed
              << std::setprecision(2) << pf_avg << "\n";
    std::cerr << "sgoc_mem_debug: mm_plan offload_before_task_marks=" << of_n << " offload_post_exec_before_consumer="
              << of_early << " offload_early_bytes=" << of_early_bytes << " offload_total_bytes=" << of_all_bytes
              << " offload_avg_topo_advance_early=" << of_avg << std::endl;
}

/**
 * After greedy (or capture) topo order is fixed, simulate linear replay; invalidate / pure-W footprint matches
 * graph_sched_op_memory_*. Offload victim selection is role-agnostic (any GPU-resident buffer). Planning only.
 * Victim = furthest next TASK appearance (Belady); if the next graph touch of that handle is \c invalidate_submit
 * with no intervening TASK on the handle, skip RAM offload and record GPU evict-only (see \c evict_only_mark_events).
 * Need times extend into a virtual copy of the batch when there is no TASK use left in the current pass; ties break
 * by last TASK appearance index in one batch.
 */
void graph_sched_gpu_mm_plan_linear_topo_offloads(const std::vector<GraphOp> &ops,
                                                         const std::vector<GraphHandleAccess> &handle_accesses,
                                                         const std::vector<size_t> &topo_order,
                                                         const graph_sched_captured_handle_groups &groups,
                                                         std::int64_t budget_bytes,
                                                         graph_sched_gpu_memory_manager &mm,
                                                         std::vector<void *> &unique_offload_handles_out, int verbose,
                                                         const std::unordered_set<void *> *starpu_gpu_resident_truth)
{
    mm.offload_prefetch_fifo.clear();
    unique_offload_handles_out.clear();
    mm.budget_bytes = budget_bytes;
    mm.sum_s_unique_bytes = graph_sched_sum_unique_handle_bytes(groups.states);
    mm.offload_mark_events = 0;
    mm.marked_offload_unique = 0;
    mm.peak_simulated_bytes = 0;
    mm.initial_resident_bytes = 0;
    mm.peak_after_plan_bytes = 0;
    mm.topo_offload_before_task.clear();
    mm.topo_prefetch_before_task.clear();
    mm.topo_post_exec_offload_order.clear();
    mm.topo_post_exec_evict_gpu_only_order.clear();
    mm.topo_pre_exec_prefetch_order.clear();
    mm.evict_only_mark_events = 0;

    if (budget_bytes <= 0 || topo_order.empty()) {
        if (verbose >= 1 && budget_bytes <= 0)
            std::cerr << "sgoc: gpu_memory_manager: budget_bytes<=0; skip linear offload planning"
                      << std::endl;
        return;
    }

    std::unordered_set<void *> resident;
    std::unordered_set<void *> offloaded;
    std::int64_t current_bytes = 0;

    auto resident_insert = [&](starpu_data_handle_t h) {
        if (!h)
            return;
        void *p = static_cast<void *>(h);
        if (!resident.insert(p).second)
            return;
        current_bytes += static_cast<std::int64_t>(starpu_data_get_size(h));
    };

    std::vector<starpu_data_handle_t> unique_handles;
    graph_sched_collect_unique_handles(handle_accesses, unique_handles);
    if (starpu_gpu_resident_truth) {
        for (starpu_data_handle_t h : unique_handles) {
            if (!h || !graph_sched_handle_live_before_graph(handle_accesses, h))
                continue;
            void *p = static_cast<void *>(h);
            if (starpu_gpu_resident_truth->count(p))
                resident_insert(h);
        }
        for (starpu_data_handle_t h : groups.parameters) {
            if (h && starpu_gpu_resident_truth->count(static_cast<void *>(h)))
                resident_insert(h);
        }
        for (starpu_data_handle_t h : groups.states) {
            if (h && starpu_gpu_resident_truth->count(static_cast<void *>(h)))
                resident_insert(h);
        }
    } else {
        for (starpu_data_handle_t h : unique_handles) {
            if (!h || !graph_sched_handle_live_before_graph(handle_accesses, h))
                continue;
            resident_insert(h);
        }
        for (starpu_data_handle_t h : groups.parameters)
            resident_insert(h);
        for (starpu_data_handle_t h : groups.states)
            resident_insert(h);
    }

    mm.initial_resident_bytes = current_bytes;

    std::unordered_map<void *, std::vector<size_t>> appearances;
    graph_sched_gpu_mm_build_task_topo_appearances(ops, topo_order, appearances);
    static const std::vector<size_t> empty_topo_app;

    const size_t topo_slots = topo_order.size();
    std::vector<std::vector<void *>> offload_before_topo(topo_slots);
    std::vector<std::vector<void *>> evict_only_before_topo(topo_slots);
    std::vector<std::vector<void *>> prefetch_before_topo(topo_slots);
    std::vector<std::int64_t> bytes_after_each_topo_slot(topo_slots, 0);
    std::unordered_map<void *, size_t> first_offload_topo_slot;

    /* Handles touched by the current TASK (including WRR rematerialization clones) are excluded from Belady victim
     * choice at this topo slot. Remat often re-touches the same activation handles as downstream consumers, so the
     * evictable resident set can shrink in ways that offset any hoped-for extra headroom — offload event counts may
     * match a checkpoint-free run even under a tight budget. */
    std::unordered_set<void *> forbidden;
    std::vector<void *> fifo_unique;
    std::unordered_set<void *> fifo_seen;

    std::int64_t peak_track = current_bytes;

    auto note_peak = [&]() {
        if (current_bytes > peak_track)
            peak_track = current_bytes;
    };

    note_peak();

    for (size_t ti = 0; ti < topo_order.size(); ++ti) {
        const size_t opi = topo_order[ti];
        if (opi >= ops.size()) {
            bytes_after_each_topo_slot[ti] = current_bytes;
            continue;
        }
        const GraphOp &op = ops[opi];

        if (op.kind == GraphOp::INVALIDATE) {
            starpu_data_handle_t ih = op.handle;
            if (ih) {
                void *p = static_cast<void *>(ih);
                if (resident.erase(p))
                    current_bytes -= static_cast<std::int64_t>(starpu_data_get_size(ih));
                offloaded.erase(p);
            }
            note_peak();
            bytes_after_each_topo_slot[ti] = current_bytes;
            continue;
        }

        if (op.kind != GraphOp::TASK) {
            bytes_after_each_topo_slot[ti] = current_bytes;
            continue;
        }

        forbidden.clear();
        forbidden.reserve(op.handle_accesses.size() + 4u);
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                continue;
            forbidden.insert(static_cast<void *>(ref.handle));
        }

        while (true) {
            const std::int64_t proj =
                graph_sched_gpu_mm_projected_bytes_before_task(current_bytes, resident, offloaded, op);
            if (proj <= budget_bytes)
                break;

            void *best_k = nullptr;
            std::uint64_t best_need = 0;
            size_t best_last_in_batch = 0;
            bool has_best = false;
            const size_t num_topo_tasks = topo_order.size();
            for (void *k : resident) {
                if (forbidden.count(k) != 0)
                    continue;
                const auto it = appearances.find(k);
                const std::vector<size_t> &app = (it != appearances.end()) ? it->second : empty_topo_app;
                const std::uint64_t need =
                    graph_sched_gpu_mm_next_need_topo_index(app, ti, num_topo_tasks);
                const size_t last_in_batch = app.empty() ? static_cast<size_t>(0) : app.back();
                if (!has_best || need > best_need
                    || (need == best_need && last_in_batch > best_last_in_batch)
                    || (need == best_need && last_in_batch == best_last_in_batch && k > best_k)) {
                    best_need = need;
                    best_last_in_batch = last_in_batch;
                    best_k = k;
                    has_best = true;
                }
            }

            if (!has_best) {
                if (verbose >= 1)
                    std::cerr << "sgoc: gpu_memory_manager: cannot offload further (no eligible GPU-resident "
                                 "handle) at topo_idx="
                              << ti << " projected_before_task_bytes=" << proj << " budget=" << budget_bytes << std::endl;
                break;
            }

            starpu_data_handle_t hk = static_cast<starpu_data_handle_t>(best_k);
            const std::int64_t sz = static_cast<std::int64_t>(starpu_data_get_size(hk));
            resident.erase(best_k);
            current_bytes -= sz;
            if (graph_sched_gpu_mm_next_graph_touch_is_invalidate_without_intervening_task(ops, topo_order, best_k, ti)) {
                mm.evict_only_mark_events++;
                evict_only_before_topo[ti].push_back(best_k);
            } else {
                offloaded.insert(best_k);
                mm.offload_mark_events++;
                offload_before_topo[ti].push_back(best_k);
                if (first_offload_topo_slot.find(best_k) == first_offload_topo_slot.end())
                    first_offload_topo_slot[best_k] = ti;
                if (fifo_seen.insert(best_k).second)
                    fifo_unique.push_back(best_k);
            }
        }

        std::unordered_set<void *> pf_once;
        pf_once.reserve(op.handle_accesses.size() + 4u);
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || graph_access_mode_is_invalidate(ref.mode))
                continue;
            void *p = static_cast<void *>(ref.handle);
            if (offloaded.count(p) == 0)
                continue;
            if (!pf_once.insert(p).second)
                continue;
            prefetch_before_topo[ti].push_back(p);
            current_bytes += static_cast<std::int64_t>(starpu_data_get_size(ref.handle));
            resident.insert(p);
            offloaded.erase(p);
        }

        const std::int64_t d = graph_sched_op_memory_delta_for_resident(op, resident);
        current_bytes += d;
        graph_sched_op_apply_memory_effect_to_resident(op, resident);
        note_peak();
        bytes_after_each_topo_slot[ti] = current_bytes;
    }

    mm.peak_simulated_bytes = peak_track;
    mm.peak_after_plan_bytes = peak_track;
    mm.marked_offload_unique = static_cast<unsigned>(fifo_unique.size());
    for (void *k : fifo_unique)
        mm.offload_prefetch_fifo.push_back(k);
    unique_offload_handles_out = std::move(fifo_unique);

    mm.topo_offload_before_task = std::move(offload_before_topo);
    mm.topo_prefetch_before_task = std::move(prefetch_before_topo);
    graph_sched_gpu_mm_derive_post_exec_offload_order(ops, topo_order, mm.topo_offload_before_task,
                                                      mm.topo_post_exec_offload_order);
    graph_sched_gpu_mm_derive_post_exec_offload_order(ops, topo_order, evict_only_before_topo,
                                                      mm.topo_post_exec_evict_gpu_only_order);
    graph_sched_gpu_mm_derive_anchor_pop_prefetch_order(ops, topo_order, mm.topo_prefetch_before_task,
                                                        bytes_after_each_topo_slot, first_offload_topo_slot,
                                                        budget_bytes, mm.topo_pre_exec_prefetch_order);

    if (mm.peak_after_plan_bytes > budget_bytes && verbose >= 1) {
        std::cerr << "sgoc: gpu_memory_manager: warning peak_after_plan_bytes=" << mm.peak_after_plan_bytes
                  << " still exceeds budget=" << budget_bytes
                  << " (consider raising STARPU_LIMIT_CUDA*_MEM or budget fraction)" << std::endl;
    }

    if (verbose >= 2) {
        std::cerr << "sgoc: gpu_memory_manager: linear_topo_plan initial_resident_bytes=" << mm.initial_resident_bytes
                  << " peak_simulated_bytes=" << mm.peak_simulated_bytes << " budget=" << budget_bytes
                  << " offload_events=" << mm.offload_mark_events << " marked_offload_unique=" << mm.marked_offload_unique
                  << " evict_only_events=" << mm.evict_only_mark_events << std::endl;
    }
}

void graph_sched_gpu_mm_restore_from_cached_plan(const graph_sched_mem_offload_plan &plan, std::int64_t budget_bytes,
                                                        graph_sched_gpu_memory_manager &mm)
{
    mm.offload_prefetch_fifo.clear();
    mm.peak_simulated_bytes = plan.peak_pga_bytes;
    mm.sum_s_unique_bytes = plan.sum_s_bytes;
    mm.budget_bytes = budget_bytes;
    mm.initial_resident_bytes = 0;
    mm.peak_after_plan_bytes = plan.peak_pga_bytes;
    mm.offload_mark_events = 0;
    mm.topo_offload_before_task = plan.topo_offload_before_task;
    mm.topo_prefetch_before_task = plan.topo_prefetch_before_task;
    mm.topo_post_exec_offload_order = plan.topo_post_exec_offload_order;
    mm.topo_post_exec_evict_gpu_only_order = plan.topo_post_exec_evict_gpu_only_order;
    mm.topo_pre_exec_prefetch_order = plan.topo_pre_exec_prefetch_order;
    for (void *k : plan.s_offload_keys)
        mm.offload_prefetch_fifo.push_back(k);
    mm.marked_offload_unique = static_cast<unsigned>(plan.s_offload_keys.size());
}

/**
 * GPU budget offload-before-task planning under linear replay of \p topo_order (greedy or capture order).
 * For policy_log_name \c "sgoc", each flush replans (no cross-batch reuse of graph_mem_offload_plan); the recorder
 * build may still pass \p outer_batch0_capture for API compatibility.
 */
void graph_sched_apply_gpu_mm_plan_from_capture(const std::vector<GraphOp> &ops,
                                                       const std::vector<GraphHandleAccess> &handle_accesses,
                                                       const std::vector<size_t> &topo_order,
                                                       graph_sched_data *policy_data,
                                                       const graph_sched_captured_handle_groups *captured_for_offload_hints,
                                                       int pin_worker, int vb, bool batch_matches_previous_flush,
                                                       bool outer_batch0_capture,
                                                       std::vector<void *> &s_offload_active_out,
                                                       const std::unordered_set<void *> *starpu_gpu_resident_truth)
{
    s_offload_active_out.clear();
    if (!graph_sched_mem_offload_auto_env() || !policy_data || !captured_for_offload_hints || pin_worker < 0)
        return;

    const bool sgoc_flush = policy_data->policy_log_name && std::strcmp(policy_data->policy_log_name, "sgoc") == 0;
    const bool eff_outer_batch0 = sgoc_flush ? true : outer_batch0_capture;
    const bool eff_batch_match_prev = sgoc_flush ? false : batch_matches_previous_flush;

    std::int64_t mem_budget = policy_data->graph_pinned_worker_max_allowed_memory_bytes;
    const std::int64_t forced_budget = graph_sched_force_mem_budget_bytes_env();
    if (forced_budget >= 0)
        mem_budget = forced_budget;
    if (mem_budget <= 0)
        return;

    mem_budget = static_cast<std::int64_t>(static_cast<double>(mem_budget) * graph_sched_mem_budget_fraction_env());
    {
        const std::int64_t sgoc_b = graph_sched_sgoc_budget_bytes_env();
        if (sgoc_b >= 0)
            mem_budget = sgoc_b;
    }

    std::int64_t mem_sum_s_log = graph_sched_sum_unique_handle_bytes(captured_for_offload_hints->states);

    const bool have_saved_plan = policy_data->graph_mem_offload_plan.valid;

    std::int64_t mem_peak_log = 0;
    bool ran_linear_topo_planner = false;

    if (!eff_outer_batch0) {
        /* Replay only: reuse batch-0 findings; never re-run the memory optimization pass (budget is fixed for the run). */
        if (have_saved_plan) {
            s_offload_active_out = policy_data->graph_mem_offload_plan.s_offload_keys;
            mem_peak_log = policy_data->graph_mem_offload_plan.peak_pga_bytes;
            mem_sum_s_log = policy_data->graph_mem_offload_plan.sum_s_bytes;
            graph_sched_gpu_mm_restore_from_cached_plan(policy_data->graph_mem_offload_plan, mem_budget,
                                                        policy_data->graph_gpu_mm);
        } else {
            policy_data->graph_gpu_mm = graph_sched_gpu_memory_manager{};
            mem_peak_log = 0;
        }
    } else {
        const bool mem_reuse_plan = eff_batch_match_prev && have_saved_plan;

        if (mem_reuse_plan) {
            s_offload_active_out = policy_data->graph_mem_offload_plan.s_offload_keys;
            mem_peak_log = policy_data->graph_mem_offload_plan.peak_pga_bytes;
            mem_sum_s_log = policy_data->graph_mem_offload_plan.sum_s_bytes;
            graph_sched_gpu_mm_restore_from_cached_plan(policy_data->graph_mem_offload_plan, mem_budget,
                                                        policy_data->graph_gpu_mm);
        } else {
            if (!handle_accesses.empty()) {
                graph_sched_gpu_mm_plan_linear_topo_offloads(ops, handle_accesses, topo_order, *captured_for_offload_hints,
                                                             mem_budget, policy_data->graph_gpu_mm, s_offload_active_out,
                                                             vb, starpu_gpu_resident_truth);
                ran_linear_topo_planner = true;
                mem_peak_log = policy_data->graph_gpu_mm.peak_simulated_bytes;
            } else {
                if (vb >= 2)
                    std::cerr << "sgoc: gpu_memory_manager: note handle_accesses_empty; skip linear offload "
                                 "planning"
                              << std::endl;
                policy_data->graph_gpu_mm = graph_sched_gpu_memory_manager{};
                mem_peak_log = 0;
            }
            policy_data->graph_mem_offload_plan.valid = true;
            policy_data->graph_mem_offload_plan.budget_bytes = mem_budget;
            policy_data->graph_mem_offload_plan.peak_pga_bytes = mem_peak_log;
            policy_data->graph_mem_offload_plan.sum_s_bytes = mem_sum_s_log;
            policy_data->graph_mem_offload_plan.s_offload_keys = s_offload_active_out;
            policy_data->graph_mem_offload_plan.topo_offload_before_task = policy_data->graph_gpu_mm.topo_offload_before_task;
            policy_data->graph_mem_offload_plan.topo_prefetch_before_task = policy_data->graph_gpu_mm.topo_prefetch_before_task;
            policy_data->graph_mem_offload_plan.topo_post_exec_offload_order =
                policy_data->graph_gpu_mm.topo_post_exec_offload_order;
            policy_data->graph_mem_offload_plan.topo_post_exec_evict_gpu_only_order =
                policy_data->graph_gpu_mm.topo_post_exec_evict_gpu_only_order;
            policy_data->graph_mem_offload_plan.topo_pre_exec_prefetch_order =
                policy_data->graph_gpu_mm.topo_pre_exec_prefetch_order;
        }
    }

    const bool reused_saved =
        !eff_outer_batch0 ? policy_data->graph_mem_offload_plan.valid
                          : (eff_batch_match_prev && have_saved_plan && !ran_linear_topo_planner);

    if (vb >= 1) {
        std::cerr << "sgoc: gpu_memory_manager: marked_offload_unique="
                  << policy_data->graph_gpu_mm.marked_offload_unique << " offload_mark_events="
                  << policy_data->graph_gpu_mm.offload_mark_events << " evict_only_mark_events="
                  << policy_data->graph_gpu_mm.evict_only_mark_events << " peak_simulated_bytes=" << mem_peak_log
                  << " sum_s=" << mem_sum_s_log << " budget=" << mem_budget << " outer_batch0_capture="
                  << (eff_outer_batch0 ? 1 : 0) << " ran_linear_topo_planner=" << (ran_linear_topo_planner ? 1 : 0)
                  << " reused_saved_plan=" << (reused_saved ? 1 : 0) << std::endl;
    }
    if (vb >= 2) {
        std::cerr << "sgoc: mem_offload_plan: budget_bytes=" << mem_budget
                  << " peak_simulated_bytes=" << mem_peak_log << " sum_s_bytes=" << mem_sum_s_log
                  << " s_offload_n=" << s_offload_active_out.size()
                  << " gpu_mm_marked_unique=" << policy_data->graph_gpu_mm.marked_offload_unique
                  << " gpu_mm_offload_events=" << policy_data->graph_gpu_mm.offload_mark_events << " outer_batch0_capture="
                  << (eff_outer_batch0 ? 1 : 0) << " ran_linear_topo_planner=" << (ran_linear_topo_planner ? 1 : 0)
                  << " reused_saved_plan=" << (reused_saved ? 1 : 0) << std::endl;
    }
}

} /* namespace graph_sgoc_bundle */
