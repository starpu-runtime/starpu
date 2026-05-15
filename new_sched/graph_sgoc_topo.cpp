/* Topological sorts, simulated GPU footprint, ready-set VRAM order. */
#include "graph_sgoc_common.hpp"
#include "graph_sgoc_bundle_dag.hpp"
#include "graph_sgoc_bundle_topo.hpp"

namespace graph_sgoc_bundle {

static double graph_sgoc_elapsed_sec(std::chrono::steady_clock::time_point a,
                                                    std::chrono::steady_clock::time_point b)
{
    return std::chrono::duration<double>(b - a).count();
}

void graph_sgoc_compute_topological_order(const std::vector<GraphOp> &ops,
                                                  std::vector<size_t> &order_out)
{
    const size_t n = ops.size();
    order_out.clear();
    if (n == 0)
        return;

    std::vector<std::vector<size_t>> succ(n);
    std::vector<unsigned> indegree(n, 0);

    auto add_edge = [&](size_t u, size_t v) {
        if (u >= n || v >= n || u == v)
            return;
        for (size_t x : succ[u]) {
            if (x == v)
                return;
        }
        succ[u].push_back(v);
        indegree[v]++;
    };

    for (size_t u = 0; u < n; ++u) {
        for (size_t v : ops[u].successors)
            add_edge(u, v);
    }
    std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready;
    for (size_t i = 0; i < n; ++i) {
        if (indegree[i] == 0)
            ready.push(i);
    }

    order_out.reserve(n);
    while (!ready.empty()) {
        const size_t u = ready.top();
        ready.pop();
        order_out.push_back(u);
        for (size_t v : succ[u]) {
            indegree[v]--;
            if (indegree[v] == 0)
                ready.push(v);
        }
    }

    if (order_out.size() != n) {
        if (graph_sgoc_verbose_env() >= 2)
            std::cerr << "sgoc: topological sort failed (cycle?), replaying capture order" << std::endl;
        order_out.resize(0);
        for (size_t i = 0; i < n; ++i)
            order_out.push_back(i);
    }
}

size_t graph_sgoc_find_chain_head_idx(const std::vector<GraphHandleAccess> &ha, starpu_data_handle_t h)
{
    if (!h)
        return GRAPH_ACCESS_NONE;
    for (size_t i = 0; i < ha.size(); ++i) {
        if (ha[i].handle == h && ha[i].prev_for_handle == GRAPH_ACCESS_NONE)
            return i;
    }
    return GRAPH_ACCESS_NONE;
}

const GraphHandleAccess *graph_sgoc_first_task_access_on_chain(const std::vector<GraphHandleAccess> &ha,
                                                                       starpu_data_handle_t h)
{
    size_t idx = graph_sgoc_find_chain_head_idx(ha, h);
    while (idx != GRAPH_ACCESS_NONE) {
        if (idx >= ha.size())
            return nullptr;
        const GraphHandleAccess &a = ha[idx];
        if (a.handle != h)
            return nullptr;
        if (a.task != nullptr)
            return &a;
        idx = a.next_for_handle;
    }
    return nullptr;
}

void graph_sgoc_collect_unique_handles(const std::vector<GraphHandleAccess> &ha,
                                               std::vector<starpu_data_handle_t> &handles_out)
{
    handles_out.clear();
    std::unordered_set<void *> seen;
    for (const GraphHandleAccess &a : ha) {
        if (!a.handle)
            continue;
        void *p = static_cast<void *>(a.handle);
        if (seen.insert(p).second)
            handles_out.push_back(a.handle);
    }
}

std::string graph_op_memory_trace_name(const GraphOp &op)
{
    if (op.kind == GraphOp::INVALIDATE) {
        if (!op.handle)
            return "invalidate_submit";
        char addr[32];
        std::snprintf(addr, sizeof addr, "%p", static_cast<void *>(op.handle));
        return std::string("invalidate_submit handle=") + addr;
    }
    if (op.kind == GraphOp::TASK && op.task) {
        if (op.task->cl && op.task->cl->name && op.task->cl->name[0])
            return std::string(op.task->cl->name);
        if (op.task->name && op.task->name[0])
            return std::string(op.task->name);
    }
    return "task";
}

/** True if the first task access on this handle's chain reads existing data (STARPU_R or STARPU_RW). */
bool graph_sgoc_handle_live_before_graph(const std::vector<GraphHandleAccess> &ha, starpu_data_handle_t h)
{
    const GraphHandleAccess *fa = graph_sgoc_first_task_access_on_chain(ha, h);
    if (!fa)
        return false;
    return (fa->mode & STARPU_R) != 0;
}

/**
 * Schedule-independent memory hint for greedy topo: pure STARPU_W adds each distinct buffer's size; read/scratch/RW
 * without pure W adds 0; invalidate_submit contributes -size(handle). Checkpoint clones match their source codelet.
 */
/** Pinned-node footprint model: byte change if \p op runs when \p resident holds live pure-write handles. */
std::int64_t graph_sgoc_op_memory_delta_for_resident(const GraphOp &op,
                                                              const std::unordered_set<void *> &resident)
{
    if (op.kind == GraphOp::INVALIDATE) {
        starpu_data_handle_t h = op.handle;
        if (!h)
            return 0;
        void *p = static_cast<void *>(h);
        if (resident.find(p) == resident.end())
            return 0;
        return -static_cast<std::int64_t>(starpu_data_get_size(h));
    }
    if (op.kind == GraphOp::TASK) {
        std::int64_t d = 0;
        std::unordered_set<void *> new_pure_writes;
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (!ref.handle || !graph_access_mode_is_pure_write(ref.mode))
                continue;
            void *p = static_cast<void *>(ref.handle);
            if (!new_pure_writes.insert(p).second)
                continue;
            if (resident.find(p) == resident.end())
                d += static_cast<std::int64_t>(starpu_data_get_size(ref.handle));
        }
        return d;
    }
    return 0;
}

void graph_sgoc_op_apply_memory_effect_to_resident(const GraphOp &op, std::unordered_set<void *> &resident)
{
    if (op.kind == GraphOp::INVALIDATE) {
        starpu_data_handle_t h = op.handle;
        if (h)
            resident.erase(static_cast<void *>(h));
        return;
    }
    if (op.kind != GraphOp::TASK)
        return;
    std::unordered_set<void *> new_pure_writes;
    for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
        if (!ref.handle || !graph_access_mode_is_pure_write(ref.mode))
            continue;
        void *p = static_cast<void *>(ref.handle);
        if (new_pure_writes.insert(p).second)
            resident.insert(p);
    }
}

/** Pinned-node footprint simulation: peak/initial bytes and optional per-op trace (verbose 6). */
void graph_sgoc_compute_memory_after_ops(const std::vector<GraphOp> &ops, const std::vector<GraphHandleAccess> &ha,
                                                 const std::vector<size_t> &topo_order, size_t *peak_topo_index_out,
                                                 std::int64_t *peak_bytes_out, std::int64_t *initial_bytes_out,
                                                 size_t *initial_live_handle_count_out, bool print_memory_trace)
{
    std::vector<starpu_data_handle_t> unique_handles;
    graph_sgoc_collect_unique_handles(ha, unique_handles);

    std::unordered_set<void *> resident;
    std::int64_t current = 0;
    size_t initial_live_handles = 0;

    for (starpu_data_handle_t h : unique_handles) {
        if (!h || !graph_sgoc_handle_live_before_graph(ha, h))
            continue;
        void *p = static_cast<void *>(h);
        if (!resident.insert(p).second)
            continue;
        initial_live_handles++;
        current += static_cast<std::int64_t>(starpu_data_get_size(h));
    }

    if (initial_bytes_out)
        *initial_bytes_out = current;
    if (initial_live_handle_count_out)
        *initial_live_handle_count_out = initial_live_handles;

    if (print_memory_trace) {
        std::cerr << "sgoc: memory trace: graph_ops=" << ops.size() << " topo_order_len=" << topo_order.size()
                  << " before_replay bytes=" << current << std::endl;
    }

    std::int64_t peak = std::numeric_limits<std::int64_t>::min();
    size_t peak_topo_i = 0;

    for (size_t ti = 0; ti < topo_order.size(); ++ti) {
        const size_t opi = topo_order[ti];
        if (opi >= ops.size())
            continue;
        const GraphOp &op = ops[opi];
        const std::int64_t d = graph_sgoc_op_memory_delta_for_resident(op, resident);
        current += d;
        graph_sgoc_op_apply_memory_effect_to_resident(op, resident);

        if (print_memory_trace) {
            const std::ios::fmtflags trace_flags = std::cerr.flags();
            std::cerr << "sgoc: memory trace: topo_idx=" << ti << " graph_op_idx=" << opi
                      << " op=" << graph_op_memory_trace_name(op) << " pred_us=" << std::scientific
                      << std::setprecision(2) << op.predicted_exec_time;
            std::cerr.flags(trace_flags);
            std::cerr << " bytes_after=" << current << std::endl;
        }

        if (peak == std::numeric_limits<std::int64_t>::min() || current >= peak) {
            peak = current;
            peak_topo_i = ti;
        }
    }

    if (peak_bytes_out)
        *peak_bytes_out = peak == std::numeric_limits<std::int64_t>::min() ? current : peak;
    if (peak_topo_index_out)
        *peak_topo_index_out = peak_topo_i;
}

/**
 * Topological order over the same DAG as graph_sgoc_compute_topological_order, using predecessors / successors.
 * Greedy: among ready ops, pick minimal graph_sgoc_op_intrinsic_memory_delta (precomputed once per op).
 *
 * If non-null, \p greedy_attempt_sec_out is the wall time for the greedy attempt (prep + main loop), whether or not
 * it succeeds. \p lex_fallback_sec_out is the time spent in lexicographic Kahn topo when greedy fails (else 0).
 * \p greedy_prep_sec_out / \p greedy_loop_sec_out split the greedy attempt (see verbose level 4 timing lines).
 */
void graph_sgoc_compute_greedy_memory_topological_order(const std::vector<GraphOp> &ops,
                                                                std::vector<size_t> &order_out,
                                                                double *greedy_attempt_sec_out,
                                                                double *lex_fallback_sec_out,
                                                                double *greedy_prep_sec_out,
                                                                double *greedy_loop_sec_out,
                                                                const std::vector<unsigned> *tie_break = nullptr)
{
    using clock = std::chrono::steady_clock;
    const size_t n = ops.size();
    order_out.clear();
    if (n == 0) {
        if (greedy_attempt_sec_out)
            *greedy_attempt_sec_out = 0;
        if (lex_fallback_sec_out)
            *lex_fallback_sec_out = 0;
        if (greedy_prep_sec_out)
            *greedy_prep_sec_out = 0;
        if (greedy_loop_sec_out)
            *greedy_loop_sec_out = 0;
        return;
    }

    const bool use_tie = tie_break && tie_break->size() == n;

    const clock::time_point t_greedy_start = clock::now();

    std::vector<std::int64_t> intrinsic_delta(n);
    for (size_t i = 0; i < n; ++i)
        intrinsic_delta[i] = graph_sgoc_op_intrinsic_memory_delta(ops[i]);

    std::vector<unsigned> indegree(n, 0);
    for (size_t i = 0; i < n; ++i)
        indegree[i] = static_cast<unsigned>(ops[i].predecessors.size());

    std::vector<size_t> ready;
    ready.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (indegree[i] == 0)
            ready.push_back(i);
    }

    const clock::time_point t_after_prep = clock::now();

    order_out.reserve(n);
    while (!ready.empty()) {
        size_t best = ready[0];
        std::int64_t best_delta = intrinsic_delta[best];
        for (size_t k = 1; k < ready.size(); ++k) {
            const size_t u = ready[k];
            const std::int64_t d = intrinsic_delta[u];
            bool better = d < best_delta;
            if (!better && d == best_delta) {
                if (use_tie) {
                    if ((*tie_break)[u] < (*tie_break)[best])
                        better = true;
                } else if (u < best) {
                    better = true;
                }
            }
            if (better) {
                best_delta = d;
                best = u;
            }
        }

        for (size_t k = 0; k < ready.size(); ++k) {
            if (ready[k] == best) {
                ready[k] = ready.back();
                ready.pop_back();
                break;
            }
        }

        order_out.push_back(best);

        for (size_t v : ops[best].successors) {
            if (v >= n)
                continue;
            indegree[v]--;
            if (indegree[v] == 0)
                ready.push_back(v);
        }
    }

    const clock::time_point t_after_loop = clock::now();

    if (greedy_prep_sec_out)
        *greedy_prep_sec_out = graph_sgoc_elapsed_sec(t_greedy_start, t_after_prep);
    if (greedy_loop_sec_out)
        *greedy_loop_sec_out = graph_sgoc_elapsed_sec(t_after_prep, t_after_loop);
    if (greedy_attempt_sec_out)
        *greedy_attempt_sec_out = graph_sgoc_elapsed_sec(t_greedy_start, t_after_loop);

    if (order_out.size() != n) {
        if (graph_sgoc_verbose_env() >= 2)
            std::cerr << "sgoc: greedy memory topo failed (cycle?), falling back to lexicographic topo"
                      << std::endl;
        const clock::time_point t_lex_start = clock::now();
        graph_sgoc_compute_topological_order(ops, order_out);
        const clock::time_point t_lex_end = clock::now();
        if (lex_fallback_sec_out)
            *lex_fallback_sec_out = graph_sgoc_elapsed_sec(t_lex_start, t_lex_end);
    } else if (lex_fallback_sec_out)
        *lex_fallback_sec_out = 0;
}
/** Default on: set STARPU_GRAPH_SCHED_SGOC_READYVRAM_TOPO=0 to use legacy lex + greedy-memory topo instead. */
bool graph_sgoc_ready_vram_topo_enabled(void)
{
    const char *e = getenv("STARPU_GRAPH_SCHED_SGOC_READYVRAM_TOPO");
    if (!e || !e[0])
        return true;
    return std::atoi(e) != 0;
}

/**
 * Ready-set scheduler: among ops with satisfied predecessors, pick next op minimizing
 * graph_sgoc_op_memory_delta_for_resident under the current pure-write footprint model; ties prefer INVALIDATE
 * before TASK (free GPU bytes earlier), then lower op index.
 */
void graph_sgoc_compute_ready_set_greedy_vram_topological_order(const std::vector<GraphOp> &ops,
                                                                       const std::vector<GraphHandleAccess> &handle_accesses,
                                                                       const std::unordered_set<void *> *starpu_gpu_resident_truth,
                                                                       std::vector<size_t> &order_out,
                                                                       int verbose)
{
    const size_t n = ops.size();
    order_out.clear();
    if (n == 0)
        return;

    std::unordered_set<void *> resident;
    std::vector<starpu_data_handle_t> unique_handles;
    graph_sgoc_collect_unique_handles(handle_accesses, unique_handles);
    for (starpu_data_handle_t h : unique_handles) {
        if (!h || !graph_sgoc_handle_live_before_graph(handle_accesses, h))
            continue;
        void *p = static_cast<void *>(h);
        if (starpu_gpu_resident_truth) {
            if (starpu_gpu_resident_truth->count(p))
                (void)resident.insert(p);
        } else {
            (void)resident.insert(p);
        }
    }

    std::vector<unsigned> indegree(n, 0u);
    for (size_t i = 0; i < n; ++i)
        indegree[i] = static_cast<unsigned>(ops[i].predecessors.size());

    std::vector<size_t> ready;
    ready.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (indegree[i] == 0u)
            ready.push_back(i);
    }

    order_out.reserve(n);
    while (!ready.empty()) {
        size_t best = ready[0];
        std::int64_t best_d = graph_sgoc_op_memory_delta_for_resident(ops[best], resident);
        bool best_inv = (ops[best].kind == GraphOp::INVALIDATE);
        for (size_t k = 1; k < ready.size(); ++k) {
            const size_t u = ready[k];
            const std::int64_t du = graph_sgoc_op_memory_delta_for_resident(ops[u], resident);
            const bool u_inv = (ops[u].kind == GraphOp::INVALIDATE);
            bool better = du < best_d;
            if (!better && du == best_d) {
                if (u_inv && !best_inv)
                    better = true;
                else if (u_inv == best_inv && u < best)
                    better = true;
            }
            if (better) {
                best = u;
                best_d = du;
                best_inv = u_inv;
            }
        }

        for (size_t k = 0; k < ready.size(); ++k) {
            if (ready[k] == best) {
                ready[k] = ready.back();
                ready.pop_back();
                break;
            }
        }

        order_out.push_back(best);
        graph_sgoc_op_apply_memory_effect_to_resident(ops[best], resident);

        for (size_t v : ops[best].successors) {
            if (v >= n)
                continue;
            indegree[v]--;
            if (indegree[v] == 0u)
                ready.push_back(v);
        }
    }

    if (order_out.size() != n) {
        if (verbose >= 2)
            std::cerr << "sgoc: ready_vram_topo failed (cycle?), falling back to lex topo\n";
        graph_sgoc_compute_topological_order(ops, order_out);
    }
}

} /* namespace graph_sgoc_bundle */
