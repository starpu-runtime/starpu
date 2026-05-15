/* DAG: StarPU access modes, edges, handle chains, capture registration, predicted exec time. */
#include "graph_sgoc_common.hpp"
#include "graph_sgoc_bundle_dag.hpp"

namespace graph_sgoc_bundle {

void graph_sgoc_bump_handle_access_op_indices_after_insert(std::vector<GraphHandleAccess> &handle_accesses,
                                                                   size_t insert_pos)
{
    for (GraphHandleAccess &access : handle_accesses) {
        if (access.op_idx >= insert_pos)
            access.op_idx++;
    }
}

/** Bump stored op indices in predecessor/successor lists after inserting one op at \p insert_pos. */
void graph_sgoc_bump_op_graph_indices_after_insert(std::vector<GraphOp> &ops, size_t insert_pos)
{
    for (GraphOp &op : ops) {
        for (size_t &idx : op.predecessors) {
            if (idx >= insert_pos)
                idx++;
        }
        for (size_t &idx : op.successors) {
            if (idx >= insert_pos)
                idx++;
        }
    }
}

void graph_sgoc_bump_indices_after_insert(graph_sgoc_data *data, size_t insert_pos)
{
    graph_sgoc_bump_handle_access_op_indices_after_insert(data->graph_handle_accesses, insert_pos);
}

bool graph_sgoc_task_runnable_on_pinned_worker(const struct starpu_task *task, unsigned workerid)
{
    if (!task->cl)
        return true;
    unsigned nimpl = 0;
    return starpu_worker_can_execute_task_first_impl(workerid, const_cast<struct starpu_task *>(task), &nimpl) != 0;
}

/**
 * Pin replay to \p pin_worker only if that worker can execute the task; otherwise clear worker binding.
 * When \p cl_runnable_cache is non-null and the codelet has no per-task can_execute hook, results are cached by
 * codelet pointer so replay avoids one StarPU query per task for repeated codelets.
 */
void graph_sgoc_apply_replay_worker_pin(struct starpu_task *task, int pin_worker, int sched_verbose,
                                                std::unordered_map<const struct starpu_codelet *, bool> *cl_runnable_cache)
{
    if (pin_worker < 0 || !task)
        return;
    const unsigned pw = static_cast<unsigned>(pin_worker);
    if (!task->cl) {
        task->execute_on_a_specific_worker = 1;
        task->workerid = pw;
        return;
    }

    bool runnable;
    if (cl_runnable_cache && !task->cl->can_execute) {
        const auto it = cl_runnable_cache->find(task->cl);
        if (it != cl_runnable_cache->end())
            runnable = it->second;
        else {
            runnable = graph_sgoc_task_runnable_on_pinned_worker(task, pw);
            cl_runnable_cache->emplace(task->cl, runnable);
        }
    } else
        runnable = graph_sgoc_task_runnable_on_pinned_worker(task, pw);

    if (!runnable) {
        if (sched_verbose >= 3) {
            const char *cln = task->cl->name ? task->cl->name : "?";
            std::cerr << "sgoc: replay leave unpinned: codelet \"" << cln
                      << "\" cannot run on graph pin worker_id=" << pin_worker << '\n';
        }
        task->execute_on_a_specific_worker = 0;
        return;
    }
    task->execute_on_a_specific_worker = 1;
    task->workerid = pw;
}

/* Default on. Set STARPU_GRAPH_SCHED_AUTO_INVALIDATE=0 to disable synthetic invalidate_submit. */
bool graph_sgoc_auto_invalidate_enabled(void)
{
    static const bool enabled = [] {
        const char *e = std::getenv("STARPU_GRAPH_SCHED_AUTO_INVALIDATE");
        return !e || std::atoi(e) != 0;
    }();
    return enabled;
}

unsigned graph_sgoc_checkpoint_max_env(void)
{
    static const unsigned checkpoint_max = [] {
        const char *e = std::getenv("STARPU_GRAPH_SCHED_CHECKPOINT_MAX");
        int value = e ? std::atoi(e) : 0;
        return value > 0 ? static_cast<unsigned>(value) : 0u;
    }();
    return checkpoint_max;
}

static double graph_sgoc_elapsed_sec(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
{
    return std::chrono::duration<double>(b - a).count();
}
/** STARPU_GRAPH_SCHED_LINEAR_REPLAY_GREEDY=0: linear flush submits in capture order; default uses greedy memory topo. */
bool graph_sgoc_linear_replay_greedy_enabled(void)
{
    const char *e = std::getenv("STARPU_GRAPH_SCHED_LINEAR_REPLAY_GREEDY");
    if (e && e[0] == '0' && e[1] == '\0')
        return false;
    return true;
}

/** Pure write (not STARPU_RW / RMW): overwrite may require a prior invalidate_submit. */
[[maybe_unused]] static bool graph_mode_is_write_only_overwrite(enum starpu_data_access_mode mode)
{
    if ((mode & STARPU_SCRATCH) != 0)
        return false;
    if ((mode & STARPU_W) == 0)
        return false;
    /* STARPU_RW implies a read of the current value; do not inject invalidate before it. */
    return (mode & STARPU_R) == 0;
}

bool graph_access_mode_is_invalidate(unsigned mode)
{
    return mode == GRAPH_ACCESS_INVALIDATE_RAW;
}

static constexpr unsigned GRAPH_SEMANTIC_ACCESS_MASK =
    STARPU_R | STARPU_W | STARPU_SCRATCH | STARPU_REDUX | STARPU_COMMUTE | STARPU_MPI_REDUX;

bool graph_access_mode_is_pure_read(unsigned mode)
{
    return (mode & GRAPH_SEMANTIC_ACCESS_MASK) == STARPU_R;
}

bool graph_access_mode_is_pure_write(unsigned mode)
{
    return (mode & GRAPH_SEMANTIC_ACCESS_MASK) == STARPU_W;
}

bool graph_access_mode_is_pure_scratch(unsigned mode)
{
    return (mode & GRAPH_SEMANTIC_ACCESS_MASK) == STARPU_SCRATCH;
}

/** Read–write data access (e.g. ::STARPU_RW, possibly with ::STARPU_COMMUTE); not pure R or pure W alone. */
bool graph_access_mode_is_read_write(unsigned mode)
{
    if (graph_access_mode_is_invalidate(mode))
        return false;
    const unsigned s = mode & GRAPH_SEMANTIC_ACCESS_MASK;
    if ((s & STARPU_SCRATCH) != 0)
        return false;
    return (s & STARPU_R) != 0 && (s & STARPU_W) != 0;
}
std::int64_t graph_sgoc_op_intrinsic_memory_delta(const GraphOp &op)
{
    if (op.kind == GraphOp::INVALIDATE) {
        if (!op.handle)
            return 0;
        return -static_cast<std::int64_t>(starpu_data_get_size(op.handle));
    }
    if (op.kind != GraphOp::TASK)
        return 0;
    std::int64_t d = 0;
    std::unordered_set<void *> seen;
    for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
        if (!ref.handle || !graph_access_mode_is_pure_write(ref.mode))
            continue;
        void *p = static_cast<void *>(ref.handle);
        if (!seen.insert(p).second)
            continue;
        d += static_cast<std::int64_t>(starpu_data_get_size(ref.handle));
    }
    return d;
}
size_t graph_sgoc_append_handle_access(graph_sgoc_data *data, size_t op_idx, starpu_data_handle_t handle,
                                               unsigned mode, struct starpu_task *task)
{
    GraphHandleAccess access{};
    access.handle = handle;
    access.mode = mode;
    access.task = task;
    access.op_idx = op_idx;

    GraphHandleAccessList &list = data->graph_handle_access_lists[static_cast<void *>(handle)];
    access.prev_for_handle = list.tail;

    const size_t access_idx = data->graph_handle_accesses.size();
    data->graph_handle_accesses.push_back(access);

    if (list.tail != GRAPH_ACCESS_NONE)
        data->graph_handle_accesses[list.tail].next_for_handle = access_idx;
    else
        list.head = access_idx;
    list.tail = access_idx;

    return access_idx;
}

void graph_sgoc_register_task_accesses_op(graph_sgoc_data *data, size_t op_idx, struct starpu_task *task,
                                                  GraphOp &op)
{
    if (!task->cl)
        return;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    op.handle_accesses.reserve(nbuf);
    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(task, i);
        if (!h)
            continue;
        const unsigned mode = (unsigned)STARPU_TASK_GET_MODE(task, i);
        const size_t access_idx = graph_sgoc_append_handle_access(data, op_idx, h, mode, task);
        op.handle_accesses.push_back({h, mode, access_idx});
    }
}

void graph_sgoc_register_task_accesses(graph_sgoc_data *data, size_t op_idx, struct starpu_task *task)
{
    graph_sgoc_register_task_accesses_op(data, op_idx, task, data->graph_ops[op_idx]);
}

void graph_sgoc_register_invalidate_access_op(graph_sgoc_data *data, GraphOp &op, size_t op_idx,
                                                      starpu_data_handle_t handle)
{
    if (!handle)
        return;
    const size_t access_idx =
        graph_sgoc_append_handle_access(data, op_idx, handle, GRAPH_ACCESS_INVALIDATE_RAW, nullptr);
    op.handle_accesses.push_back({handle, GRAPH_ACCESS_INVALIDATE_RAW, access_idx});
}

void graph_sgoc_register_invalidate_access(graph_sgoc_data *data, size_t op_idx, starpu_data_handle_t handle)
{
    graph_sgoc_register_invalidate_access_op(data, data->graph_ops[op_idx], op_idx, handle);
}

bool graph_access_mode_is_writer(unsigned mode)
{
    return !graph_access_mode_is_invalidate(mode) && ((mode & STARPU_W) != 0);
}

/** Task writers and explicit invalidates both end a handle "version" for dependency purposes. */
bool graph_access_is_handle_producer_for_deps(const GraphHandleAccess &a)
{
    if (graph_access_mode_is_invalidate(a.mode))
        return true;
    return graph_access_mode_is_writer(a.mode) && a.task != nullptr;
}

/** \p producer_op_idx must finish before \p consumer_op_idx (maintains predecessors + successors). */
void graph_op_add_edge(std::vector<GraphOp> &ops, size_t consumer_op_idx, size_t producer_op_idx)
{
    if (producer_op_idx == GRAPH_ACCESS_NONE || consumer_op_idx == GRAPH_ACCESS_NONE)
        return;
    if (producer_op_idx == consumer_op_idx)
        return;
    if (producer_op_idx >= ops.size() || consumer_op_idx >= ops.size())
        return;

    GraphOp &consumer = ops[consumer_op_idx];
    for (size_t existing : consumer.predecessors) {
        if (existing == producer_op_idx)
            return;
    }
    consumer.predecessors.push_back(producer_op_idx);

    GraphOp &producer = ops[producer_op_idx];
    for (size_t existing : producer.successors) {
        if (existing == consumer_op_idx)
            return;
    }
    producer.successors.push_back(consumer_op_idx);
}

void graph_sgoc_validate_invalidate_then_pure_write_windows(graph_sgoc_data *data);

/** SGOC list capture: add one pred/succ edge (indices are capture_stable_id values). O(1) map + degree scan. */
void graph_op_add_edge_stable_sgoc(graph_sgoc_data *data, GraphOp *consumer, size_t producer_stable)
{
    if (!data || !data->graph_sgoc || !consumer)
        return;
    if (producer_stable == GRAPH_ACCESS_NONE)
        return;
    graph_sgoc_data::graph_sgoc_runtime &G = *data->graph_sgoc;
    const size_t cons = consumer->capture_stable_id;
    if (producer_stable == cons)
        return;
    const auto pit = G.capture_id_to_iter.find(producer_stable);
    if (pit == G.capture_id_to_iter.end() || pit->second == G.capture_ops.end())
        return;
    GraphOp *producer = &*pit->second;

    for (size_t existing : consumer->predecessors) {
        if (existing == producer_stable)
            return;
    }
    consumer->predecessors.push_back(producer_stable);

    for (size_t existing : producer->successors) {
        if (existing == cons)
            return;
    }
    producer->successors.push_back(cons);
}

void graph_op_add_pure_read_dependencies_sgoc(graph_sgoc_data *data, GraphOp *consumer, size_t access_idx)
{
    size_t prev_idx = data->graph_handle_accesses[access_idx].prev_for_handle;
    while (prev_idx != GRAPH_ACCESS_NONE) {
        if (prev_idx >= data->graph_handle_accesses.size())
            break;
        const GraphHandleAccess &prev = data->graph_handle_accesses[prev_idx];
        if (graph_access_is_handle_producer_for_deps(prev)) {
            graph_op_add_edge_stable_sgoc(data, consumer, prev.op_idx);
            break;
        }
        prev_idx = prev.prev_for_handle;
    }
}

void graph_op_add_writer_or_invalidate_dependencies_sgoc(graph_sgoc_data *data, GraphOp *consumer,
                                                                  size_t access_idx)
{
    size_t prev_idx = data->graph_handle_accesses[access_idx].prev_for_handle;
    while (prev_idx != GRAPH_ACCESS_NONE) {
        if (prev_idx >= data->graph_handle_accesses.size())
            break;
        const GraphHandleAccess &prev = data->graph_handle_accesses[prev_idx];
        if (graph_access_mode_is_pure_read(prev.mode) && prev.task != nullptr)
            graph_op_add_edge_stable_sgoc(data, consumer, prev.op_idx);
        else if (graph_access_is_handle_producer_for_deps(prev)) {
            graph_op_add_edge_stable_sgoc(data, consumer, prev.op_idx);
            break;
        }
        prev_idx = prev.prev_for_handle;
    }
}

/**
 * Incremental SGOC capture: add dependency edges for \p op only from current handle-access chains.
 * Amortized O(bufs × chain depth) per call instead of rebuilding the full capture graph each append.
 */
void graph_sgoc_capture_add_edges_for_op(graph_sgoc_data *data, GraphOp &op)
{
    if (!data || !data->graph_sgoc)
        return;
    if (op.kind == GraphOp::INVALIDATE) {
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (ref.access_idx >= data->graph_handle_accesses.size())
                continue;
            graph_op_add_writer_or_invalidate_dependencies_sgoc(data, &op, ref.access_idx);
        }
        return;
    }
    if (op.kind != GraphOp::TASK)
        return;
    for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
        if (ref.access_idx >= data->graph_handle_accesses.size())
            continue;
        const unsigned mode = ref.mode;
        if (graph_access_mode_is_pure_read(mode)) {
            graph_op_add_pure_read_dependencies_sgoc(data, &op, ref.access_idx);
            continue;
        }
        if (graph_access_mode_is_writer(mode))
            graph_op_add_writer_or_invalidate_dependencies_sgoc(data, &op, ref.access_idx);
    }
}

/**
 * Per-handle dependency rules (see graph_sgoc_append_handle_access chains):
 *
 * - Pure STARPU_R (not STARPU_RW): depend only on the nearest previous handle producer — task
 *   STARPU_W / STARPU_RW, or an invalidate op on this handle. No prior producer matches external init.
 *
 * - STARPU_W / STARPU_RW on a task: depend on every pure STARPU_R back to the previous producer,
 *   then on that producer (task writer or invalidate), so reads of the current version finish first
 *   and writes serialize.
 *
 * - starpu_data_invalidate_submit (INVALIDATE op): same back-edges as a writer (invalidation is a
 *   version change). Pure STARPU_W after invalidate depends on that invalidate as the producer.
 */
void graph_op_add_pure_read_dependencies(graph_sgoc_data *data, size_t consumer_op_idx, size_t access_idx)
{
    size_t prev_idx = data->graph_handle_accesses[access_idx].prev_for_handle;
    while (prev_idx != GRAPH_ACCESS_NONE) {
        if (prev_idx >= data->graph_handle_accesses.size())
            break;
        const GraphHandleAccess &prev = data->graph_handle_accesses[prev_idx];
        if (graph_access_is_handle_producer_for_deps(prev)) {
            graph_op_add_edge(data->graph_ops, consumer_op_idx, prev.op_idx);
            break;
        }
        prev_idx = prev.prev_for_handle;
    }
}

void graph_op_add_writer_or_invalidate_dependencies(graph_sgoc_data *data, size_t consumer_op_idx,
                                                           size_t access_idx)
{
    size_t prev_idx = data->graph_handle_accesses[access_idx].prev_for_handle;
    while (prev_idx != GRAPH_ACCESS_NONE) {
        if (prev_idx >= data->graph_handle_accesses.size())
            break;
        const GraphHandleAccess &prev = data->graph_handle_accesses[prev_idx];
        if (graph_access_mode_is_pure_read(prev.mode) && prev.task != nullptr)
            graph_op_add_edge(data->graph_ops, consumer_op_idx, prev.op_idx);
        else if (graph_access_is_handle_producer_for_deps(prev)) {
            graph_op_add_edge(data->graph_ops, consumer_op_idx, prev.op_idx);
            break;
        }
        prev_idx = prev.prev_for_handle;
    }
}

/** StarPU: after invalidate, the next use of the handle must be pure STARPU_W until data is valid again. */
void graph_sgoc_validate_invalidate_then_pure_write_windows(graph_sgoc_data *data)
{
    for (const auto &entry : data->graph_handle_access_lists) {
        size_t idx = entry.second.head;
        bool need_pure_w_after_invalidate = false;
        bool reported_this_window = false;

        while (idx != GRAPH_ACCESS_NONE) {
            if (idx >= data->graph_handle_accesses.size())
                break;
            const GraphHandleAccess &a = data->graph_handle_accesses[idx];
            const unsigned m = a.mode;

            if (graph_access_mode_is_invalidate(m)) {
                need_pure_w_after_invalidate = true;
                reported_this_window = false;
                idx = a.next_for_handle;
                continue;
            }

            if (need_pure_w_after_invalidate) {
                if (graph_access_mode_is_pure_write(m)) {
                    need_pure_w_after_invalidate = false;
                    reported_this_window = false;
                } else if (!reported_this_window) {
                    std::cerr << "sgoc: invalid recording — handle access between invalidate and pure "
                                   "STARPU_W (StarPU requires pure write after invalidate). handle="
                                << entry.first << " op_idx=" << a.op_idx << std::endl;
                    reported_this_window = true;
                }
            }

            idx = a.next_for_handle;
        }
    }
}

void graph_sgoc_refresh_op_dependencies(graph_sgoc_data *data)
{
    for (GraphOp &op : data->graph_ops) {
        op.predecessors.clear();
        op.successors.clear();
    }

    for (size_t op_idx = 0; op_idx < data->graph_ops.size(); ++op_idx) {
        GraphOp &op = data->graph_ops[op_idx];
        if (op.kind == GraphOp::INVALIDATE) {
            for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
                if (ref.access_idx >= data->graph_handle_accesses.size())
                    continue;
                graph_op_add_writer_or_invalidate_dependencies(data, op_idx, ref.access_idx);
            }
            continue;
        }

        if (op.kind != GraphOp::TASK)
            continue;

        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (ref.access_idx >= data->graph_handle_accesses.size())
                continue;
            const unsigned mode = ref.mode;

            if (graph_access_mode_is_pure_read(mode)) {
                graph_op_add_pure_read_dependencies(data, op_idx, ref.access_idx);
                continue;
            }

            if (graph_access_mode_is_writer(mode))
                graph_op_add_writer_or_invalidate_dependencies(data, op_idx, ref.access_idx);
        }
    }

    graph_sgoc_validate_invalidate_then_pure_write_windows(data);
}

void graph_sgoc_graph_op_set_stage_from_sched_ctx(GraphOp &op, unsigned task_sched_ctx_id,
                                                           struct starpu_task *);
unsigned graph_sgoc_iteration_source_sched_ctx(unsigned task_sched_ctx_id);

/**
 * If handle H will be write-only (STARPU_W): insert a synthetic invalidate_submit before that write when needed.
 *
 * - First recorded access on H is pure STARPU_W: append INVALIDATE immediately before the new task is appended.
 * - H was already used in this recording: insert INVALIDATE after the op that held the last access on H,
 *   unless that last access is already an invalidate.
 *
 * Resulting chain on H is ... -> invalidate -> pure STARPU_W (no STARPU_R on H in between).
 */
void graph_sgoc_insert_missing_pre_write_invalidates(graph_sgoc_data *data, struct starpu_task *task)
{
    if (!graph_sgoc_auto_invalidate_enabled())
        return;
    if (!task->cl)
        return;

    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);

    if (data->graph_sgoc) {
        graph_sgoc_data::graph_sgoc_runtime &G = *data->graph_sgoc;
        for (unsigned i = 0; i < nbuf; i++) {
            starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(task, i);
            if (!h)
                continue;
            enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, i);
            if (!graph_mode_is_write_only_overwrite(mode))
                continue;

            auto it_list = data->graph_handle_access_lists.find(static_cast<void *>(h));
            const bool no_prior_recorded_access =
                (it_list == data->graph_handle_access_lists.end() || it_list->second.tail == GRAPH_ACCESS_NONE);

            if (no_prior_recorded_access) {
                GraphOp inv{};
                inv.kind = GraphOp::INVALIDATE;
                inv.task = nullptr;
                inv.handle = h;
                inv.capture_synthetic_invalidate = true;
                graph_sgoc_graph_op_set_stage_from_sched_ctx(inv, task->sched_ctx, task);
                const size_t sid = G.capture_next_stable_id++;
                inv.capture_stable_id = sid;
                G.capture_ops.push_back(std::move(inv));
                G.capture_id_to_iter[sid] = std::prev(G.capture_ops.end());
                graph_sgoc_register_invalidate_access_op(data, G.capture_ops.back(), sid, h);
                data->graph_added_invalidate_submit++;
                graph_sgoc_capture_add_edges_for_op(data, G.capture_ops.back());
                continue;
            }

            const GraphHandleAccess &last_access = data->graph_handle_accesses[it_list->second.tail];
            if (graph_access_mode_is_invalidate(last_access.mode))
                continue;

            const size_t after_sid = last_access.op_idx;
            const auto it_m = G.capture_id_to_iter.find(after_sid);
            if (it_m == G.capture_id_to_iter.end() || it_m->second == G.capture_ops.end())
                continue;

            GraphOp inv{};
            inv.kind = GraphOp::INVALIDATE;
            inv.task = nullptr;
            inv.handle = h;
            inv.capture_synthetic_invalidate = true;
            graph_sgoc_graph_op_set_stage_from_sched_ctx(inv, task->sched_ctx, task);
            const size_t sid = G.capture_next_stable_id++;
            inv.capture_stable_id = sid;
            auto it_new = G.capture_ops.insert(std::next(it_m->second), std::move(inv));
            G.capture_id_to_iter[sid] = it_new;
            graph_sgoc_register_invalidate_access_op(data, *it_new, sid, h);
            data->graph_added_invalidate_submit++;
            graph_sgoc_capture_add_edges_for_op(data, *it_new);
        }
        return;
    }

    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(task, i);
        if (!h)
            continue;
        enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, i);
        if (!graph_mode_is_write_only_overwrite(mode))
            continue;

        auto it_list = data->graph_handle_access_lists.find(static_cast<void *>(h));
        const bool no_prior_recorded_access =
            (it_list == data->graph_handle_access_lists.end() || it_list->second.tail == GRAPH_ACCESS_NONE);

        if (no_prior_recorded_access) {
            GraphOp inv{};
            inv.kind = GraphOp::INVALIDATE;
            inv.task = nullptr;
            inv.handle = h;
            inv.capture_synthetic_invalidate = true;
            graph_sgoc_graph_op_set_stage_from_sched_ctx(inv, task->sched_ctx, task);
            data->graph_ops.push_back(inv);
            graph_sgoc_register_invalidate_access(data, data->graph_ops.size() - 1, h);
            data->graph_added_invalidate_submit++;
            graph_sgoc_refresh_op_dependencies(data);
            continue;
        }

        const GraphHandleAccess &last_access = data->graph_handle_accesses[it_list->second.tail];
        if (graph_access_mode_is_invalidate(last_access.mode))
            continue;

        const size_t insert_pos = last_access.op_idx + 1;
        GraphOp inv{};
        inv.kind = GraphOp::INVALIDATE;
        inv.task = nullptr;
        inv.handle = h;
        inv.capture_synthetic_invalidate = true;
        graph_sgoc_graph_op_set_stage_from_sched_ctx(inv, task->sched_ctx, task);
        data->graph_ops.insert(data->graph_ops.begin() + insert_pos, inv);
        graph_sgoc_bump_indices_after_insert(data, insert_pos);
        graph_sgoc_register_invalidate_access(data, insert_pos, h);
        data->graph_added_invalidate_submit++;
        graph_sgoc_refresh_op_dependencies(data);
    }
}

/** Min expected duration (µs) on \p pin_worker over codelet implementations that worker can run. */
double graph_sgoc_predicted_exec_time_us_for_pinned_worker(struct starpu_task *task, int pin_worker,
                                                                   unsigned sched_ctx_id)
{
    if (!task || !task->cl || pin_worker < 0)
        return std::numeric_limits<double>::quiet_NaN();

    unsigned impl_mask = 0;
    if (!starpu_worker_can_execute_task_impl(static_cast<unsigned>(pin_worker), task, &impl_mask))
        return std::numeric_limits<double>::quiet_NaN();

    double best = std::numeric_limits<double>::infinity();
    for (unsigned nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; ++nimpl) {
        if (!(impl_mask & (1U << nimpl)))
            continue;
        const double len =
            starpu_task_worker_expected_length(task, static_cast<unsigned>(pin_worker), sched_ctx_id, nimpl);
        if (len < best)
            best = len;
    }
    if (!(best < std::numeric_limits<double>::infinity()))
        return std::numeric_limits<double>::quiet_NaN();
    return best;
}

/** Map StarPU expected-length outliers to +inf for ordering (user: unknown / -1 → inf). */
double graph_sgoc_effective_predicted_us(double starpu_expected_length_us)
{
    if (std::isnan(starpu_expected_length_us) || !std::isfinite(starpu_expected_length_us) ||
        starpu_expected_length_us < 0.0)
        return std::numeric_limits<double>::infinity();
    return starpu_expected_length_us;
}

/**
 * Scheduling context whose iteration stack applies to this task at capture time.
 * Graph capture runs before starpu_task_submit(), so task->sched_ctx is often still
 * STARPU_NMAX_SCHED_CTXS; submit later sets it from starpu_sched_ctx_get_context() / initial ctx.
 */
unsigned graph_sgoc_iteration_source_sched_ctx(unsigned task_sched_ctx_id)
{
    if (task_sched_ctx_id < STARPU_NMAX_SCHED_CTXS)
        return task_sched_ctx_id;
    unsigned cur = starpu_sched_ctx_get_context();
    if (cur < STARPU_NMAX_SCHED_CTXS)
        return cur;
    return 0u;
}

/**
 * Read StarPU iteration nest into \p op.
 *
 * Graph capture runs from \c starpu_task_insert before \c starpu_task_submit(); new tasks are cleared with memset,
 * so \c task->iterations are (0,0) and must not be trusted until submit copies live values from sched_ctx. Always
 * read \c starpu_sched_ctx_get_iteration for the live stack only — never prefer \c task->iterations (memset zeros look
 * like valid tags 0/0 before submit copies real values from sched_ctx).
 *
 * Stage index: nested \c iteration_push uses slot 1 when set (\p g1 >= 0); if only one \c iteration_push level is
 * active (\p g1 < 0), slot 0 holds the stage index.
 */
void graph_sgoc_graph_op_set_stage_from_sched_ctx(GraphOp &op, unsigned task_sched_ctx_id,
                                                           struct starpu_task *)
{
    op.graph_stage_batch_iteration_valid = false;
    op.graph_stage_batch_iteration = 0;
    op.graph_stage_subiteration_valid = false;
    op.graph_stage_subiteration = 0;
    const unsigned ctx = graph_sgoc_iteration_source_sched_ctx(task_sched_ctx_id);

    long g0 = starpu_sched_ctx_get_iteration(ctx, 0);
    long g1 = starpu_sched_ctx_get_iteration(ctx, 1);

    long b = g0;

    if (b >= 0 && b <= static_cast<long>(std::numeric_limits<std::uint32_t>::max())) {
        op.graph_stage_batch_iteration_valid = true;
        op.graph_stage_batch_iteration = static_cast<std::uint32_t>(b);
    }

    long v = g1;
    if (v < 0 && g0 >= 0)
        v = g0;
    if (v < 0)
        return;
    if (v > static_cast<long>(std::numeric_limits<std::uint32_t>::max()))
        return;
    op.graph_stage_subiteration_valid = true;
    op.graph_stage_subiteration = static_cast<std::uint32_t>(v);
}

} /* namespace graph_sgoc_bundle */
