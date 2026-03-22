/* Graph recording for graph_recorder policy: queues task_insert / invalidate_submit
 * while a session is open, then replays via StarPU impl symbols (see starpu_graph_recorder.h).
 * Operations live in a std::vector; flush builds a DAG from per-handle ordering, then
 * topologically sorts before submit. Built as part of libgraph_sched.cpp. */

#include "graph_sched_internal.hpp"

#include <starpu_graph_recorder.h>
#include <starpu_data.h>
#include <starpu_sched_ctx.h>
#include <starpu_scheduler.h>
#include <starpu_task.h>
#include <starpu_task_util.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <queue>
#include <vector>

/* Default off. Set STARPU_GRAPH_SCHED_AUTO_INVALIDATE=1 to insert synthetic invalidate_submit. */
static bool graph_sched_auto_invalidate_enabled(void)
{
    static const bool enabled = [] {
        const char *e = std::getenv("STARPU_GRAPH_SCHED_AUTO_INVALIDATE");
        return e && std::atoi(e) != 0;
    }();
    return enabled;
}

static graph_sched_data *graph_recorder_policy_data(unsigned sched_ctx_id)
{
    if (sched_ctx_id == 0)
        sched_ctx_id = starpu_sched_ctx_get_context();
    if (sched_ctx_id >= (unsigned)STARPU_NMAX_SCHED_CTXS)
        sched_ctx_id = 0;
    struct starpu_sched_policy *pol = starpu_sched_get_sched_policy_in_ctx(sched_ctx_id);
    if (!pol || !pol->policy_name || std::strcmp(pol->policy_name, "graph_recorder"))
        return nullptr;
    void *p = starpu_sched_ctx_get_policy_data(sched_ctx_id);
    return p ? static_cast<graph_sched_data *>(p) : nullptr;
}

namespace {

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

static bool graph_access_mode_is_invalidate(unsigned mode)
{
    return mode == GRAPH_ACCESS_INVALIDATE_RAW;
}

/** Bump stored op indices after inserting one element at \p insert_pos (indices at or after shift by 1). */
static void graph_sched_bump_indices_after_insert(graph_sched_data *data, size_t insert_pos)
{
    for (GraphHandleAccess &access : data->graph_handle_accesses) {
        if (access.op_idx >= insert_pos)
            access.op_idx++;
    }
}

static size_t graph_sched_append_handle_access(graph_sched_data *data, size_t op_idx, starpu_data_handle_t handle,
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

[[maybe_unused]] static const GraphHandleAccess *
graph_sched_find_op_access(const std::vector<GraphHandleAccess> &handle_accesses, const GraphOp &op,
                           starpu_data_handle_t handle)
{
    for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
        if (ref.handle == handle && ref.access_idx < handle_accesses.size())
            return &handle_accesses[ref.access_idx];
    }
    return nullptr;
}

static void graph_sched_register_task_accesses(graph_sched_data *data, size_t op_idx, struct starpu_task *task)
{
    if (!task->cl)
        return;
    GraphOp &op = data->graph_ops[op_idx];
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    op.handle_accesses.reserve(nbuf);
    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(task, i);
        if (!h)
            continue;
        const size_t access_idx =
            graph_sched_append_handle_access(data, op_idx, h, (unsigned)STARPU_TASK_GET_MODE(task, i), task);
        op.handle_accesses.push_back({h, access_idx});
    }
}

static void graph_sched_register_invalidate_access(graph_sched_data *data, size_t op_idx, starpu_data_handle_t handle)
{
    if (!handle)
        return;
    GraphOp &op = data->graph_ops[op_idx];
    const size_t access_idx = graph_sched_append_handle_access(data, op_idx, handle, GRAPH_ACCESS_INVALIDATE_RAW, nullptr);
    op.handle_accesses.push_back({handle, access_idx});
}

static bool graph_access_mode_is_writer(unsigned mode)
{
    return !graph_access_mode_is_invalidate(mode) && ((mode & STARPU_W) != 0);
}

static void graph_op_add_dependency(GraphOp &op, size_t dep_idx)
{
    if (dep_idx == GRAPH_ACCESS_NONE)
        return;
    for (size_t existing : op.dependencies) {
        if (existing == dep_idx)
            return;
    }
    op.dependencies.push_back(dep_idx);
}

static void graph_sched_refresh_op_dependencies(graph_sched_data *data)
{
    for (GraphOp &op : data->graph_ops)
        op.dependencies.clear();

    for (GraphOp &op : data->graph_ops) {
        if (op.kind != GraphOp::TASK)
            continue;
        for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
            if (ref.access_idx >= data->graph_handle_accesses.size())
                continue;
            size_t prev_idx = data->graph_handle_accesses[ref.access_idx].prev_for_handle;
            while (prev_idx != GRAPH_ACCESS_NONE) {
                if (prev_idx >= data->graph_handle_accesses.size())
                    break;
                const GraphHandleAccess &prev = data->graph_handle_accesses[prev_idx];
                if (graph_access_mode_is_writer(prev.mode) && prev.task != nullptr) {
                    graph_op_add_dependency(op, prev.op_idx);
                    break;
                }
                prev_idx = prev.prev_for_handle;
            }
        }
    }
}

/**
 * If handle H will be write-only (STARPU_W) and some task already used H, insert a
 * synthetic invalidate_submit after that last task only when the user has not
 * already recorded invalidate_submit for H between that task and this write.
 */
static void graph_sched_insert_missing_pre_write_invalidates(graph_sched_data *data, struct starpu_task *task)
{
    if (!::graph_sched_auto_invalidate_enabled())
        return;
    if (!task->cl)
        return;

    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);

    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(task, i);
        if (!h)
            continue;
        enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, i);
        if (!graph_mode_is_write_only_overwrite(mode))
            continue;

        auto it_list = data->graph_handle_access_lists.find(static_cast<void *>(h));
        if (it_list == data->graph_handle_access_lists.end() || it_list->second.tail == GRAPH_ACCESS_NONE)
            continue;
        const GraphHandleAccess &last_access = data->graph_handle_accesses[it_list->second.tail];
        if (graph_access_mode_is_invalidate(last_access.mode))
            continue;

        const size_t insert_pos = last_access.op_idx + 1;
        GraphOp inv{};
        inv.kind = GraphOp::INVALIDATE;
        inv.task = nullptr;
        inv.handle = h;
        data->graph_ops.insert(data->graph_ops.begin() + insert_pos, inv);
        graph_sched_bump_indices_after_insert(data, insert_pos);
        graph_sched_register_invalidate_access(data, insert_pos, h);
        data->graph_added_invalidate_submit++;
        graph_sched_refresh_op_dependencies(data);
    }
}

static void graph_sched_append_captured_task(graph_sched_data *data, struct starpu_task *task)
{
    graph_sched_insert_missing_pre_write_invalidates(data, task);

    GraphOp op{};
    op.kind = GraphOp::TASK;
    op.task = task;
    op.handle = nullptr;
    data->graph_ops.push_back(op);
    graph_sched_register_task_accesses(data, data->graph_ops.size() - 1, task);
    graph_sched_refresh_op_dependencies(data);
}

static void graph_sched_compute_topological_order(const std::vector<GraphOp> &ops,
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

    for (size_t op_idx = 0; op_idx < n; ++op_idx) {
        for (size_t dep_idx : ops[op_idx].dependencies)
            add_edge(dep_idx, op_idx);
    }
    /* Capture order must hold globally: handle edges alone allow unrelated ops to reorder, and StarPU
     * may assert (e.g. read before handle init) if a reader is submitted ahead of record order. */
    for (size_t i = 0; i + 1 < n; ++i)
        add_edge(i, i + 1);

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
        if (graph_sched_verbose_env() >= 2)
            std::cerr << "graph_recorder: topological sort failed (cycle?), replaying capture order" << std::endl;
        order_out.resize(0);
        for (size_t i = 0; i < n; ++i)
            order_out.push_back(i);
    }
}

/* Call only with policy_mutex released: replay submits tasks and may call push_task_graph. */
void graph_sched_replay_recorded_ops(std::vector<GraphOp> ops, unsigned added_invalidate_submit)
{
    if (graph_sched_verbose_env() >= 2) {
        size_t n_task = 0, n_invalidate = 0;
        for (const GraphOp &op : ops) {
            switch (op.kind) {
            case GraphOp::TASK:
                n_task++;
                break;
            case GraphOp::INVALIDATE:
                n_invalidate++;
                break;
            }
        }
        std::cerr << "graph_recorder: flush ops: tasks=" << n_task
                  << " recorded_invalidate_ops=" << n_invalidate
                  << " synthetic_invalidate_inserts=" << added_invalidate_submit
                  << " auto_invalidate_env=" << (::graph_sched_auto_invalidate_enabled() ? 1 : 0) << std::endl;
    }

    std::vector<size_t> topo_order;
    graph_sched_compute_topological_order(ops, topo_order);

    _starpu_graph_recorder_set_flushing(1);
    for (size_t op_idx : topo_order) {
        const GraphOp &op = ops[op_idx];
        switch (op.kind) {
        case GraphOp::TASK:
            _starpu_task_insert_submit_built_task(op.task);
            break;
        case GraphOp::INVALIDATE:
            _starpu_data_invalidate_submit_impl(op.handle);
            break;
        }
    }
    _starpu_graph_recorder_set_flushing(0);
}

} /* namespace */

extern "C" {

static int graph_sched_capture_task_hook(struct starpu_task *task, void *arg)
{
    auto *data = static_cast<graph_sched_data *>(arg);
    std::lock_guard<std::mutex> lock(data->policy_mutex);
    if (data->graph_record_nested == 0)
        return -1;
    graph_sched_append_captured_task(data, task);
    return 0;
}

static int graph_sched_capture_invalidate_hook(starpu_data_handle_t handle, void *arg)
{
    auto *data = static_cast<graph_sched_data *>(arg);
    std::lock_guard<std::mutex> lock(data->policy_mutex);
    if (data->graph_record_nested == 0)
        return -1;
    GraphOp op{};
    op.kind = GraphOp::INVALIDATE;
    op.task = nullptr;
    op.handle = handle;
    data->graph_ops.push_back(op);
    graph_sched_register_invalidate_access(data, data->graph_ops.size() - 1, handle);
    graph_sched_refresh_op_dependencies(data);
    return 0;
}

void graph_sched_recorder_register(graph_sched_data *data)
{
    _starpu_graph_recorder_register(
        graph_sched_capture_task_hook,
        graph_sched_capture_invalidate_hook,
        nullptr,
        data);
}

void graph_sched_recorder_deinit(graph_sched_data *data, unsigned sched_ctx_id)
{
    (void)sched_ctx_id;
    for (;;) {
        std::vector<GraphOp> replay;
        unsigned added_invalidate_submit = 0;
        {
            std::unique_lock<std::mutex> lock(data->policy_mutex);
            if (data->graph_record_nested == 0)
                break;
            data->graph_record_nested--;
            if (data->graph_record_nested == 0) {
                added_invalidate_submit = data->graph_added_invalidate_submit;
                replay = std::move(data->graph_ops);
                data->graph_handle_accesses.clear();
                data->graph_handle_access_lists.clear();
            }
        }
        graph_sched_replay_recorded_ops(std::move(replay), added_invalidate_submit);
        _starpu_graph_recording_pop();
    }
    _starpu_graph_recorder_unregister(data);
}

void starpu_graph_sched_graph_recording_begin(unsigned sched_ctx_id)
{
    graph_sched_data *data = graph_recorder_policy_data(sched_ctx_id);
    if (!data)
        return;

    _starpu_graph_recording_push();

    std::lock_guard<std::mutex> lock(data->policy_mutex);
    if (data->graph_record_nested == 0) {
        data->graph_ops.clear();
        data->graph_handle_accesses.clear();
        data->graph_handle_access_lists.clear();
        data->graph_added_invalidate_submit = 0;
    }
    data->graph_record_nested++;
}

void starpu_graph_sched_graph_recording_end(unsigned sched_ctx_id)
{
    graph_sched_data *data = graph_recorder_policy_data(sched_ctx_id);
    if (!data)
        return;

    std::vector<GraphOp> replay;
    bool outermost_end = false;
    unsigned added_invalidate_submit = 0;
    {
        std::unique_lock<std::mutex> lock(data->policy_mutex);
        if (data->graph_record_nested == 0)
            return;

        data->graph_record_nested--;
        if (data->graph_record_nested == 0) {
            added_invalidate_submit = data->graph_added_invalidate_submit;
            replay = std::move(data->graph_ops);
            data->graph_handle_accesses.clear();
            data->graph_handle_access_lists.clear();
            outermost_end = true;
        }
    }

    if (outermost_end)
        graph_sched_replay_recorded_ops(std::move(replay), added_invalidate_submit);

    _starpu_graph_recording_pop();
}

} /* extern "C" */
