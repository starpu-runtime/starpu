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

/* Default on. Set STARPU_GRAPH_SCHED_AUTO_INVALIDATE=0 to disable synthetic invalidate_submit. */
static bool graph_sched_auto_invalidate_enabled(void)
{
    static const bool enabled = [] {
        const char *e = std::getenv("STARPU_GRAPH_SCHED_AUTO_INVALIDATE");
        return !e || std::atoi(e) != 0;
    }();
    return enabled;
}

static unsigned graph_sched_checkpoint_max_env(void)
{
    static const unsigned checkpoint_max = [] {
        const char *e = std::getenv("STARPU_GRAPH_SCHED_CHECKPOINT_MAX");
        int value = e ? std::atoi(e) : 0;
        return value > 0 ? static_cast<unsigned>(value) : 0u;
    }();
    return checkpoint_max;
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

struct GraphReplayStats {
    unsigned inserted_checkpoints = 0;
    unsigned added_invalidate_submit = 0;
};

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

static constexpr unsigned GRAPH_SEMANTIC_ACCESS_MASK =
    STARPU_R | STARPU_W | STARPU_SCRATCH | STARPU_REDUX | STARPU_COMMUTE | STARPU_MPI_REDUX;

static bool graph_access_mode_is_pure_read(unsigned mode)
{
    return (mode & GRAPH_SEMANTIC_ACCESS_MASK) == STARPU_R;
}

static bool graph_access_mode_is_pure_write(unsigned mode)
{
    return (mode & GRAPH_SEMANTIC_ACCESS_MASK) == STARPU_W;
}

static bool graph_access_mode_is_pure_scratch(unsigned mode)
{
    return (mode & GRAPH_SEMANTIC_ACCESS_MASK) == STARPU_SCRATCH;
}

/** Bump stored op indices after inserting one element at \p insert_pos (indices at or after shift by 1). */
static void graph_sched_bump_handle_access_op_indices_after_insert(std::vector<GraphHandleAccess> &handle_accesses,
                                                                   size_t insert_pos)
{
    for (GraphHandleAccess &access : handle_accesses) {
        if (access.op_idx >= insert_pos)
            access.op_idx++;
    }
}

static void graph_sched_bump_indices_after_insert(graph_sched_data *data, size_t insert_pos)
{
    graph_sched_bump_handle_access_op_indices_after_insert(data->graph_handle_accesses, insert_pos);
}

static void graph_sched_bump_op_dependency_indices_after_insert(std::vector<GraphOp> &ops, size_t insert_pos)
{
    for (GraphOp &op : ops) {
        for (size_t &dep_idx : op.dependencies) {
            if (dep_idx >= insert_pos)
                dep_idx++;
        }
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
        const unsigned mode = (unsigned)STARPU_TASK_GET_MODE(task, i);
        const size_t access_idx =
            graph_sched_append_handle_access(data, op_idx, h, mode, task);
        op.handle_accesses.push_back({h, mode, access_idx});
    }
}

static void graph_sched_register_invalidate_access(graph_sched_data *data, size_t op_idx, starpu_data_handle_t handle)
{
    if (!handle)
        return;
    GraphOp &op = data->graph_ops[op_idx];
    const size_t access_idx = graph_sched_append_handle_access(data, op_idx, handle, GRAPH_ACCESS_INVALIDATE_RAW, nullptr);
    op.handle_accesses.push_back({handle, GRAPH_ACCESS_INVALIDATE_RAW, access_idx});
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

static bool graph_sched_has_cycle(const std::vector<GraphOp> &ops)
{
    const size_t n = ops.size();
    if (n == 0)
        return false;

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
    for (size_t i = 0; i + 1 < n; ++i)
        add_edge(i, i + 1);

    std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready;
    for (size_t i = 0; i < n; ++i) {
        if (indegree[i] == 0)
            ready.push(i);
    }

    size_t visited = 0;
    while (!ready.empty()) {
        const size_t u = ready.top();
        ready.pop();
        visited++;
        for (size_t v : succ[u]) {
            indegree[v]--;
            if (indegree[v] == 0)
                ready.push(v);
        }
    }

    return visited != n;
}

static bool graph_op_is_checkpoint_idempotent(const GraphOp &op)
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

static const GraphOpHandleAccessRef *graph_op_find_single_pure_write_access(const GraphOp &op)
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

static const GraphHandleAccess *graph_sched_find_next_task_access(const std::vector<GraphHandleAccess> &handle_accesses,
                                                                  size_t access_idx)
{
    size_t next_idx = access_idx;
    while (next_idx != GRAPH_ACCESS_NONE) {
        if (next_idx >= handle_accesses.size())
            return nullptr;
        const GraphHandleAccess &access = handle_accesses[next_idx];
        if (access.task != nullptr)
            return &access;
        next_idx = access.next_for_handle;
    }
    return nullptr;
}

static size_t graph_sched_find_prev_task_access_idx(const std::vector<GraphHandleAccess> &handle_accesses, size_t access_idx)
{
    size_t prev_idx = access_idx;
    while (prev_idx != GRAPH_ACCESS_NONE) {
        if (prev_idx >= handle_accesses.size())
            return GRAPH_ACCESS_NONE;
        if (handle_accesses[prev_idx].task != nullptr)
            return prev_idx;
        prev_idx = handle_accesses[prev_idx].prev_for_handle;
    }
    return GRAPH_ACCESS_NONE;
}

static size_t graph_sched_find_next_task_access_idx(const std::vector<GraphHandleAccess> &handle_accesses, size_t access_idx)
{
    size_t next_idx = access_idx;
    while (next_idx != GRAPH_ACCESS_NONE) {
        if (next_idx >= handle_accesses.size())
            return GRAPH_ACCESS_NONE;
        if (handle_accesses[next_idx].task != nullptr)
            return next_idx;
        next_idx = handle_accesses[next_idx].next_for_handle;
    }
    return GRAPH_ACCESS_NONE;
}

static bool graph_op_is_checkpoint_wrr(const GraphOp &op, const std::vector<GraphHandleAccess> &handle_accesses)
{
    const GraphOpHandleAccessRef *write_ref = graph_op_find_single_pure_write_access(op);
    if (!write_ref || write_ref->access_idx >= handle_accesses.size())
        return false;

    size_t next_idx = handle_accesses[write_ref->access_idx].next_for_handle;
    for (unsigned read_count = 0; read_count < 2; ++read_count) {
        const GraphHandleAccess *next_access = graph_sched_find_next_task_access(handle_accesses, next_idx);
        if (!next_access)
            return false;
        if (!graph_access_mode_is_pure_read(next_access->mode))
            return false;
        next_idx = next_access->next_for_handle;
    }

    return true;
}

static void graph_sched_destroy_checkpoint_task(struct starpu_task *task);
static bool graph_op_is_checkpointable_with_wrr(const std::vector<GraphOp> &ops,
                                                const std::vector<GraphHandleAccess> &handle_accesses, size_t op_idx);

static struct starpu_task *graph_sched_clone_task_for_checkpoint(const struct starpu_task *task)
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
            graph_sched_destroy_checkpoint_task(clone);
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
            graph_sched_destroy_checkpoint_task(clone);
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
            graph_sched_destroy_checkpoint_task(clone);
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

static void graph_sched_destroy_checkpoint_task(struct starpu_task *task)
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

static size_t graph_sched_insert_handle_access_after(std::vector<GraphHandleAccess> &handle_accesses, size_t prev_idx,
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

static bool graph_sched_insert_checkpoint_for_wrr_task(std::vector<GraphOp> &ops,
                                                       std::vector<GraphHandleAccess> &handle_accesses, size_t op_idx,
                                                       struct starpu_task *checkpoint_task)
{
    if (op_idx >= ops.size())
        return false;

    const GraphOp &op = ops[op_idx];
    if (!op.task)
        return false;

    const GraphOpHandleAccessRef *write_ref = graph_op_find_single_pure_write_access(op);
    if (!write_ref || write_ref->access_idx >= handle_accesses.size())
        return false;

    const size_t r1_access_idx =
        graph_sched_find_next_task_access_idx(handle_accesses, handle_accesses[write_ref->access_idx].next_for_handle);
    const GraphHandleAccess *r1_access = r1_access_idx != GRAPH_ACCESS_NONE ? &handle_accesses[r1_access_idx] : nullptr;
    if (!r1_access || !graph_access_mode_is_pure_read(r1_access->mode))
        return false;

    const size_t r2_access_idx = graph_sched_find_next_task_access_idx(handle_accesses, r1_access->next_for_handle);
    const GraphHandleAccess *r2_access = r2_access_idx != GRAPH_ACCESS_NONE ? &handle_accesses[r2_access_idx] : nullptr;
    if (!r2_access || !graph_access_mode_is_pure_read(r2_access->mode))
        return false;

    const GraphOpHandleAccessRef write_ref_copy = *write_ref;
    const size_t r1_op_idx = r1_access->op_idx;
    const size_t r2_op_idx = r2_access->op_idx;
    std::vector<GraphOpHandleAccessRef> read_refs;
    read_refs.reserve(op.handle_accesses.size());
    for (const GraphOpHandleAccessRef &ref : op.handle_accesses) {
        if (graph_access_mode_is_pure_read(ref.mode))
            read_refs.push_back(ref);
    }

    const size_t insert_pos = r2_op_idx;
    GraphOp checkpoint_op{};
    checkpoint_op.kind = GraphOp::TASK;
    checkpoint_op.task = checkpoint_task;

    ops.insert(ops.begin() + insert_pos, checkpoint_op);
    graph_sched_bump_op_dependency_indices_after_insert(ops, insert_pos);
    graph_sched_bump_handle_access_op_indices_after_insert(handle_accesses, insert_pos);

    GraphOp &w2 = ops[insert_pos];
    graph_op_add_dependency(w2, op_idx);
    graph_op_add_dependency(w2, r1_op_idx);
    graph_op_add_dependency(ops[r2_op_idx + 1], insert_pos);

    const size_t primary_access_idx =
        graph_sched_insert_handle_access_after(handle_accesses, r1_access_idx, insert_pos, write_ref_copy.handle,
                                               write_ref_copy.mode, w2.task);
    w2.handle_accesses.push_back({write_ref_copy.handle, write_ref_copy.mode, primary_access_idx});

    for (const GraphOpHandleAccessRef &ref : read_refs) {
        if (ref.access_idx >= handle_accesses.size())
            continue;

        const size_t inserted_idx =
            graph_sched_insert_handle_access_after(handle_accesses, ref.access_idx, insert_pos, ref.handle, ref.mode,
                                                   w2.task);
        w2.handle_accesses.push_back({ref.handle, ref.mode, inserted_idx});

        const GraphHandleAccess *next_task =
            graph_sched_find_next_task_access(handle_accesses, handle_accesses[inserted_idx].next_for_handle);
        if (next_task)
            graph_op_add_dependency(ops[next_task->op_idx], insert_pos);
    }

    return true;
}

static void graph_sched_append_unique_op_idx(std::vector<size_t> &op_indices, size_t op_idx)
{
    if (op_idx == GRAPH_ACCESS_NONE)
        return;
    for (size_t existing : op_indices) {
        if (existing == op_idx)
            return;
    }
    op_indices.push_back(op_idx);
}

static void graph_sched_update_checkpoint_state_for_op(std::vector<GraphOp> &ops,
                                                       const std::vector<GraphHandleAccess> &handle_accesses,
                                                       size_t op_idx)
{
    if (op_idx >= ops.size())
        return;

    GraphOp &op = ops[op_idx];
    op.checkpoint_idempotent = graph_op_is_checkpoint_idempotent(op);
    op.checkpoint_wrr = op.checkpoint_idempotent && graph_op_is_checkpoint_wrr(op, handle_accesses);
    op.checkpointable = op.checkpoint_wrr && graph_op_is_checkpointable_with_wrr(ops, handle_accesses, op_idx);
}

static bool graph_op_is_checkpointable_with_wrr(const std::vector<GraphOp> &ops,
                                                const std::vector<GraphHandleAccess> &handle_accesses, size_t op_idx)
{
    if (op_idx >= ops.size())
        return false;

    const GraphOp &op = ops[op_idx];
    if (!op.checkpoint_wrr || !op.task)
        return false;

    std::vector<GraphOp> trial_ops = ops;
    std::vector<GraphHandleAccess> trial_handle_accesses = handle_accesses;
    struct starpu_task *checkpoint_task = graph_sched_clone_task_for_checkpoint(op.task);
    if (!checkpoint_task)
        return false;
    if (!graph_sched_insert_checkpoint_for_wrr_task(trial_ops, trial_handle_accesses, op_idx, checkpoint_task)) {
        graph_sched_destroy_checkpoint_task(checkpoint_task);
        return false;
    }
    const bool has_cycle = graph_sched_has_cycle(trial_ops);
    graph_sched_destroy_checkpoint_task(checkpoint_task);
    return !has_cycle;
}

static void graph_sched_run_checkpointing_pass(std::vector<GraphOp> &ops,
                                               const std::vector<GraphHandleAccess> &handle_accesses,
                                               std::vector<size_t> &idempotent_task_ops_out,
                                               std::vector<size_t> &wrr_task_ops_out,
                                               std::vector<size_t> &checkpointable_task_ops_out)
{
    idempotent_task_ops_out.clear();
    wrr_task_ops_out.clear();
    checkpointable_task_ops_out.clear();
    idempotent_task_ops_out.reserve(ops.size());
    wrr_task_ops_out.reserve(ops.size());
    checkpointable_task_ops_out.reserve(ops.size());

    for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
        graph_sched_update_checkpoint_state_for_op(ops, handle_accesses, op_idx);
        GraphOp &op = ops[op_idx];
        if (op.checkpoint_idempotent) {
            idempotent_task_ops_out.push_back(op_idx);
            if (op.checkpoint_wrr) {
                wrr_task_ops_out.push_back(op_idx);
                if (op.checkpointable)
                    checkpointable_task_ops_out.push_back(op_idx);
            }
        }
    }
}

static void graph_sched_collect_checkpoint_counts(const std::vector<GraphOp> &ops, size_t &idempotent_tasks_out,
                                                  size_t &wrr_tasks_out, size_t &checkpointable_tasks_out)
{
    idempotent_tasks_out = 0;
    wrr_tasks_out = 0;
    checkpointable_tasks_out = 0;

    for (const GraphOp &op : ops) {
        if (!op.checkpoint_idempotent)
            continue;
        idempotent_tasks_out++;
        if (!op.checkpoint_wrr)
            continue;
        wrr_tasks_out++;
        if (op.checkpointable)
            checkpointable_tasks_out++;
    }
}

static size_t graph_sched_find_first_checkpointable_op(const std::vector<GraphOp> &ops)
{
    for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
        if (ops[op_idx].checkpointable)
            return op_idx;
    }
    return GRAPH_ACCESS_NONE;
}

static unsigned graph_sched_insert_checkpoints(std::vector<GraphOp> &ops, std::vector<GraphHandleAccess> &handle_accesses)
{
    const unsigned checkpoint_max = graph_sched_checkpoint_max_env();
    unsigned inserted = 0;

    std::vector<size_t> checkpoint_idempotent_tasks;
    std::vector<size_t> checkpoint_wrr_tasks;
    std::vector<size_t> checkpointable_tasks;
    graph_sched_run_checkpointing_pass(ops, handle_accesses, checkpoint_idempotent_tasks, checkpoint_wrr_tasks,
                                       checkpointable_tasks);

    while (inserted < checkpoint_max) {
        const size_t op_idx = graph_sched_find_first_checkpointable_op(ops);
        if (op_idx == GRAPH_ACCESS_NONE)
            break;

        struct starpu_task *checkpointed_task = ops[op_idx].task;
        std::vector<size_t> affected_op_indices;
        graph_sched_append_unique_op_idx(affected_op_indices, op_idx);
        for (const GraphOpHandleAccessRef &ref : ops[op_idx].handle_accesses) {
            if (!graph_access_mode_is_pure_read(ref.mode) || ref.access_idx >= handle_accesses.size())
                continue;
            const size_t prev_task_idx =
                graph_sched_find_prev_task_access_idx(handle_accesses, handle_accesses[ref.access_idx].prev_for_handle);
            if (prev_task_idx != GRAPH_ACCESS_NONE)
                graph_sched_append_unique_op_idx(affected_op_indices, handle_accesses[prev_task_idx].op_idx);
        }

        struct starpu_task *checkpoint_task = graph_sched_clone_task_for_checkpoint(ops[op_idx].task);
        if (!checkpoint_task) {
            if (graph_sched_verbose_env() >= 2)
                std::cerr << "graph_recorder: failed to allocate checkpoint task clone" << std::endl;
            break;
        }
        if (!graph_sched_insert_checkpoint_for_wrr_task(ops, handle_accesses, op_idx, checkpoint_task)) {
            graph_sched_destroy_checkpoint_task(checkpoint_task);
            if (graph_sched_verbose_env() >= 2)
                std::cerr << "graph_recorder: failed to insert checkpoint task" << std::endl;
            break;
        }

        graph_sched_append_unique_op_idx(affected_op_indices, op_idx);
        for (size_t candidate_op_idx = 0; candidate_op_idx < ops.size(); ++candidate_op_idx) {
            if (ops[candidate_op_idx].task == checkpoint_task)
                graph_sched_append_unique_op_idx(affected_op_indices, candidate_op_idx);
        }
        for (const GraphHandleAccess &access : handle_accesses) {
            if (access.task != checkpoint_task)
                continue;
            const GraphHandleAccess *next_task =
                graph_sched_find_next_task_access(handle_accesses, access.next_for_handle);
            if (next_task)
                graph_sched_append_unique_op_idx(affected_op_indices, next_task->op_idx);
        }
        for (size_t affected_op_idx : affected_op_indices)
            graph_sched_update_checkpoint_state_for_op(ops, handle_accesses, affected_op_idx);

        if (graph_sched_verbose_env() >= 5) {
            const char *cl_name = (checkpointed_task && checkpointed_task->cl && checkpointed_task->cl->name)
                                      ? checkpointed_task->cl->name
                                      : "unknown";
            const unsigned long task_id = checkpointed_task ? starpu_task_get_job_id(checkpointed_task) : 0;
            std::cerr << "graph_recorder: checkpointed task: cl=" << cl_name << " task_id=" << task_id << std::endl;
        }
        inserted++;
    }

    return inserted;
}

/* Call only with policy_mutex released: replay submits tasks and may call push_task_graph. */
GraphReplayStats graph_sched_replay_recorded_ops(std::vector<GraphOp> ops,
                                                 std::vector<GraphHandleAccess> handle_accesses,
                                                 unsigned added_invalidate_submit)
{
    GraphReplayStats stats{};
    stats.added_invalidate_submit = added_invalidate_submit;
    const unsigned inserted_checkpoints = graph_sched_insert_checkpoints(ops, handle_accesses);
    stats.inserted_checkpoints = inserted_checkpoints;

    size_t checkpoint_idempotent_task_count = 0;
    size_t checkpoint_wrr_task_count = 0;
    size_t checkpointable_task_count = 0;
    graph_sched_collect_checkpoint_counts(ops, checkpoint_idempotent_task_count, checkpoint_wrr_task_count,
                                          checkpointable_task_count);

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
    if (graph_sched_verbose_env() >= 4)
        std::cerr << "graph_recorder: checkpoint pass: idempotent_tasks=" << checkpoint_idempotent_task_count
                  << " wrr_tasks=" << checkpoint_wrr_task_count
                  << " checkpointable_tasks=" << checkpointable_task_count
                  << " inserted_checkpoints=" << inserted_checkpoints
                  << " checkpoint_max=" << graph_sched_checkpoint_max_env()
                  << std::endl;

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
    return stats;
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
        std::vector<GraphHandleAccess> replay_handle_accesses;
        unsigned added_invalidate_submit = 0;
        {
            std::unique_lock<std::mutex> lock(data->policy_mutex);
            if (data->graph_record_nested == 0)
                break;
            data->graph_record_nested--;
            if (data->graph_record_nested == 0) {
                added_invalidate_submit = data->graph_added_invalidate_submit;
                replay = std::move(data->graph_ops);
                replay_handle_accesses = std::move(data->graph_handle_accesses);
                data->graph_handle_accesses.clear();
                data->graph_handle_access_lists.clear();
            }
        }
        GraphReplayStats stats =
            graph_sched_replay_recorded_ops(std::move(replay), std::move(replay_handle_accesses), added_invalidate_submit);
        {
            std::lock_guard<std::mutex> lock(data->policy_mutex);
            data->graph_total_checkpoint_inserts += stats.inserted_checkpoints;
            data->graph_total_synthetic_invalidate_inserts += stats.added_invalidate_submit;
        }
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
    std::vector<GraphHandleAccess> replay_handle_accesses;
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
            replay_handle_accesses = std::move(data->graph_handle_accesses);
            data->graph_handle_accesses.clear();
            data->graph_handle_access_lists.clear();
            outermost_end = true;
        }
    }

    if (outermost_end) {
        GraphReplayStats stats =
            graph_sched_replay_recorded_ops(std::move(replay), std::move(replay_handle_accesses), added_invalidate_submit);
        std::lock_guard<std::mutex> lock(data->policy_mutex);
        data->graph_total_checkpoint_inserts += stats.inserted_checkpoints;
        data->graph_total_synthetic_invalidate_inserts += stats.added_invalidate_submit;
    }

    _starpu_graph_recording_pop();
}

} /* extern "C" */
