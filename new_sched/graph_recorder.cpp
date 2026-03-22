/* Graph recording for graph_recorder policy: queues task_insert / invalidate_submit / wont_use
 * while a session is open, then replays via StarPU impl symbols (see starpu_graph_recorder.h).
 * Built as part of libgraph_sched.cpp (single TU; graph_sched_internal.hpp defines policy data). */

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
#include <utility>
#include <vector>

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

/** Heap copy of packed cl_arg; StarPU frees via STARPU_CL_ARGS (cl_arg_free). */
static void *graph_dup_cl_arg(const void *src, size_t n)
{
    if (!src || n == 0)
        return nullptr;
    void *p = std::malloc(n);
    if (!p)
        return nullptr;
    std::memcpy(p, src, n);
    return p;
}

/** Pure write (not STARPU_RW / RMW): overwrite may require a prior invalidate_submit. */
static bool graph_mode_is_write_only_overwrite(enum starpu_data_access_mode mode)
{
    if ((mode & STARPU_SCRATCH) != 0)
        return false;
    if ((mode & STARPU_W) == 0)
        return false;
    /* STARPU_RW implies a read of the current value; do not inject invalidate before it. */
    return (mode & STARPU_R) == 0;
}

/**
 * If handle H will be write-only (STARPU_W) and some task already used H, insert a
 * synthetic invalidate_submit after that last task only when the user has not
 * already recorded invalidate_submit for H between that task and this write.
 */
static void graph_sched_insert_missing_pre_write_invalidates(graph_sched_data *data, struct starpu_task *task)
{
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

        auto &refs = data->graph_handle_task_refs[static_cast<void *>(h)];
        if (refs.empty())
            continue;

        std::list<GraphRecordedOp>::iterator last_task_it = refs.back();

        bool user_invalidated = false;
        for (auto scan = std::next(last_task_it); scan != data->graph_record_ops.end(); ++scan) {
            if (scan->kind == GraphRecordedOp::INVALIDATE && scan->handle == h) {
                user_invalidated = true;
                break;
            }
        }

        if (!user_invalidated) {
            GraphRecordedOp inv{};
            inv.kind = GraphRecordedOp::INVALIDATE;
            inv.task = nullptr;
            inv.handle = h;
            data->graph_record_ops.insert(std::next(last_task_it), inv);
            data->graph_added_invalidate_submit++;
        }
    }
}

static void graph_sched_register_task_handles(graph_sched_data *data, std::list<GraphRecordedOp>::iterator task_it,
                                              struct starpu_task *task)
{
    if (!task->cl)
        return;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(task, i);
        if (!h)
            continue;
        data->graph_handle_task_refs[static_cast<void *>(h)].push_back(task_it);
    }
}

static bool graph_task_reads_handle(struct starpu_task *task, starpu_data_handle_t h)
{
    if (!task->cl)
        return false;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    for (unsigned i = 0; i < nbuf; i++) {
        if (STARPU_TASK_GET_HANDLE(task, i) != h)
            continue;
        enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, i);
        if ((mode & STARPU_SCRATCH) != 0)
            continue;
        return (mode & STARPU_R) != 0;
    }
    return false;
}

/** W1: task writes \p h (any non-scratch STARPU_W). */
static bool graph_task_writes_handle(struct starpu_task *task, starpu_data_handle_t h)
{
    if (!task->cl)
        return false;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    for (unsigned i = 0; i < nbuf; i++) {
        if (STARPU_TASK_GET_HANDLE(task, i) != h)
            continue;
        enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, i);
        if ((mode & STARPU_SCRATCH) != 0)
            continue;
        return (mode & STARPU_W) != 0;
    }
    return false;
}

/**
 * Checkpoint W1 must use exactly one STARPU_W buffer and otherwise only STARPU_R or STARPU_SCRATCH
 * (no STARPU_RW, redux, or scratch combined with R/W).
 */
static bool graph_task_is_valid_w1_access_shape(struct starpu_task *task)
{
    if (!task->cl)
        return false;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(task);
    if (nbuf == 0)
        return false;

    const enum starpu_data_access_mode allowed_annotations =
        (enum starpu_data_access_mode)(STARPU_COMMUTE | STARPU_SSEND | STARPU_LOCALITY | STARPU_NOFOOTPRINT | STARPU_NOPLAN
                                       | STARPU_UNMAP);

    unsigned n_write = 0;
    for (unsigned i = 0; i < nbuf; i++) {
        enum starpu_data_access_mode m = STARPU_TASK_GET_MODE(task, i);
        if (m & (STARPU_REDUX | STARPU_MPI_REDUX))
            return false;
        if (m & ~(STARPU_R | STARPU_W | STARPU_SCRATCH | allowed_annotations))
            return false;

        const bool r = (m & STARPU_R) != 0;
        const bool w = (m & STARPU_W) != 0;
        const bool s = (m & STARPU_SCRATCH) != 0;
        if (r && w)
            return false; /* STARPU_RW */
        if (s && (r || w))
            return false;
        if (!r && !w && !s)
            return false;

        if (w)
            n_write++;
    }
    return n_write == 1;
}

static starpu_data_handle_t graph_find_checkpoint_handle(struct starpu_task *w1, struct starpu_task *r1,
                                                         struct starpu_task *r2)
{
    if (!w1->cl)
        return nullptr;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(w1);
    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t h = STARPU_TASK_GET_HANDLE(w1, i);
        if (!h)
            continue;
        if (!graph_task_writes_handle(w1, h))
            continue;
        if (graph_task_reads_handle(r1, h) && graph_task_reads_handle(r2, h))
            return h;
    }
    return nullptr;
}

/**
 * Same handles and access modes as \p w1 (for starpu_task_build + STARPU_DATA_MODE_ARRAY).
 * Supports STARPU_VARIABLE_NBUFFERS (uses starpu_task::nbuffers / dyn_handles).
 */
static bool graph_copy_w1_data_descrs(struct starpu_task *w1, std::vector<starpu_data_descr> &descrs_out)
{
    if (!w1->cl)
        return false;
    const unsigned nbuf = STARPU_TASK_GET_NBUFFERS(w1);
    if (nbuf == 0)
        return false;
    if (w1->cl->nbuffers == STARPU_VARIABLE_NBUFFERS && w1->nbuffers != (int)nbuf)
        return false;

    descrs_out.resize(nbuf);
    for (unsigned i = 0; i < nbuf; i++) {
        starpu_data_handle_t hi = STARPU_TASK_GET_HANDLE(w1, i);
        if (!hi)
            return false;
        descrs_out[i].handle = hi;
        descrs_out[i].mode = STARPU_TASK_GET_MODE(w1, i);
    }
    return true;
}

/**
 * Scan consecutive TASK ops in recorded order; on the first W1->R1->R2 match, insert W2 (clone of W1)
 * immediately after R1. Returns true if one checkpoint was inserted, false if no match.
 */
static bool graph_sched_try_insert_one_checkpoint(graph_sched_data *data)
{
    std::vector<std::list<GraphRecordedOp>::iterator> task_iters;
    task_iters.reserve(64);
    for (auto it = data->graph_record_ops.begin(); it != data->graph_record_ops.end(); ++it) {
        if (it->kind == GraphRecordedOp::TASK && it->task && it->task->cl)
            task_iters.push_back(it);
    }
    if (task_iters.size() < 3)
        return false;

    for (size_t i = 0; i + 2 < task_iters.size(); i++) {
        struct starpu_task *w1 = task_iters[i]->task;
        struct starpu_task *r1 = task_iters[i + 1]->task;
        struct starpu_task *r2 = task_iters[i + 2]->task;
        if (!graph_task_is_valid_w1_access_shape(w1))
            continue;
        if (!graph_find_checkpoint_handle(w1, r1, r2))
            continue;

        std::vector<starpu_data_descr> descrs;
        if (!graph_copy_w1_data_descrs(w1, descrs))
            continue;

        void *cl_arg_copy = graph_dup_cl_arg(w1->cl_arg, w1->cl_arg_size);
        if (w1->cl_arg != nullptr && w1->cl_arg_size > 0 && !cl_arg_copy)
            continue;

        unsigned sched_ctx = r1->sched_ctx;
        struct starpu_task *w2 = nullptr;
        if (cl_arg_copy)
            w2 = starpu_task_build(w1->cl, STARPU_SCHED_CTX, (int)sched_ctx, STARPU_DATA_MODE_ARRAY, descrs.data(),
                                   (int)descrs.size(), STARPU_CL_ARGS, cl_arg_copy, w1->cl_arg_size, STARPU_NAME,
                                   "graph_checkpoint", 0);
        else
            w2 = starpu_task_build(w1->cl, STARPU_SCHED_CTX, (int)sched_ctx, STARPU_DATA_MODE_ARRAY, descrs.data(),
                                   (int)descrs.size(), STARPU_NAME, "graph_checkpoint", 0);
        if (!w2) {
            std::free(cl_arg_copy);
            continue;
        }

        GraphRecordedOp op{};
        op.kind = GraphRecordedOp::TASK;
        op.task = w2;
        op.handle = nullptr;
        data->graph_record_ops.insert(std::next(task_iters[i + 1]), op);
        data->graph_checkpointed_tasks++;
        return true;
    }
    return false;
}

/** Replay-order pass: synthetic checkpoint tasks need the same pre-write invalidate logic as captured tasks. */
static void graph_sched_reapply_auto_invalidates(graph_sched_data *data)
{
    data->graph_handle_task_refs.clear();
    for (auto it = data->graph_record_ops.begin(); it != data->graph_record_ops.end(); ++it) {
        if (it->kind != GraphRecordedOp::TASK || !it->task)
            continue;
        graph_sched_insert_missing_pre_write_invalidates(data, it->task);
        graph_sched_register_task_handles(data, it, it->task);
    }
}

static void graph_sched_finalize_recorded_graph(graph_sched_data *data)
{
    /* Each insertion can create new W1->R1->R2 triples among TASK ops; repeat until stable. */
    while (graph_sched_try_insert_one_checkpoint(data))
        ;
    graph_sched_reapply_auto_invalidates(data);
}

static void graph_sched_append_captured_task(graph_sched_data *data, struct starpu_task *task)
{
    graph_sched_insert_missing_pre_write_invalidates(data, task);

    GraphRecordedOp op{};
    op.kind = GraphRecordedOp::TASK;
    op.task = task;
    op.handle = nullptr;
    data->graph_record_ops.push_back(op);
    graph_sched_register_task_handles(data, std::prev(data->graph_record_ops.end()), task);
}

/* Call only with policy_mutex released: replay submits tasks and may call push_task_graph. */
void graph_sched_replay_recorded_ops(std::list<GraphRecordedOp> ops, unsigned checkpointed_tasks,
                                     unsigned added_invalidate_submit)
{
    if (graph_sched_verbose_env() >= 1) {
        size_t n_task = 0, n_invalidate = 0, n_wont_use = 0;
        for (const GraphRecordedOp &op : ops) {
            switch (op.kind) {
            case GraphRecordedOp::TASK:
                n_task++;
                break;
            case GraphRecordedOp::INVALIDATE:
                n_invalidate++;
                break;
            case GraphRecordedOp::WONT_USE:
                n_wont_use++;
                break;
            }
        }
        std::cerr << "graph_recorder: flush recorded ops: tasks=" << n_task
                  << " invalidate_submit=" << n_invalidate << " wont_use=" << n_wont_use
                  << " checkpointed_tasks=" << checkpointed_tasks
                  << " added_invalidate_submit=" << added_invalidate_submit << std::endl;
    }

    _starpu_graph_recorder_set_flushing(1);
    for (const GraphRecordedOp &op : ops) {
        switch (op.kind) {
        case GraphRecordedOp::TASK:
            _starpu_task_insert_submit_built_task(op.task);
            break;
        case GraphRecordedOp::INVALIDATE:
            _starpu_data_invalidate_submit_impl(op.handle);
            break;
        case GraphRecordedOp::WONT_USE:
            _starpu_data_wont_use_impl(op.handle);
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
    GraphRecordedOp op{};
    op.kind = GraphRecordedOp::INVALIDATE;
    op.task = nullptr;
    op.handle = handle;
    data->graph_record_ops.push_back(op);
    return 0;
}

static int graph_sched_capture_wont_use_hook(starpu_data_handle_t handle, void *arg)
{
    auto *data = static_cast<graph_sched_data *>(arg);
    std::lock_guard<std::mutex> lock(data->policy_mutex);
    if (data->graph_record_nested == 0)
        return -1;
    GraphRecordedOp op{};
    op.kind = GraphRecordedOp::WONT_USE;
    op.task = nullptr;
    op.handle = handle;
    data->graph_record_ops.push_back(op);
    return 0;
}

void graph_sched_recorder_register(graph_sched_data *data)
{
    _starpu_graph_recorder_register(
        graph_sched_capture_task_hook,
        graph_sched_capture_invalidate_hook,
        graph_sched_capture_wont_use_hook,
        data);
}

void graph_sched_recorder_deinit(graph_sched_data *data, unsigned sched_ctx_id)
{
    (void)sched_ctx_id;
    for (;;) {
        std::list<GraphRecordedOp> replay;
        unsigned checkpointed_tasks = 0;
        unsigned added_invalidate_submit = 0;
        {
            std::unique_lock<std::mutex> lock(data->policy_mutex);
            if (data->graph_record_nested == 0)
                break;
            data->graph_record_nested--;
            if (data->graph_record_nested == 0) {
                graph_sched_finalize_recorded_graph(data);
                checkpointed_tasks = data->graph_checkpointed_tasks;
                added_invalidate_submit = data->graph_added_invalidate_submit;
                replay = std::move(data->graph_record_ops);
                data->graph_handle_task_refs.clear();
            }
        }
        graph_sched_replay_recorded_ops(std::move(replay), checkpointed_tasks, added_invalidate_submit);
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
        data->graph_record_ops.clear();
        data->graph_handle_task_refs.clear();
        data->graph_checkpointed_tasks = 0;
        data->graph_added_invalidate_submit = 0;
    }
    data->graph_record_nested++;
}

void starpu_graph_sched_graph_recording_end(unsigned sched_ctx_id)
{
    graph_sched_data *data = graph_recorder_policy_data(sched_ctx_id);
    if (!data)
        return;

    std::list<GraphRecordedOp> replay;
    bool outermost_end = false;
    unsigned checkpointed_tasks = 0;
    unsigned added_invalidate_submit = 0;
    {
        std::unique_lock<std::mutex> lock(data->policy_mutex);
        if (data->graph_record_nested == 0)
            return;

        data->graph_record_nested--;
        if (data->graph_record_nested == 0) {
            graph_sched_finalize_recorded_graph(data);
            checkpointed_tasks = data->graph_checkpointed_tasks;
            added_invalidate_submit = data->graph_added_invalidate_submit;
            replay = std::move(data->graph_record_ops);
            data->graph_handle_task_refs.clear();
            outermost_end = true;
        }
    }

    if (outermost_end)
        graph_sched_replay_recorded_ops(std::move(replay), checkpointed_tasks, added_invalidate_submit);

    _starpu_graph_recording_pop();
}

} /* extern "C" */
