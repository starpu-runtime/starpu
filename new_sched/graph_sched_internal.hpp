/* Shared types for graph_recorder policy (libgraph_sched + graph_recorder). */

#pragma once

#include <deque>
#include <list>
#include <mutex>
#include <unordered_map>

#include <starpu.h>

struct GraphRecordedOp {
    enum Kind { TASK, INVALIDATE, WONT_USE } kind;
    struct starpu_task *task;
    starpu_data_handle_t handle;
};

struct graph_sched_data {
    std::mutex policy_mutex;
    std::deque<struct starpu_task *> ready_queue;
    const char *policy_log_name = "graph_recorder";

    /** Recorded ops in replay order (linked list for mid-sequence inserts). */
    std::list<GraphRecordedOp> graph_record_ops;

    /**
     * For each data handle, ordered list of iterators into graph_record_ops for TASK
     * entries that reference that handle (any access mode). Used to locate the last
     * such task before a write-only (STARPU_W, not STARPU_RW) access and to insert a
     * synthetic invalidate_submit after it when the user has not recorded invalidate_submit
     * for that handle since that task.
     */
    std::unordered_map<void *, std::list<std::list<GraphRecordedOp>::iterator>> graph_handle_task_refs;

    unsigned graph_record_nested = 0;

    /** Placeholder for future checkpoint-injected tasks; reset at outermost recording_begin. */
    unsigned graph_checkpointed_tasks = 0;

    /** Synthetic invalidate_submit ops this session; reset at outermost recording_begin. */
    unsigned graph_added_invalidate_submit = 0;
};
