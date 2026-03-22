/* Shared types for graph_recorder policy (libgraph_sched + graph_recorder). */

#pragma once

#include <deque>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

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

    /** Recorded ops in capture order (synthetic invalidates may be inserted; checkpoints spliced after R1 at finalize). */
    std::vector<GraphRecordedOp> graph_record_ops;

    /**
     * Index in graph_record_ops of the last TASK that references each handle (any access mode).
     * Used for synthetic pre-write invalidate insertion. Updated when inserting ops before the tail
     * shifts indices (see graph_recorder.cpp).
     */
    std::unordered_map<void *, size_t> graph_handle_last_task_idx;

    /** Extra edges (predecessor -> successor) for checkpoint tasks; cleared each recording session. */
    std::vector<std::pair<size_t, size_t>> graph_checkpoint_edges;

    unsigned graph_record_nested = 0;

    /** Checkpoint tasks (cloned producers) injected this recording session; reset at outermost recording_begin. */
    unsigned graph_checkpointed_tasks = 0;

    /** Synthetic invalidate_submit ops this session; reset at outermost recording_begin. */
    unsigned graph_added_invalidate_submit = 0;
};
