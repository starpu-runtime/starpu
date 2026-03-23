/* Shared types for graph_recorder policy (libgraph_sched + graph_recorder). */

#pragma once

#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <starpu.h>

constexpr size_t GRAPH_ACCESS_NONE = static_cast<size_t>(-1);
constexpr unsigned GRAPH_ACCESS_INVALIDATE_RAW = 1u << 30;

struct GraphOpHandleAccessRef {
    starpu_data_handle_t handle = nullptr;
    unsigned mode = 0;
    size_t access_idx = GRAPH_ACCESS_NONE;
};

struct GraphHandleAccess {
    starpu_data_handle_t handle = nullptr;
    unsigned mode = 0;
    struct starpu_task *task = nullptr;
    size_t op_idx = GRAPH_ACCESS_NONE;
    size_t prev_for_handle = GRAPH_ACCESS_NONE;
    size_t next_for_handle = GRAPH_ACCESS_NONE;
};

struct GraphHandleAccessList {
    size_t head = GRAPH_ACCESS_NONE;
    size_t tail = GRAPH_ACCESS_NONE;
};

struct GraphOp {
    enum Kind { TASK, INVALIDATE } kind;
    struct starpu_task *task = nullptr;
    starpu_data_handle_t handle = nullptr;
    std::vector<GraphOpHandleAccessRef> handle_accesses;
    std::vector<size_t> dependencies;
    bool checkpoint_idempotent = false;
    bool checkpoint_wrr = false;
    bool checkpointable = false;
    /** Change in bytes required on the pinned worker memory node after this op completes (replay analysis). */
    std::int64_t memory_bytes_delta_after = 0;
    /** Expected execution time on the graph target worker (µs), from StarPU perf models; NaN if N/A or INVALIDATE. */
    double predicted_exec_time = std::numeric_limits<double>::quiet_NaN();
};

struct graph_sched_data {
    std::mutex policy_mutex;
    std::deque<struct starpu_task *> ready_queue;
    const char *policy_log_name = "graph_recorder";

    /** Operations in capture order; synthetic ops may be inserted and indexed here too. */
    std::vector<GraphOp> graph_ops;

    /** Linked per-handle access chains for task buffers and invalidate hints. */
    std::vector<GraphHandleAccess> graph_handle_accesses;
    std::unordered_map<void *, GraphHandleAccessList> graph_handle_access_lists;

    unsigned graph_record_nested = 0;

    /** Capture-time only: synthetic invalidate_submit before pure STARPU_W; reset at outermost recording_begin. */
    unsigned graph_added_invalidate_submit = 0;

    /** Cumulative replay stats across all recording sessions for this policy instance. */
    unsigned graph_total_checkpoint_inserts = 0;
    /** Sum of capture_pre_write invalidates only (flush-time checkpoint invalidates are not included). */
    unsigned graph_total_synthetic_invalidate_inserts = 0;

    /** STARPU_GRAPH_SCHED_WORKER → starpu_worker_get_by_devid; -1 if unset or not resolved. */
    int graph_pinned_worker_id = -1;
};

/** Policy init: resolve STARPU_GRAPH_SCHED_WORKER into graph_pinned_worker_id and log target. */
void graph_sched_init_pinned_worker(graph_sched_data *data);
