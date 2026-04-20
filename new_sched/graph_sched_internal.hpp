/* Shared types for graph_recorder policy (libgraph_sched + graph_recorder). */

#pragma once

#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <string>
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
    /** Op indices that must complete before this op (tasks and INVALIDATE). */
    std::vector<size_t> predecessors;
    /** Op indices that depend on this op (tasks and INVALIDATE). */
    std::vector<size_t> successors;
    /** Single pure-W producer whose written handle is a parsed checkpointable activation (see graph_sched_parse_captured_data_handles). */
    bool checkpoint_idempotent = false;
    /** Same as checkpoint_idempotent: activation producers are treated as WRR-eligible for checkpoint insertion. */
    bool checkpoint_wrr = false;
    /** Expected execution time on the graph target worker (µs), from StarPU perf models; NaN if N/A or INVALIDATE. */
    double predicted_exec_time = std::numeric_limits<double>::quiet_NaN();

    /**
     * Outer sched_ctx iteration from starpu_sched_ctx_get_iteration(ctx, 0) when set (batch / epoch index with nested
     * starpu_iteration_push). Independent of graph_stage_subiteration (inner slot or single-level fallback).
     */
    bool graph_stage_batch_iteration_valid = false;
    std::uint32_t graph_stage_batch_iteration = 0;

    /**
     * Sched_ctx iteration at capture when set: get_iteration(ctx,1) if nested push, else get_iteration(ctx,0).
     * Convention: 0 = preparation (e.g. clear gradients); 1 .. UINT32_MAX-1 = minibatch index;
     * UINT32_MAX (4294967295) = optimizer step.
     */
    bool graph_stage_subiteration_valid = false;
    std::uint32_t graph_stage_subiteration = 0;
};

/** Per-task capture signature for batch-repeatability checks (codelet name + buffer footprint; see graph_recorder). */
struct GraphBatchTaskStructureSig {
    std::string codelet_name;
    std::vector<size_t> buffer_sizes;
};

/** Handles classified from a finished graph capture (see graph_sched_parse_captured_data_handles). */
struct graph_sched_captured_handle_groups {
    /** Weights etc.: optimizer-step candidates that also appear on any task in subiteration 1 (first forward). */
    std::vector<starpu_data_handle_t> parameters;
    /** Gradient tensors: pure ::STARPU_R on optimizer step (subiteration UINT32_MAX). */
    std::vector<starpu_data_handle_t> gradients;
    /** Optimizer buffers (e.g. Adam m/v) not touched in first forward. */
    std::vector<starpu_data_handle_t> states;
    /** Checkpointable activations: produced in subiter 1 (one W, ≥1 R, no RW), consumed in subiter 2 (R-only).
     *  Graph checkpoint insertion uses \e only this list (never \a offloadable_activations). */
    std::vector<starpu_data_handle_t> activations;
    /** Like \a activations but subiter 1 may use ::STARPU_RW (same backward rule). Superset of checkpointable. */
    std::vector<starpu_data_handle_t> offloadable_activations;
};

/** WRR checkpoint candidate ranked by rematerialization speed (see graph_sched_replay_recorded_ops). */

struct GraphIdempotentTaskPredicted {
    struct starpu_task *task = nullptr;
    /** Expected execution time on the designated (pinned) worker in µs; +inf if StarPU has no valid estimate. */
    double predicted_exec_time_us = std::numeric_limits<double>::infinity();
    /** Distinct pure-STARPU_W buffers: sum of starpu_data_get_size (bytes to rematerialize). */
    std::int64_t rematerialization_bytes = 0;
    /** rematerialization_bytes / (predicted_exec_time_us / 1e6); 0 when time invalid or bytes 0. */
    double rematerialization_speed_bps = 0;
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

    /** STARPU_GRAPH_SCHED_WORKER → CUDA worker id (cuda:n; devid then ordinal); -1 if unset. */
    int graph_pinned_worker_id = -1;

    /**
     * Driver-reported GPU capacity (cudaMemGetInfo total / cudaGetDeviceProperties), or StarPU node total if unset.
     * Can exceed the StarPU allocation budget when STARPU_LIMIT_CUDA*_MEM caps RAM.
     */
    std::int64_t graph_pinned_worker_max_memory_bytes = -1;
    /** Companion to graph_pinned_worker_max_memory_bytes (free / available bytes). */
    std::int64_t graph_pinned_worker_available_memory_bytes = -1;
    /**
     * Scheduler budget on the pinned CUDA node: from starpu_memory_get_total / STARPU_LIMIT_* / fallbacks,
     * then clamped so it never exceeds graph_pinned_worker_available_memory_bytes when that is known.
     */
    std::int64_t graph_pinned_worker_max_allowed_memory_bytes = -1;
    /** starpu_memory_get_used(same node) — StarPU-accounted use on that node. */
    std::uint64_t graph_pinned_worker_starpu_used_bytes = 0;

    /**
     * Filled when outermost graph recording ends: checkpoint-eligible TASK ops (checkpointable-activation producers),
     * descending by rematerialization_speed_bps (fastest rematerializers first → checkpoint insertion order under checkpoint_max).
     */
    std::vector<GraphIdempotentTaskPredicted> graph_idempotent_tasks_sorted;

    /** Filled when outermost recording ends, before replay (parser runs on captured ops). */
    graph_sched_captured_handle_groups graph_captured_handle_groups;

    /** Previous flush TASK-only structure (operation + footprint); replay reuse and checkpoint compat vs previous batch. */
    bool graph_prev_flush_task_sigs_valid = false;
    std::vector<GraphBatchTaskStructureSig> graph_prev_flush_task_structure_sigs;

    /** Last computed replay submission order (indices into ops after checkpoint insertion); see graph_recorder. */
    bool graph_cached_replay_order_valid = false;
    std::vector<size_t> graph_cached_replay_op_order;
    size_t graph_cached_pre_checkpoint_op_count = 0;
    unsigned graph_cached_inserted_checkpoints = 0;
    size_t graph_cached_replay_final_op_count = 0;

    /**
     * Optimized TASK submission order within the first \e repeating minibatch pair (graph subiter 3=forward, 4=backward),
     * as a permutation of capture-order indices into that pair's TASK list. Minibatch 0 (subiter 1/2) may include
     * subiteration 0 (init); it is not used here. Later pairs (5/6), (7/8)… reuse this pattern via tie-break when the
     * minibatch chain matches.
     */
    bool graph_minibatch_pair_task_toporder_pattern_valid = false;
    std::vector<unsigned> graph_minibatch_pair_task_toporder_pattern;
    unsigned graph_minibatch_pair_task_count = 0;
};

/** Policy init: resolve STARPU_GRAPH_SCHED_WORKER into graph_pinned_worker_id and log target. */
void graph_sched_init_pinned_worker(graph_sched_data *data);
