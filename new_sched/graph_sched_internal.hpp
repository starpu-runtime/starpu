/* Shared types for the SGOC graph scheduler (libgraph_sgoc_sched.so). graph_recorder.cpp + libgraph_sched.cpp
 * remain in-tree as a reference for batch/minibatch-oriented features and are not built by new_sched/Makefile. */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <list>
#include <map>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <utility>

#include <starpu.h>

#include <cstring>

#include <starpu_sched_ctx.h>
#include <starpu_scheduler.h>

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
    /**
     * INVALIDATE only: set by graph_sched_insert_missing_pre_write_invalidates (auto pre-W invalidates).
     * Omitted from batch-compat trace so comparison matches the application graph (explicit invalidate_submit only).
     */
    bool capture_synthetic_invalidate = false;
    /**
     * SGOC linked-list capture: unique id until linearized to graph_ops; pred/succ reference peer stable ids during
     * capture, then remap to dense indices.
     */
    size_t capture_stable_id = 0;
};

/** Per-task capture signature for batch-repeatability checks (codelet name + buffer footprint; see graph_recorder). */
struct GraphBatchTaskStructureSig {
    std::string codelet_name;
    std::vector<size_t> buffer_sizes;
    /** Filled only for batch-0 template matching; when empty, equality checks ignore modes (see graph_recorder). */
    std::vector<unsigned> buffer_modes;
};

/** One step in capture order for batch-vs-batch0 compat: TASK row or explicit invalidate_submit (see graph_recorder). */
struct GraphBatchCompatStep {
    bool is_invalidate = false;
    /** Index into parallel TASK signature / handle rows (TASK steps only). */
    size_t task_row = 0;
    /** INVALIDATE steps only: handle passed to invalidate_submit. */
    starpu_data_handle_t invalidate_handle = nullptr;
};

/**
 * Batch-0 optimized flush sequence: either the i-th TASK in user capture order, or an invalidate_submit on a
 * batch-0 template handle (mapped to the current batch at replay). Built from full graph_ops including synthetic
 * pre-W invalidates. See graph_sched_replay_linear_from_batch0_plan.
 */
struct GraphBatchReplayStep {
    enum Kind : std::uint8_t { USER_TASK = 0, INVALIDATE_HINT = 1 } kind = USER_TASK;
    /** USER_TASK: index in user TASK list (0 .. n_tasks-1). */
    size_t user_task_index = 0;
    /** INVALIDATE_HINT: handle from batch-0 recording; replay maps to current batch via TASK-buffer pairing. */
    starpu_data_handle_t batch0_invalidate_handle = nullptr;
    bool synthetic_invalidate = false;
};

/** One batch-0 task worth of synthetic invalidate_submit calls to run before that task (buffer indices into the task). */
struct graph_batch0_task_hint_entry {
    std::vector<unsigned> prefix_invalidate_buffer_indices;
};

/** Cached CUDA budget offload plan from batch-0 linear MM optimization; later outer batches reuse without replanning. */
struct graph_sched_mem_offload_plan {
    bool valid = false;
    /** Snapshot of effective budget (bytes) when the plan was computed; informational — replay does not require equality. */
    std::int64_t budget_bytes = -1;
    /** Peak simulated GPU bytes from the last linear offload plan (bytes); used for logs when reusing the plan. */
    std::int64_t peak_pga_bytes = 0;
    std::int64_t sum_s_bytes = 0;
    /** Handles to cycle through CPU RAM during minibatch; stable order for replay. */
    std::vector<void *> s_offload_keys;
    /**
     * Parallel to batch-0 replay topo order (same length as cached topo): simulation lists and derived exec placement.
     * Indexed by topo slot (includes INVALIDATE steps; TASK-only rows carry handle lists).
     */
    std::vector<std::vector<void *>> topo_offload_before_task;
    std::vector<std::vector<void *>> topo_prefetch_before_task;
    /** Derived: offload after this topo TASK completes (last-use-before-offload mapping). */
    std::vector<std::vector<void *>> topo_post_exec_offload_order;
    /** Derived: GPU prefetch at pop of this topo TASK (anchor); may be earlier than the consumer that needs the data. */
    std::vector<std::vector<void *>> topo_pre_exec_prefetch_order;
};

/**
 * GPU memory planning snapshot for graph_recorder: linear replay over a topo order with simulated offload-before-task
 * marks (role-agnostic victim selection). Planning only — no StarPU moves.
 */
struct graph_sched_gpu_memory_manager {
    /** Distinct persistent handles selected for offload, first-seen order during the planning pass. */
    std::deque<void *> offload_prefetch_fifo;
    /** Peak simulated GPU bytes along the modeled replay (invalidate / pure-W / prefetch-from-offload). */
    std::int64_t peak_simulated_bytes = 0;
    /** Sum of unique optimizer-state (S) handle sizes from capture (logging). */
    std::int64_t sum_s_unique_bytes = 0;
    std::int64_t budget_bytes = 0;
    /** Bytes modeled before the first op (live handles + forced P/S resident). */
    std::int64_t initial_resident_bytes = 0;
    /** Peak bytes after applying offload marks in simulation (same as peak_simulated when hints fit). */
    std::int64_t peak_after_plan_bytes = 0;
    /** Total offload-hint insertions (same handle may be counted more than once if re-offloaded later). */
    unsigned offload_mark_events = 0;
    /** Distinct handles that received at least one offload mark. */
    unsigned marked_offload_unique = 0;
    /** Parallel to replay topo_order index: simulated offload-before-task marks (planning pass order per slot). */
    std::vector<std::vector<void *>> topo_offload_before_task;
    /** Parallel to topo index: simulated prefetch-before-task (from offload state). */
    std::vector<std::vector<void *>> topo_prefetch_before_task;
    /** Parallel to topo index: handles to offload in post_exec of this TASK (reverse walk + wrap passes). */
    std::vector<std::vector<void *>> topo_post_exec_offload_order;
    /** Parallel to topo index: handles to prefetch at pop of this TASK (earliest slot with simulated budget headroom). */
    std::vector<std::vector<void *>> topo_pre_exec_prefetch_order;
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
    /**
     * Separate lock for deferred S offload (task→handles map + pending GPU evict queue). post_exec and pop_task
     * interact with this without touching the ready queue, avoiding deadlocks with push/pop on policy_mutex.
     */
    std::mutex graph_offload_mutex;
    /**
     * When true, starpu_data_register_victim_selector is active: StarPU may call our Belady callback for implicit GPU
     * allocation/reuse pressure. Explicit S-offloads still use graph_pending_gpu_evict_handles + drain (RAM-valid
     * then starpu_data_evict_from_node) so offload is not assumed instantaneous.
     */
    bool graph_runtime_starpu_victim = false;
    std::deque<struct starpu_task *> ready_queue;
    const char *policy_log_name = "sgoc";

    /** Operations in capture order; synthetic ops may be inserted and indexed here too. */
    std::vector<GraphOp> graph_ops;

    /** Linked per-handle access chains for task buffers and invalidate hints. */
    std::vector<GraphHandleAccess> graph_handle_accesses;
    std::unordered_map<void *, GraphHandleAccessList> graph_handle_access_lists;

    unsigned graph_record_nested = 0;
    /** Wall start of outermost capture: set after recording_begin's starpu_task_wait_for_all (not before). */
    std::chrono::steady_clock::time_point graph_capture_wall_start{};

    /** Capture-time only: synthetic invalidate_submit before pure STARPU_W; reset at outermost recording_begin. */
    unsigned graph_added_invalidate_submit = 0;

    /** Cumulative replay stats across all recording sessions for this policy instance. */
    unsigned graph_total_checkpoint_inserts = 0;
    /** Sum of capture_pre_write invalidates only (flush-time checkpoint invalidates are not included). */
    unsigned graph_total_synthetic_invalidate_inserts = 0;

    /** STARPU_GRAPH_SCHED_WORKER → CUDA worker id (cuda:n; devid then ordinal); -1 if unset. */
    int graph_pinned_worker_id = -1;
    /** starpu_worker_get_memory_node(pinned worker); -1 until graph_sched_read_pinned_worker_memory_into runs. */
    int graph_pinned_worker_mem_node = -1;

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

    /** Last successful automatic S offload plan (compatible batch reuses). */
    graph_sched_mem_offload_plan graph_mem_offload_plan;

    /** Latest linear topo offload-plan snapshot (see graph_sched_gpu_mm_plan_linear_topo_offloads). */
    graph_sched_gpu_memory_manager graph_gpu_mm;

    /**
     * After async RAM prefetch from post_exec offload hints: handles queued for GPU eviction once
     * starpu_data_can_evict allows (RAM replica valid). Drained from pop_task, pre_exec, and end of post_exec offload.
     */
    std::vector<starpu_data_handle_t> graph_pending_gpu_evict_handles;
    /** Mirror of graph_pending_gpu_evict_handles.size(); updated under graph_offload_mutex. Skips drain in pop_task when 0. */
    std::atomic<std::size_t> graph_pending_gpu_evict_pending_count{0};
    /**
     * After \p task completes (post_exec), async RAM replicate for planned S-offloads; handles are queued for GPU
     * eviction (graph_pending_gpu_evict_handles) and drain runs from post_exec, pop_task, pre_exec, and before GPU fetch.
     */
    std::unordered_map<struct starpu_task *, std::vector<starpu_data_handle_t>> graph_offload_after_task_handles;
    /** Successful starpu_data_evict_from_node calls from drain_pending (diagnostics). */
    unsigned graph_pending_gpu_evict_drained = 0;

    /**
     * Depth > 0 while graph_sched_replay_recorded_ops runs (flush replay). Used to attribute pop/post_exec time to
     * graph flush vs all other scheduled tasks.
     */
    std::atomic<int> graph_replay_accounting_depth{0};

    /** Sum of pop_task durations over all threads and calls (can exceed wall-clock if workers run in parallel). */
    std::atomic<std::uint64_t> graph_sched_pop_time_ns{0};
    std::atomic<unsigned> graph_sched_pop_calls{0};
    /** pop_task time/calls only while graph_replay_accounting_depth > 0 (graph flush replay path). */
    std::atomic<std::uint64_t> graph_sched_pop_time_ns_replay{0};
    std::atomic<unsigned> graph_sched_pop_calls_replay{0};

    /** Same split for post_exec_hook. */
    std::atomic<std::uint64_t> graph_sched_post_exec_time_ns{0};
    std::atomic<unsigned> graph_sched_post_exec_calls{0};
    std::atomic<std::uint64_t> graph_sched_post_exec_time_ns_replay{0};
    std::atomic<unsigned> graph_sched_post_exec_calls_replay{0};

    /** Cumulative wall time spent in graph_sched_replay_recorded_ops from flush start until replay submit begins. */
    std::atomic<std::uint64_t> graph_sched_graph_planning_time_ns{0};
    std::atomic<unsigned> graph_sched_graph_planning_flush_count{0};

    /** Cumulative wall time between outermost starpu_graph_sched_graph_recording_begin/end (capture only; no replay). */
    std::atomic<std::uint64_t> graph_sched_graph_capture_wall_time_ns{0};
    std::atomic<unsigned> graph_sched_graph_capture_sessions{0};

    /**
     * Batch-0 restricted optimization template (P/G/A/S + pre-W invalidates + post-optimizer G invalidates).
     * When valid, batch iteration != 0 uses incremental replay in push_task; graph capture hooks are disabled.
     * Not set when STARPU_GRAPH_SCHED_DEBUG_SIMPLE is on (default): only signatures/handles are stored for compat logs.
     */
    bool graph_batch0_hints_valid = false;
    /** Bitmask: 1=P/G/A/S classified, 2=pre-STARPU_W invalidates (replay), 4=post-optimizer G invalidates. */
    std::uint32_t graph_batch0_optimization_flags = 0;
    /** TASK-only order for batch 0 (same order as linear scan of graph TASK ops). */
    std::vector<GraphBatchTaskStructureSig> graph_batch0_task_structure_sigs;
    /** Batch 0: TASK + INVALIDATE steps in full recording session order (no per-op batch tag filter; see graph_recorder). */
    std::vector<GraphBatchCompatStep> graph_batch0_compat_trace;
    /**
     * Batch 0: full extended replay list (USER_TASK indices + invalidate hints with batch-0 handles).
     * Used for outer iteration > 0 when STARPU_GRAPH_SCHED_BATCH0_PLAN_REPLAY is on: skip re-inserting synthetic
     * invalidates during capture and replay this plan with mapped handles.
     */
    std::vector<GraphBatchReplayStep> graph_batch0_extended_replay_plan;
    bool graph_batch0_extended_replay_plan_valid = false;
    std::vector<graph_batch0_task_hint_entry> graph_batch0_task_hints;
    /** Slots (task_index_in_batch0, buffer_index) that are optimizer \e states in batch 0 (STARPU_W there). */
    std::unordered_set<std::uint64_t> graph_batch0_state_buffer_slots;
    /** Slots for parameter \e gradients (G): allow STARPU_W vs STARPU_RW across batches after first touch. */
    std::unordered_set<std::uint64_t> graph_batch0_gradient_buffer_slots;
    /** If < n_task, after that task submit run post_optimizer_gradient invalidates on listed buffer indices. */
    size_t graph_batch0_post_optimizer_task_index = static_cast<size_t>(-1);
    std::vector<unsigned> graph_batch0_post_optimizer_gradient_buffer_indices;

    /**
     * Incremental replay: map footprint key (codelet name + buffer sizes) -> template indices (capture order).
     * Ambiguous keys (duplicate structure) are resolved by matching starpu_data_handle_t per buffer to batch-0 refs.
     */
    std::unordered_map<std::string, std::vector<size_t>> graph_batch0_footprint_tpl_index_lists;
    /** Batch-0 STARPU handles per template index (same order as graph_batch0_task_structure_sigs). */
    std::vector<std::vector<starpu_data_handle_t>> graph_batch0_task_template_handles;
    /** Per outer-iteration pass: each template index is consumed at most once per batch. */
    std::vector<unsigned char> graph_batch0_tpl_consumed;

    /** Last outer sched_ctx iteration (slot 0) seen in incremental push; used to reset per-batch cursors. */
    long graph_push_last_outer_iteration = -1;

    /**
     * SGOC (graph_sgoc): flush-time hints and runtime GPU transfer bookkeeping. Null when policy is not sgoc.
     */
    struct graph_sgoc_runtime {
        /** Legacy: MM anchor prefetch lists (no longer issued at pop; GPU uses demand fetch for the popped task). */
        std::unordered_map<struct starpu_task *, std::vector<starpu_data_handle_t>> pre_exec_prefetch;
        /** Reserved; not used in the demand-fetch pop path. */
        std::unordered_map<struct starpu_task *, std::vector<starpu_data_handle_t>> post_exec_prefetch;
        /** RAM offload then GPU evict queue registration targets (post_exec of prior task). */
        std::unordered_map<struct starpu_task *, std::vector<starpu_data_handle_t>> post_exec_offload_order;
        /** Handles whose GPU demand fetch was deferred (e.g. VRAM budget); drained from pre_exec/post_exec. */
        std::deque<starpu_data_handle_t> deferred_prefetch;
        /** Optimistic model: handles we issued a GPU fetch for and treat as GPU-resident until evict drains / victim picks. */
        std::unordered_set<void *> tracked_gpu_resident;
        std::int64_t tracked_gpu_bytes = 0;
        /** Protects Belady tables + victim eviction predict bookkeeping (victim selector vs post_exec). */
        std::mutex victim_state_mutex;
        /** Replay topo slot (0..len-1) per TASK pointer for the last flushed graph (Belady future reference). */
        std::unordered_map<const struct starpu_task *, unsigned> belady_task_topo_slot;
        /** Remaining TASK topo slots where each handle is still referenced (multiset as map<slot,count>). */
        std::unordered_map<void *, std::map<size_t, unsigned>> belady_remaining_slots;
        /** Handles we optimistically removed from tracked_gpu_* when returning them from the victim selector. */
        std::unordered_set<void *> victim_evict_predict_removed;
        int mm_execute = 0;
        unsigned gpu_mem_node = 0;
        std::int64_t mem_budget_bytes = 0;
        /** After flush-time quiescence: handles valid on pinned GPU (starpu_data_query_status2 v bit). */
        std::unordered_set<void *> flush_starpu_gpu_resident;
        /** Same snapshot pass: handle valid on main RAM but not on the pinned GPU node (initialized, not GPU-resident). */
        std::unordered_set<void *> flush_starpu_ram_valid_not_gpu;

        /** Capture order (SGOC); synthetic invalidates inserted in O(1). Linearized to graph_ops at recording end. */
        std::list<GraphOp> capture_ops;
        std::unordered_map<size_t, std::list<GraphOp>::iterator> capture_id_to_iter;
        size_t capture_next_stable_id = 1;

        /** Non-zero when STARPU_GRAPH_SCHED_SGOC_MEM_DEBUG is set (flush-scoped). */
        int mem_debug = 0;
        std::atomic<std::uint64_t> dbg_offload_ram_issue{0};
        std::atomic<std::uint64_t> dbg_offload_ram_bytes{0};
        std::atomic<std::uint64_t> dbg_gpu_prefetch_issue{0};
        std::atomic<std::uint64_t> dbg_gpu_prefetch_bytes{0};
        std::atomic<std::uint64_t> dbg_evict_ok{0};
    };
    std::unique_ptr<graph_sgoc_runtime> graph_sgoc;

    /** push_task: tasks that matched batch-0 template and used incremental path (pin + enqueue). */
    std::atomic<std::uint64_t> graph_stat_push_incremental{0};
    /** push_task: tasks queued on the standard path (includes outer iter 0, incremental disabled, or no template match). */
    std::atomic<std::uint64_t> graph_stat_push_fifo{0};
    /** Times batch-0 hints were invalidated (template mismatch / missing footprint / exhausted templates). */
    std::atomic<std::uint64_t> graph_stat_batch0_hints_invalidate{0};
    /** sched_ctx outer iteration (slot 0) at first hints invalidation, or -1 if never. */
    std::atomic<long> graph_batch0_first_invalidate_outer_b{-1};
    /** Log once: first incremental success when outer iteration is 1 (second batch if iter 0 is first). */
    std::atomic<bool> graph_logged_iter1_incremental_ok{false};
};

/** RAII: increment graph_replay_accounting_depth for the dynamic scope of graph_sched_replay_recorded_ops. */
struct graph_sched_replay_accounting_scope {
    graph_sched_data *d = nullptr;
    explicit graph_sched_replay_accounting_scope(graph_sched_data *p) : d(p)
    {
        if (d)
            d->graph_replay_accounting_depth.fetch_add(1, std::memory_order_relaxed);
    }
    graph_sched_replay_accounting_scope(const graph_sched_replay_accounting_scope &) = delete;
    graph_sched_replay_accounting_scope &operator=(const graph_sched_replay_accounting_scope &) = delete;
    ~graph_sched_replay_accounting_scope()
    {
        if (d)
            d->graph_replay_accounting_depth.fetch_sub(1, std::memory_order_relaxed);
    }
};

/** Policy data for sched_ctx when the active policy is sgoc (shared recording API). */
graph_sched_data *graph_sched_graph_policy_data(unsigned sched_ctx_id);

/** graph_recorder.cpp (reference) — outermost capture flush with batch/minibatch template and checkpoints. */
void graph_sched_recorder_release_outermost_capture(graph_sched_data *data, std::vector<GraphOp> replay,
                                                    std::vector<GraphHandleAccess> replay_ha,
                                                    graph_sched_captured_handle_groups &parsed, bool has_batch,
                                                    std::uint32_t batch_val, int vb, unsigned sched_ctx_id);

void graph_sched_recorder_register(graph_sched_data *data);
void graph_sched_recorder_deinit(graph_sched_data *data, unsigned sched_ctx_id);

namespace graph_sgoc_bundle {
/** Non-zero if STARPU_GRAPH_SCHED_SGOC_MEM_DEBUG enables MM offload/prefetch advance logging. */
int graph_sched_sgoc_mem_debug_env(void);
/** Log planned topo-slot advance for prefetches/offloads (requires populated \p mm lists). */
void graph_sched_sgoc_log_mm_plan_advance_debug(const std::vector<GraphOp> &ops, const std::vector<size_t> &topo_order,
                                                const graph_sched_gpu_memory_manager &mm);
} /* namespace graph_sgoc_bundle */

void graph_sched_sgoc_pre_exec_hook(graph_sched_data *data, struct starpu_task *task);
void graph_sched_sgoc_post_exec_hook(graph_sched_data *data, struct starpu_task *task, unsigned gpu_mem_node);
void graph_sched_sgoc_pop_prefetch_hook(graph_sched_data *data, struct starpu_task *task);

/** graph_sgoc.cpp — SGOC flush (greedy memory topo, GPU MM, synthetic invalidates). */
void graph_sched_sgoc_release_outermost_capture(graph_sched_data *data, std::vector<GraphOp> replay,
                                                std::vector<GraphHandleAccess> replay_ha,
                                                graph_sched_captured_handle_groups &parsed, bool has_batch,
                                                std::uint32_t batch_val, int vb, unsigned sched_ctx_id);

void graph_sched_sgoc_clear_runtime(graph_sched_data *data);

/** Belady victim selector (starpu_data_register_victim_selector); no-op when built with STARPUSGOC_HAS_VICTIM_SELECTOR=0. */
void graph_sched_sgoc_victim_policy_init(graph_sched_data *data);
void graph_sched_sgoc_victim_policy_deinit(graph_sched_data *data);
void graph_sched_sgoc_victim_rebuild_belady(graph_sched_data *data, const std::vector<size_t> &topo_order);
void graph_sched_sgoc_victim_note_task_completed(graph_sched_data *data, struct starpu_task *task);
void graph_sched_sgoc_victim_clear_belady(graph_sched_data *data);
void graph_sched_sgoc_register(graph_sched_data *data);
void graph_sched_sgoc_deinit(graph_sched_data *data, unsigned sched_ctx_id);
void graph_sched_account_outermost_capture_end(graph_sched_data *data);

/** Policy init: resolve STARPU_GRAPH_SCHED_WORKER into graph_pinned_worker_id and log target. */
void graph_sched_init_pinned_worker(graph_sched_data *data);

/** Register S handles to offload after \p task completes (replay boundary); actual RAM prefetch + pending list in post_exec. */
void graph_sched_register_offload_after_task(graph_sched_data *data, struct starpu_task *task,
                                             const std::vector<void *> &s_offload_keys);
void graph_sched_run_post_exec_offloads(graph_sched_data *data, struct starpu_task *task, unsigned gpu_mem_node);
void graph_sched_drain_pending_gpu_evicts(graph_sched_data *data, unsigned gpu_mem_node);
void graph_sched_clear_offload_task_registrations(graph_sched_data *data);

/**
 * When batch-0 hints are active and outer iteration (slot 0) is > 0: match task to batch-0 template (footprint +
 * handle identity, with structural fallback), apply worker pin, enqueue on ready_queue + starpu_push_task_end.
 * Synthetic invalidates are not run here (unsafe from inside push_task); they run only on batch-0 linear replay.
 */
bool graph_sched_try_push_task_incremental(graph_sched_data *data, struct starpu_task *task);

/** Outer sched_ctx iteration slot 0 for this task (batch / epoch index), or -1 if unavailable. */
long graph_sched_task_outer_iteration(struct starpu_task *task);
