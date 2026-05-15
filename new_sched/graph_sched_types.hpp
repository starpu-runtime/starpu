/* Core data structures for the SGOC graph scheduler (no non-type API declarations). */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <limits>
#include <list>
#include <map>
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

/** Key for WRR checkpoint eligibility when aggregating activation access stats per inner minibatch (\c graph_stage_batch_iteration). */
struct graph_sched_checkpoint_act_slot {
    void *handle = nullptr;
    std::uint32_t inner_batch = 0;
    bool inner_batch_valid = false;
    bool operator==(const graph_sched_checkpoint_act_slot &o) const noexcept
    {
        return handle == o.handle && inner_batch_valid == o.inner_batch_valid && inner_batch == o.inner_batch;
    }
};

struct graph_sched_checkpoint_act_slot_hash {
    size_t operator()(const graph_sched_checkpoint_act_slot &s) const noexcept
    {
        size_t h = std::hash<void *>{}(s.handle);
        h ^= std::hash<std::uint32_t>{}(s.inner_batch) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(s.inner_batch_valid) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

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

/** Per-task capture signature (codelet name + buffer footprint + optional modes). */
struct GraphBatchTaskStructureSig {
    std::string codelet_name;
    std::vector<size_t> buffer_sizes;
    /** When non-empty, template equality may require matching modes per buffer. */
    std::vector<unsigned> buffer_modes;
};

/** WRR expansion: same codelet + buffer footprint + pure-W slot as a first–minibatch checkpointable producer. */
struct SgocWrrCheckpointTemplate {
    GraphBatchTaskStructureSig sig;
    unsigned pure_write_buffer_index = 0;
};

/**
 * WRR chain: pair reads at producer forward sub \e f and sub \e f + \c forward_backward_delta_sub (NNTile delta=1).
 */
struct graph_sched_wrr_activation_sub_policy {
    bool use_producer_relative_pair = false;
    std::uint32_t forward_backward_delta_sub = 1u;
};

/** Cached CUDA budget offload plan from linear MM optimization; a later compatible flush may reuse without replanning. */
struct graph_sched_mem_offload_plan {
    bool valid = false;
    /** Snapshot of effective budget (bytes) when the plan was computed; informational — replay does not require equality. */
    std::int64_t budget_bytes = -1;
    /** Peak simulated GPU bytes from the last linear offload plan (bytes); used for logs when reusing the plan. */
    std::int64_t peak_pga_bytes = 0;
    std::int64_t sum_s_bytes = 0;
    /** Distinct handles selected by the linear Belady offload plan (any role); stable order for replay. Legacy field
     *  name \c s_offload_* kept from when only optimizer-state (S) buffers were classified in capture. */
    std::vector<void *> s_offload_keys;
    /**
     * Parallel to replay topo order (same length as cached topo): simulation lists and derived exec placement.
     * Indexed by topo slot (includes INVALIDATE steps; TASK-only rows carry handle lists).
     */
    std::vector<std::vector<void *>> topo_offload_before_task;
    std::vector<std::vector<void *>> topo_prefetch_before_task;
    /** Derived: offload after this topo TASK completes (last-use-before-offload mapping). */
    std::vector<std::vector<void *>> topo_post_exec_offload_order;
    /** Like \a topo_post_exec_offload_order but GPU evict only (no RAM replicate); next graph touch is invalidate_submit. */
    std::vector<std::vector<void *>> topo_post_exec_evict_gpu_only_order;
    /** Derived: GPU prefetch at pop of this topo TASK (anchor); may be earlier than the consumer that needs the data. */
    std::vector<std::vector<void *>> topo_pre_exec_prefetch_order;
};

/**
 * GPU memory planning snapshot: linear replay over a topo order with simulated offload-before-task
 * marks (role-agnostic victim selection). Planning only — no StarPU moves.
 */
struct graph_sched_gpu_memory_manager {
    /** Distinct persistent handles selected for offload, first-seen order during the planning pass. */
    std::deque<void *> offload_prefetch_fifo;
    /** Peak simulated GPU bytes along the modeled replay (invalidate / pure-W / prefetch-from-offload). */
    std::int64_t peak_simulated_bytes = 0;
    /** Sum of unique optimizer-state handle sizes from capture groups (logging only; victims are not limited to S). */
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
    /** Parallel to topo index: GPU evict-only after this TASK (no RAM offload); see linear MM planner. */
    std::vector<std::vector<void *>> topo_post_exec_evict_gpu_only_order;
    /** Parallel to topo index: handles to prefetch at pop of this TASK (earliest slot with simulated budget headroom). */
    std::vector<std::vector<void *>> topo_pre_exec_prefetch_order;
    /** Count of pressure reliefs that skipped RAM replicate because the next use was invalidate_submit. */
    unsigned evict_only_mark_events = 0;
};

/** Handles classified from a finished graph capture (see graph_sched_parse_captured_data_handles). */
struct graph_sched_captured_handle_groups {
    /**
     * Smallest inner graph subiteration seen on a forward training task (odd; NNTile: first microbatch forward, 1) and
     * smallest backward training sub (even; typically 2). Used to aggregate activation stats over only that pair so
     * per-microbatch rewrites do not inflate \c f_w across the whole capture (global odd/even merge would).
     */
    bool activation_checkpoint_min_pair_valid = false;
    std::uint32_t activation_checkpoint_min_forward_sub = 0;
    std::uint32_t activation_checkpoint_min_backward_sub = 0;
    /** When \a activation_checkpoint_min_pair_valid: \c min_backward - \c min_forward (NNTile: 1). Forward read / backward
     *  read on the handle chain for a producer at sub \e f are matched at \e f and \e f + this delta. */
    std::uint32_t activation_forward_backward_delta_sub = 0;

    /** Weights etc.: optimizer-step candidates that also appear on any task in subiteration 1 (first forward). */
    std::vector<starpu_data_handle_t> parameters;
    /** Gradient tensors: pure ::STARPU_R on optimizer step (subiteration UINT32_MAX). */
    std::vector<starpu_data_handle_t> gradients;
    /** Optimizer buffers (e.g. Adam m/v) not touched in first forward. */
    std::vector<starpu_data_handle_t> states;
    /** Checkpointable activations: produced in subiter 1 (one W, ≥1 R, no RW), consumed in subiter 2 (R-only).
     *  Graph checkpoint insertion uses this set (never \a offloadable_activations alone). Handles may repeat when
     *  \c activation_checkpoint_per_inner_batch classifies the same StarPU handle in multiple inner minibatches. */
    std::vector<starpu_data_handle_t> activations;
    /** Like \a activations but subiter 1 may use ::STARPU_RW (same backward rule). Superset of checkpointable. */
    std::vector<starpu_data_handle_t> offloadable_activations;
    /**
     * When true: access stats and checkpoint targets are keyed by (\e handle, \c graph_stage_batch_iteration) for
     * subiter-1/2 tasks (requires valid batch tags on those tasks). Needed when the same buffer is (re)written each
     * inner minibatch (e.g. batch_size>1, minibatch_size=1): global merge would set f_w>1 and drop checkpointability.
     */
    bool activation_checkpoint_per_inner_batch = false;
    /** Populated when \a activation_checkpoint_per_inner_batch; producer matches on (written handle, TASK batch tag). */
    std::unordered_set<graph_sched_checkpoint_act_slot, graph_sched_checkpoint_act_slot_hash,
                      std::equal_to<graph_sched_checkpoint_act_slot>>
        checkpointable_activation_slots;
};

/** WRR checkpoint candidate ranked by rematerialization speed (flush-time ranking before replay). */

struct GraphIdempotentTaskPredicted {
    struct starpu_task *task = nullptr;
    /** Index into \c graph_ops for this activation producer (each \c GraphOp is unique; do not dedupe by \c task pointer — pool reuse). */
    size_t producer_op_idx = GRAPH_ACCESS_NONE;
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
     * Separate lock for deferred GPU→RAM offload (task→handles map + pending GPU evict queue). post_exec and pop_task
     * interact with this without touching the ready queue, avoiding deadlocks with push/pop on policy_mutex.
     */
    std::mutex graph_offload_mutex;
    /**
     * When true, starpu_data_register_victim_selector is active: StarPU may call our Belady callback for implicit GPU
     * allocation/reuse pressure. Explicit MM-planned offloads still use graph_pending_gpu_evict_handles + drain
     * (RAM-valid then starpu_data_evict_from_node) so offload is not assumed instantaneous.
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
     * Filled when outermost graph recording ends: one row per checkpoint-candidate \c GraphOp (producer index in \c graph_ops),
     * descending by rematerialization_speed_bps. Do not dedupe by \c starpu_task* — task structs are often reused/pooled.
     */
    std::vector<GraphIdempotentTaskPredicted> graph_idempotent_tasks_sorted;

    /** Filled when outermost recording ends, before replay (parser runs on captured ops). */
    graph_sched_captured_handle_groups graph_captured_handle_groups;

    /** Last successful automatic GPU MM offload plan (compatible flush may reuse). */
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
    /** GPU evict without RAM replicate (next scheduled graph touch is invalidate on that handle). */
    std::unordered_map<struct starpu_task *, std::vector<starpu_data_handle_t>> graph_evict_gpu_only_after_task_handles;
    /**
     * GPU→RAM offload: post_exec runs on the CUDA driver thread where starpu_data_prefetch_on_node(..., RAM) can
     * assert for handles not yet read-initialized. Queue handles here; drain from push_task (non-driver) with
     * starpu_data_acquire_on_node_try(..., STARPU_MAIN_RAM, STARPU_W) then release_on_node, then GPU evict.
     */
    std::deque<starpu_data_handle_t> graph_offload_defer_ram_w_acquire;
    /** Successful starpu_data_evict_from_node calls from drain_pending (diagnostics). */
    unsigned graph_pending_gpu_evict_drained = 0;

    /**
     * Depth > 0 while flush replay submits recorded graph tasks (graph_sched_sgoc_release_outermost_capture). Used to attribute pop/post_exec time to
     * graph flush vs all other scheduled tasks.
     */
    std::atomic<int> graph_replay_accounting_depth{0};

    /** Sum of pop_task durations over all threads and calls (can exceed wall-clock if workers run in parallel). */
    std::atomic<std::uint64_t> graph_sched_pop_time_ns{0};
    std::atomic<unsigned> graph_sched_pop_calls{0};
    /** pop_task time/calls only while graph_replay_accounting_depth > 0 (graph flush replay path). */
    std::atomic<std::uint64_t> graph_sched_pop_time_ns_replay{0};
    std::atomic<unsigned> graph_sched_pop_calls_replay{0};

    /**
     * SGOC pinned CUDA worker: task returned from pop had all buffers data-ready (starpu_st_non_ready_buffers_size:
     * non_ready==0 and non_allocated==0 for that worker).
     */
    std::atomic<std::uint64_t> dbg_sgoc_pop_picked_data_ready{0};
    /** SGOC pinned worker: task returned from pop still had pending transfers or unallocated buffers (non_ready>0 or non_allocated>0). */
    std::atomic<std::uint64_t> dbg_sgoc_pop_picked_data_not_ready{0};

    /** Same split for post_exec_hook. */
    std::atomic<std::uint64_t> graph_sched_post_exec_time_ns{0};
    std::atomic<unsigned> graph_sched_post_exec_calls{0};
    std::atomic<std::uint64_t> graph_sched_post_exec_time_ns_replay{0};
    std::atomic<unsigned> graph_sched_post_exec_calls_replay{0};

    /** Cumulative wall time spent in flush replay planning/submit from graph_sched_sgoc_release_outermost_capture. */
    std::atomic<std::uint64_t> graph_sched_graph_planning_time_ns{0};
    std::atomic<unsigned> graph_sched_graph_planning_flush_count{0};

    /** Cumulative wall time between outermost starpu_graph_sched_graph_recording_begin/end (capture only; no replay). */
    std::atomic<std::uint64_t> graph_sched_graph_capture_wall_time_ns{0};
    std::atomic<unsigned> graph_sched_graph_capture_sessions{0};

    /**
     * SGOC (graph_sgoc): flush-time hints and runtime GPU transfer bookkeeping. Null when policy is not sgoc.
     */
    struct graph_sgoc_runtime {
        /** RAM offload then GPU evict queue registration targets (post_exec of prior task). */
        std::unordered_map<struct starpu_task *, std::vector<starpu_data_handle_t>> post_exec_offload_order;
        /** Handles whose GPU demand fetch was deferred (e.g. VRAM budget); drained from pre_exec/post_exec. */
        std::deque<starpu_data_handle_t> deferred_prefetch;
        /** Optimistic model: handles we issued a GPU fetch for and treat as GPU-resident until evict drains / victim picks. */
        std::unordered_set<void *> tracked_gpu_resident;
        std::int64_t tracked_gpu_bytes = 0;
        /** Protects Belady tables + victim eviction predict bookkeeping (victim selector vs post_exec). */
        std::mutex victim_state_mutex;
        /**
         * Full replay topo index (INVALIDATE + TASK slots) per TASK for the last flushed graph.
         * Filled before any replay submit so \c graph_sched_sgoc_pop_prefetch_hook can issue MM anchor prefetches.
         */
        std::unordered_map<const struct starpu_task *, unsigned> replay_task_topo_slot;
        /** Replay topo slot per TASK when StarPU victim/Belady is enabled (same indices as \c replay_task_topo_slot). */
        std::unordered_map<const struct starpu_task *, unsigned> belady_task_topo_slot;
        /** Remaining TASK topo slots where each handle is still referenced (multiset as map<slot,count>). */
        std::unordered_map<void *, std::map<size_t, unsigned>> belady_remaining_slots;
        /** Handles we optimistically removed from tracked_gpu_* when returning them from the victim selector. */
        std::unordered_set<void *> victim_evict_predict_removed;
        int mm_execute = 0;
        /** Non-zero: stderr one-line MM plan vs replay ordering trace (STARPU_GRAPH_SCHED_MM_ORDER_TRACE). */
        int mm_order_trace = 0;
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
        /** MM_ORDER_TRACE: offload handles appended in register_offload_after_task (replay flush). */
        std::atomic<std::uint64_t> dbg_mm_trace_offload_regs{0};
        /** MM_ORDER_TRACE: sgoc_try_demand_fetch_handle_to_gpu calls from anchor prefetch lists. */
        std::atomic<std::uint64_t> dbg_mm_trace_anchor_fetch_try{0};
        /** MM_ORDER_TRACE: sgoc_try_demand_fetch_handle_to_gpu calls from pop_task task buffers. */
        std::atomic<std::uint64_t> dbg_mm_trace_taskbuf_fetch_try{0};
        /** MM_ORDER_TRACE: post_exec offload hook invocations with a non-empty work list. */
        std::atomic<std::uint64_t> dbg_mm_trace_post_exec_offload_tasks{0};

        /** Last completed flush: snapshot for deinit-time MM logs (ops/topo cleared before tasks finish). */
        bool mm_obs_last_flush_valid = false;
        size_t mm_obs_last_topo_slots = 0;
        size_t mm_obs_last_replay_tasks_submitted = 0;
        int mm_obs_last_mem_offload_auto = 0;
        int mm_obs_last_pin_worker = -1;
        int mm_obs_last_mm_execute = 0;
    };
    std::unique_ptr<graph_sgoc_runtime> graph_sgoc;

    /** push_task enqueue count (standard ready path). */
    std::atomic<std::uint64_t> graph_stat_push_fifo{0};
};

/** RAII: increment graph_replay_accounting_depth during flush replay submit loop. */
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
