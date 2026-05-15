/* Shared internal API for the SGOC graph scheduler (types in graph_sched_types.hpp). */

#pragma once

#include "graph_sched_types.hpp"

/** Policy data for sched_ctx when the active policy is sgoc (shared recording API). */
graph_sched_data *graph_sched_graph_policy_data(unsigned sched_ctx_id);

namespace graph_sgoc_bundle {
/** Non-zero if STARPU_GRAPH_SCHED_SGOC_MEM_DEBUG enables MM offload/prefetch advance logging. */
int graph_sched_sgoc_mem_debug_env(void);
/** Non-zero if STARPU_GRAPH_SCHED_MM_ORDER_TRACE enables per-flush MM plan vs replay ordering summary on stderr. */
int graph_sched_mm_order_trace_env(void);
/** Log planned topo-slot advance for prefetches/offloads (\p topo_slots = full replay topo length; \p mm lists). */
void graph_sched_sgoc_log_mm_plan_advance_debug(size_t topo_slots, const graph_sched_gpu_memory_manager &mm);
/** WRR checkpoints (STARPU_GRAPH_SCHED_CHECKPOINT_MAX); mutates ops/HA then rebuilds lists + op DAG. */
void graph_sgoc_apply_wrr_checkpoints_before_topo(graph_sched_data *data, const graph_sched_captured_handle_groups &parsed,
                                                  int vb);
/** Rebuild per-handle access lists and predecessor/successor DAG from \p graph_handle_accesses (post-linearize). */
void graph_sgoc_rebuild_lists_and_refresh_deps(graph_sched_data *data);
/** Parse + flush captured graph after outermost recording_end (calls \c graph_sched_sgoc_release_outermost_capture). */
void graph_sgoc_finalize_outermost_capture(graph_sched_data *data, std::vector<GraphOp> &&replay,
                                           std::vector<GraphHandleAccess> &&replay_ha,
                                           unsigned added_invalidate_submit, unsigned sched_ctx_id);
/** Linked-list capture → dense \p graph_ops (recording_end / deinit). */
void graph_sgoc_linearize_capture_to_ops(graph_sched_data *data);
/** Append one captured task under policy_mutex during graph recording. */
void graph_sched_append_captured_task(graph_sched_data *data, struct starpu_task *task);
/** Append explicit invalidate_submit edge during graph recording. */
void graph_sched_append_capture_explicit_invalidate(graph_sched_data *data, starpu_data_handle_t handle);
} /* namespace graph_sgoc_bundle */

void graph_sched_sgoc_pre_exec_hook(graph_sched_data *data, struct starpu_task *task);
void graph_sched_sgoc_post_exec_hook(graph_sched_data *data, struct starpu_task *task, unsigned gpu_mem_node);
void graph_sched_sgoc_pop_prefetch_hook(graph_sched_data *data, struct starpu_task *task);

/** graph_sgoc_{dag,parse,topo,mm_plan,checkpoint_wrr,checkpoint,capture_linearize,flush}.cpp — SGOC algorithms (see README). */
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
/** Policy deinit: stderr MM plan / replay / pop data-readiness when MEM_DEBUG and/or MM_ORDER_TRACE (no StarPU wait). */
void graph_sched_sgoc_print_memory_observations(graph_sched_data *data);
void graph_sched_account_outermost_capture_end(graph_sched_data *data);

/** Policy init: resolve STARPU_GRAPH_SCHED_WORKER into graph_pinned_worker_id and log target. */
void graph_sched_init_pinned_worker(graph_sched_data *data);

/** Register S handles to offload after \p task completes; RAM replicate is deferred (see graph_offload_defer_ram_w_acquire). */
void graph_sched_register_offload_after_task(graph_sched_data *data, struct starpu_task *task,
                                             const std::vector<void *> &s_offload_keys);
void graph_sched_register_evict_gpu_only_after_task(graph_sched_data *data, struct starpu_task *task,
                                                    const std::vector<void *> &handles);
void graph_sched_run_post_exec_offloads(graph_sched_data *data, struct starpu_task *task, unsigned gpu_mem_node);
void graph_sched_run_post_exec_evict_gpu_only(graph_sched_data *data, struct starpu_task *task, unsigned gpu_mem_node);
void graph_sched_drain_deferred_ram_offload_copies(graph_sched_data *data, unsigned gpu_mem_node);
void graph_sched_drain_pending_gpu_evicts(graph_sched_data *data, unsigned gpu_mem_node);
void graph_sched_clear_offload_task_registrations(graph_sched_data *data);

/** Outer sched_ctx iteration slot 0 for this task (batch / epoch index), or -1 if unavailable. */
long graph_sched_task_outer_iteration(struct starpu_task *task);
