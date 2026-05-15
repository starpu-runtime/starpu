/* SGOC graph scheduler: declarations shared across translation units (types in graph_sgoc_types.hpp). */

#pragma once

#include "graph_sgoc_types.hpp"

/** Policy data for sched_ctx when the active policy is sgoc. */
graph_sgoc_data *graph_sgoc_policy_data(unsigned sched_ctx_id);

namespace graph_sgoc_bundle {
/** Non-zero if STARPU_GRAPH_SCHED_SGOC_MEM_DEBUG enables MM offload/prefetch advance logging. */
int graph_sgoc_mem_debug_env(void);
/** Non-zero if STARPU_GRAPH_SCHED_MM_ORDER_TRACE enables per-flush MM plan vs replay ordering summary on stderr. */
int graph_sgoc_mm_order_trace_env(void);
/** Log planned topo-slot advance for prefetches/offloads (\p topo_slots = full replay topo length; \p mm lists). */
void graph_sgoc_log_mm_plan_advance_debug(size_t topo_slots, const graph_sgoc_gpu_memory_manager &mm);
/** WRR checkpoints (STARPU_GRAPH_SCHED_CHECKPOINT_MAX); mutates ops/HA then rebuilds lists + op DAG. */
void graph_sgoc_apply_wrr_checkpoints_before_topo(graph_sgoc_data *data, const graph_sgoc_captured_handle_groups &parsed,
                                                  int vb);
/** Rebuild per-handle access lists and predecessor/successor DAG from \p graph_handle_accesses (post-linearize). */
void graph_sgoc_rebuild_lists_and_refresh_deps(graph_sgoc_data *data);
/** Parse + flush captured graph after outermost recording_end (calls \c graph_sgoc_release_outermost_capture). */
void graph_sgoc_finalize_outermost_capture(graph_sgoc_data *data, std::vector<GraphOp> &&replay,
                                           std::vector<GraphHandleAccess> &&replay_ha,
                                           unsigned added_invalidate_submit, unsigned sched_ctx_id);
/** Linked-list capture → dense \p graph_ops (recording_end / deinit). */
void graph_sgoc_linearize_capture_to_ops(graph_sgoc_data *data);
/** Append one captured task under policy_mutex during graph recording. */
void graph_sgoc_append_captured_task(graph_sgoc_data *data, struct starpu_task *task);
/** Append explicit invalidate_submit edge during graph recording. */
void graph_sgoc_append_capture_explicit_invalidate(graph_sgoc_data *data, starpu_data_handle_t handle);
} /* namespace graph_sgoc_bundle */

void graph_sgoc_pre_exec_hook(graph_sgoc_data *data, struct starpu_task *task);
void graph_sgoc_post_exec_hook(graph_sgoc_data *data, struct starpu_task *task, unsigned gpu_mem_node);
void graph_sgoc_pop_prefetch_hook(graph_sgoc_data *data, struct starpu_task *task);

/** graph_sgoc_{dag,parse,topo,mm_plan,checkpoint_wrr,checkpoint,capture_linearize,flush}.cpp — SGOC algorithms (see README). */
void graph_sgoc_release_outermost_capture(graph_sgoc_data *data, std::vector<GraphOp> replay,
                                                std::vector<GraphHandleAccess> replay_ha,
                                                graph_sgoc_captured_handle_groups &parsed, bool has_batch,
                                                std::uint32_t batch_val, int vb, unsigned sched_ctx_id);

void graph_sgoc_clear_runtime(graph_sgoc_data *data);

/** Belady victim selector (starpu_data_register_victim_selector); no-op when built with STARPUSGOC_HAS_VICTIM_SELECTOR=0. */
void graph_sgoc_victim_policy_init(graph_sgoc_data *data);
void graph_sgoc_victim_policy_deinit(graph_sgoc_data *data);
void graph_sgoc_victim_rebuild_belady(graph_sgoc_data *data, const std::vector<size_t> &topo_order);
void graph_sgoc_victim_note_task_completed(graph_sgoc_data *data, struct starpu_task *task);
void graph_sgoc_victim_clear_belady(graph_sgoc_data *data);
void graph_sgoc_register(graph_sgoc_data *data);
void graph_sgoc_deinit(graph_sgoc_data *data, unsigned sched_ctx_id);
/** Policy deinit: stderr MM plan / replay / pop data-readiness when MEM_DEBUG and/or MM_ORDER_TRACE (no StarPU wait). */
void graph_sgoc_print_memory_observations(graph_sgoc_data *data);
void graph_sgoc_account_outermost_capture_end(graph_sgoc_data *data);

/** Policy init: resolve STARPU_GRAPH_SCHED_WORKER into graph_pinned_worker_id and log target. */
void graph_sgoc_init_pinned_worker(graph_sgoc_data *data);

void graph_sgoc_register_ram_offload_for_followup_pop(graph_sgoc_data *data, struct starpu_task *followup_task,
                                                       struct starpu_task *producer_task,
                                                       const std::vector<void *> &s_offload_keys);
void graph_sgoc_register_evict_gpu_only_for_followup_pop(graph_sgoc_data *data, struct starpu_task *followup_task,
                                                          const std::vector<void *> &handles);
void graph_sgoc_register_ram_offload_flush_tail(graph_sgoc_data *data, struct starpu_task *producer_task,
                                                const std::vector<void *> &s_offload_keys);
void graph_sgoc_register_evict_gpu_flush_tail(graph_sgoc_data *data, const std::vector<void *> &handles);
void graph_sgoc_run_followup_pop_offloads(graph_sgoc_data *data, struct starpu_task *picked_task, unsigned gpu_mem_node);
void graph_sgoc_run_flush_tail_offloads(graph_sgoc_data *data, unsigned gpu_mem_node);
void graph_sgoc_drain_pending_gpu_evicts(graph_sgoc_data *data, unsigned gpu_mem_node);
void graph_sgoc_clear_offload_task_registrations(graph_sgoc_data *data);

/** Outer sched_ctx iteration slot 0 for this task (batch / epoch index), or -1 if unavailable. */
long graph_sgoc_task_outer_iteration(struct starpu_task *task);
