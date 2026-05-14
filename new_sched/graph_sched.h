/* SGOC graph scheduler — loadable StarPU policy (libgraph_sgoc_sched.so).
 *
 * Graph capture uses StarPU's internal dispatch (starpu_graph_capture.h): starpu_graph_sched_graph_recording_begin / end
 * defer task_insert / invalidate_submit while a session is open, then the policy flushes and replays.
 *
 * **Quiescence:** On outermost recording_begin the policy mutex is released, starpu_task_wait_for_all() runs (so the
 * capture starts from a quiescent scheduler), then the mutex is re-acquired; graph_capture_wall_start is taken after
 * that wait. On outermost recording_end the same pattern runs before linearizing capture into graph_ops. Do not call
 * recording_begin/end from a StarPU context where waiting could deadlock. graph_sgoc_finalize_outermost_capture then
 * parses the graph (no further wait inside finalize). Application code may still call starpu_task_wait_for_all() after
 * recording_end to wait for flush replay submissions to complete.
 *
 * Load with STARPU_SCHED=sgoc and STARPU_SCHED_LIB pointing at libgraph_sgoc_sched.so (see new_sched/Makefile run).
 *
 * Policy init reads STARPU_GRAPH_SCHED_WORKER as cuda:num only (case-insensitive), num = CUDA device id
 * (not a global worker index). Resolution: starpu_worker_get_by_devid, then starpu_worker_get_by_type.
 * The value is trimmed (whitespace / CR / LF). If unset, default is CUDA:0. CPU workers are not supported
 * for the pinned execution path. Recording may include non-CUDA tasks; flush replay pins only when the codelet
 * can run on the CUDA pin worker.
 *
 * Design stance: each SGOC flush is self-contained (GPU MM Belady plan + mem_offload_plan cache are refreshed every
 * outer recording_end).
 *
 * Optional: STARPU_GRAPH_SCHED_CHECKPOINT_MAX (default 0 = off) — max WRR checkpoint clones per recording flush,
 * after activation classification (P/S/G/A via graph_sched_parse_captured_data_handles)
 * and before topological sort. Candidates are ordered by best rematerialization bytes per predicted microsecond on the
 * pinned CUDA worker (then shorter predicted time, then task pointer). Insertion uses the same chain rule as the reference
 * (forward pure-read in graph subiter 1, backward pure-read in subiter 2; invalidate + clone immediately before the
 * backward read). Linearize ends with a handle-list rebuild + full pred/succ refresh so dense \c op_idx matches edges
 * before checkpoint insertion; greedy VRAM topo and MM Belady then run on the post-checkpoint DAG (extra TASK ops add
 * topo appearances for rematerialized handles). With STARPU_GRAPH_SCHED_VERBOSE>=3, stderr lists WRR-ranked candidates, pool sizes, each successful
 * insert (producer op index, codelet, job id, written handle, clone task pointer, remat_bytes), and skip reasons at
 * verbose >= 2.
 *
 * Activation classification for checkpoint keys merges **all** subiter-1/2 tasks in the capture by default; a second
 * forward pure-W on the same handle (e.g. next minibatch) sets f_w>1 and removes checkpointability. Set
 * STARPU_GRAPH_SCHED_ACTIVATION_AGG_FIRST_OUTER_BATCH=1 to aggregate only tasks at the minimum \c graph_stage_batch_iteration
 * among subiter-1/2 tasks (requires a valid batch tag on **every** such task; otherwise the filter is ignored).
 *
 * **Host–device traffic:** these checkpoints are *not* “skip storing activations to save PCIe” (that would be a different
 * algorithm). Each insert adds an explicit invalidate on the activation handle and an extra cloned producer task before
 * the backward read. MM Belady and runtime victim state are driven from the **post-checkpoint** graph and topo order, but
 * offload/prefetch **counts** can still match a no-checkpoint run when pressure is similar: remat TASKs touch the same
 * handles as consumers (Belady's forbidden set per slot shrinks), and non-activation tensors may dominate evictions.
 * STARPU_BUS_STATS totals are often dominated by full training tensor traffic either way.
 *
 * Optional: STARPU_GRAPH_SCHED_SGOC_READYVRAM_TOPO (default on, non-zero) — ready-set topological order that
 * greedily minimizes simulated GPU pure-write footprint; set to 0 to use legacy lex + greedy-memory topo.
 *
 * Optional: STARPU_GRAPH_SCHED_SGOC_BUDGET_BYTES — if set to a non-negative integer, overrides the planner GPU memory
 * budget in bytes (after STARPU_GRAPH_SCHED_STARPU_MEM_AVAILABLE_FRACTION and STARPU_GRAPH_SCHED_MEM_BUDGET_FRACTION
 * would otherwise scale the pinned allowance).
 *
 * Optional: STARPU_GRAPH_SCHED_SGOC_MEM_DEBUG — when non-empty and non-zero, stderr logs at **policy deinit**
 * (scheduler teardown; no starpu_task_wait_for_all — unsafe during starpu_shutdown). MM plan advance, replay counters,
 * and pinned-worker pop_task data-readiness. Call starpu_task_wait_for_all from the application before StarPU shutdown
 * if you need counters after all replay tasks have finished.
 *
 * Optional: STARPU_GRAPH_SCHED_MM_ORDER_TRACE — when non-empty and non-zero, stderr prints at **policy deinit** one line
 * summarizing the last flush MM plan lists vs accumulated replay hook counters (same teardown caveat as MEM_DEBUG).
 *
 * Optional: STARPU_GRAPH_SCHED_OPTIMIZER_STATE_OFFLOAD (default 0) — when the captured graph includes an optimizer
 * phase (graph subiteration UINT32_MAX) and parsed optimizer-state handles (Adam m/v, etc.), emit StarPU hints before
 * the first optimizer task: replicate to main RAM and evict from the pinned GPU, then prefetch back to GPU for the
 * optimizer step. Set to 0 to disable.
 *
 * Optional: STARPU_GRAPH_SCHED_STARPU_MEM_AVAILABLE_FRACTION (SGOC default 0.6) — fraction of StarPU's CUDA memory
 * limit (starpu_memory_get_total on the worker node, or STARPU_LIMIT_CUDA*_MEM when total is unknown) used as planner
 * allowance before STARPU_GRAPH_SCHED_MEM_BUDGET_FRACTION (SGOC default 1.0). Must be in (0,1].
 *
 * Runtime: post_exec registers planned S-offloads (async RAM replicate), queues each handle for GPU eviction, and
 * immediately tries graph_sched_drain_pending_gpu_evicts (starpu_data_can_evict + starpu_data_evict_from_node when the
 * RAM copy is valid — offload is not treated as instantaneous). When the linear MM planner sees that the next graph
 * touch of a victim is \c invalidate_submit with no intervening TASK on that handle, it skips RAM offload and only
 * schedules GPU eviction (same post_exec drain path, no starpu_data_acquire to main RAM). pop_task and pre_exec drain the same queue before
 * scheduling fetches. A GPU starpu_data_fetch_on_node is started only when starpu_memory_get_available reports enough
 * free bytes on the pinned CUDA node (if known) and the planner mem_budget headroom allows it. With StarPU new enough
 * (STARPUSGOC_HAS_VICTIM_SELECTOR=1), a Belady victim selector still handles implicit allocation pressure; explicit
 * offload evictions use the pending queue above.
 *
 * Optional: STARPU_GRAPH_SCHED_CAPTURE_TIMING — when non-empty and non-zero, stderr lines `sgoc_capture_timing:` with
 * per-phase +delta ms for recording_begin (begin wait), recording_end / deinit_flush / linearize / finalize / flush
 * (see graph_sgoc.cpp). The same `sgoc_capture_timing:` phase lines are printed when STARPU_GRAPH_SCHED_VERBOSE>=2
 * without setting CAPTURE_TIMING.
 */

#ifndef GRAPH_SCHED_H
#define GRAPH_SCHED_H

#include <starpu.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Defer task_insert / invalidate_submit into a queue; on end, the sgoc policy parses and replays the graph.
 * sched_ctx_id 0 = current context. Nested begin/end supported. */
void starpu_graph_sched_graph_recording_begin(unsigned sched_ctx_id);
void starpu_graph_sched_graph_recording_end(unsigned sched_ctx_id);

#ifdef __cplusplus
}
#endif

#endif /* GRAPH_SCHED_H */
