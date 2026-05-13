/* SGOC graph scheduler — loadable StarPU policy (libgraph_sgoc_sched.so).
 *
 * Recording uses StarPU's starpu_graph_recorder hooks; starpu_graph_sched_graph_recording_begin / end
 * defer task_insert / invalidate_submit while a session is open, then the policy flushes and replays.
 *
 * **Quiescence (outermost recording_end):** while graph_record_nested is still held, the implementation unlocks the
 * policy mutex, calls starpu_task_wait_for_all(), then re-locks before linearizing linked-list capture into graph_ops
 * and moving the capture out for parsing. Do not call recording_end from a StarPU context where waiting could
 * deadlock. graph_sgoc_finalize_outermost_capture then parses the graph (no second wait). Application code may still
 * call starpu_task_wait_for_all() after recording_end to wait for flush replay submissions to complete.
 *
 * Load with STARPU_SCHED=sgoc and STARPU_SCHED_LIB pointing at libgraph_sgoc_sched.so (see new_sched/Makefile run).
 *
 * Policy init reads STARPU_GRAPH_SCHED_WORKER as cuda:num only (case-insensitive), num = CUDA device id
 * (not a global worker index). Resolution: starpu_worker_get_by_devid, then starpu_worker_get_by_type.
 * The value is trimmed (whitespace / CR / LF). If unset, default is CUDA:0. CPU workers are not supported
 * for the pinned execution path. Recording may include non-CUDA tasks; flush replay pins only when the codelet
 * can run on the CUDA pin worker.
 *
 * Design stance: SGOC is the single graph scheduler; batch/minibatch incremental replay lives only in the in-tree
 * graph_recorder.cpp reference. Each SGOC flush is self-contained (GPU MM Belady plan + mem_offload_plan cache are
 * refreshed every outer recording_end for the sgoc policy).
 *
 * Optional: STARPU_GRAPH_SCHED_SGOC_READYVRAM_TOPO (default on, non-zero) — ready-set topological order that
 * greedily minimizes simulated GPU pure-write footprint; set to 0 to use legacy lex + greedy-memory topo.
 *
 * Optional: STARPU_GRAPH_SCHED_SGOC_BUDGET_BYTES — if set to a non-negative integer, overrides the planner GPU memory
 * budget in bytes (after STARPU_GRAPH_SCHED_STARPU_MEM_AVAILABLE_FRACTION and STARPU_GRAPH_SCHED_MEM_BUDGET_FRACTION
 * would otherwise scale the pinned allowance).
 *
 * Optional: STARPU_GRAPH_SCHED_SGOC_MEM_DEBUG — when non-empty and non-zero, stderr logs per-flush MM plan advance
 * (mean topo-slot lead for planned MM prefetches/offloads vs consumer) and replay counters (RAM offload / GPU fetch / evict).
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
 * RAM copy is valid — offload is not treated as instantaneous). pop_task and pre_exec drain the same queue before
 * scheduling fetches. A GPU starpu_data_fetch_on_node is started only when starpu_memory_get_available reports enough
 * free bytes on the pinned CUDA node (if known) and the planner mem_budget headroom allows it. With StarPU new enough
 * (STARPUSGOC_HAS_VICTIM_SELECTOR=1), a Belady victim selector still handles implicit allocation pressure; explicit
 * offload evictions use the pending queue above.
 *
 * Optional: STARPU_GRAPH_SCHED_CAPTURE_TIMING — when non-empty and non-zero, stderr lines `sgoc_capture_timing:` with
 * per-phase +delta ms for recording_end / deinit_flush / linearize / finalize / flush (see graph_sgoc.cpp).
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
