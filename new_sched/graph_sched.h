/* Loadable policy graph_recorder — skeleton scheduler (see libgraph_sched.cpp).
 *
 * Placeholder for a custom task graph; current implementation uses a simple FIFO of ready tasks.
 * StarPU extension hooks (starpu_graph_recorder) register with this policy for deferred
 * starpu_task_insert / starpu_data_invalidate_submit while a session is open.
 *
 * Policy init resolves the pin (type, id / devid, worker_id). If STARPU_GRAPH_SCHED_WORKER is unset, the default is
 * CUDA:0 when present, else CPU:0. If the explicit worker cannot be resolved, or both defaults are unavailable
 * (e.g. STARPU_NCPU=0), policy init exits the process with status 1.
 * While recording, each captured task is checked with starpu_worker_can_execute_task_first_impl; mismatch returns
 * EINVAL from starpu_task_insert (task destroyed).
 */

#ifndef GRAPH_SCHED_H
#define GRAPH_SCHED_H

#include <starpu.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Defer the above APIs into a queue; on end, replay in order (recording depth + policy graph_recorder).
 * sched_ctx_id 0 = current context. Nested begin/end supported. */
void starpu_graph_sched_graph_recording_begin(unsigned sched_ctx_id);
void starpu_graph_sched_graph_recording_end(unsigned sched_ctx_id);

#ifdef __cplusplus
}
#endif

#endif /* GRAPH_SCHED_H */
