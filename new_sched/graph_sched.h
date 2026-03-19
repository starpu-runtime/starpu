/* Optional checkpointing config for graph_standalone scheduler.
 * Checkpointing is OFF by default. Set count or env STARPU_GRAPH_SCHED_CHECKPOINT_COUNT.
 */

#ifndef GRAPH_SCHED_H
#define GRAPH_SCHED_H

#include <starpu.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Set how many checkpointable tasks (W in W->R->R chains) to randomly checkpoint.
 * Call before starpu_resume() or before submitting tasks. 0 = none.
 * Alternatively, set env STARPU_GRAPH_SCHED_CHECKPOINT_COUNT. */
void starpu_graph_sched_set_checkpoint_count(unsigned n);

#ifdef __cplusplus
}
#endif

#endif /* GRAPH_SCHED_H */
