/* graph_standalone scheduler: optional checkpointing and graph-aware invalidation.
 *
 * --- User-provided graphs (checkpoint + invalidation) ---
 *
 * Automatic checkpointing looks for per-handle sequences W→R→R (write then two pure reads).
 * To avoid crashes and silent wrong rematerialization when you enable
 * STARPU_GRAPH_SCHED_CHECKPOINT_COUNT (or insert checkpoints manually later):
 *
 *   1. Writer rematerialization: the checkpoint task reuses the writer’s codelet and copies
 *      every buffer handle from the original task, replacing only the written slot with the
 *      checkpointed handle. Input handles must still be valid when _ckp runs (same dependencies
 *      as the original writer). Your kernel must tolerate re-execution after R1.
 *      Each _ckp task sets starpu_task::sched_data to the original writer task pointer (scheduler
 *      field; StarPU does not free the pointed-to task).
 *
 *   2. Writer shape for *automatic* selection: the producing task must have exactly one
 *      pure STARPU_W buffer and every other buffer pure STARPU_R (no STARPU_RW / redux / etc.).
 *      Check with starpu_graph_sched_task_ok_for_checkpoint(task) once the task exists (e.g. after
 *      you build it with starpu_task_create / insert).
 *
 *   3. Eligibility audit: after your tasks have been submitted to this policy (submit_hook has
 *      registered them), call starpu_graph_sched_get_checkpoint_eligibility() to compare
 *      how many W→R→R chains exist vs how many satisfy rule (2). Chains that fail (2) are
 *      skipped by automatic checkpointing. Init writers (codelet name "cl_init") are never
 *      checkpoint candidates. A writer may pure-read another handle that already has an automatic
 *      _ckp (e.g. add_f reads hc/he); that writer’s own chain can still be checkpointed — StarPU
 *      data dependencies and per-handle _ckp ordering must keep inputs valid when its _ckp runs.
 *
 *   4. Invalidation: post_exec may submit starpu_data_invalidate_submit_no_sequential_consistency
 *      on a handle when the *next* access on that handle (in submission order on the graph) is a
 *      pure overwrite. Splicing _ckp after R1 rematerializes before later reads; the scheduler
 *      wires StarPU and graph dependencies so every reader of that handle after R1 in the chain
 *      (not only the nominal R2) depends on _ckp — e.g. add_f reading hc after read_c_1 must wait
 *      for _ckp, not only add_c.
 *
 * Verbose scheduler stderr (STARPU_GRAPH_SCHED_VERBOSE, default 0): 0 = none; 1 = init/deinit;
 * 2 = push/pop and _ckp checkpoint hook lines + checkpoint discovery/remat detail; 3 = submit hook
 * (non-_ckp) + do_schedule one line (total_tasks, ready_tasks); 4 = full do_schedule (remat + task list).
 * Values above 4 are treated as 4.
 *
 * Task names: if you omit STARPU_NAME / starpu_task::name, submit_hook sets name to the codelet’s
 * name when present, so the graph and traces see a stable label (e.g. cl_add3).
 *
 * On policy deinit, stderr logs cumulative stats: tasks that ran through post_exec_hook, automatic
 * _ckp tasks successfully submitted, and handles passed to invalidate after post_exec (graph rule).
 */

#ifndef GRAPH_SCHED_H
#define GRAPH_SCHED_H

#include <starpu.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Set how many checkpointable tasks (W in W→R→R chains) to checkpoint automatically, in order of
 * descending rematerialization throughput (perf model on STARPU_GRAPH_SCHED_WORKER_ID). 0 = none.
 * After submitting tasks, call starpu_graph_sched_apply_auto_checkpoints() to discover candidates,
 * print info, and insert up to this many _ckp tasks (runs under the policy mutex).
 * Alternatively, set env STARPU_GRAPH_SCHED_CHECKPOINT_COUNT before init (read on first use). */
void starpu_graph_sched_set_checkpoint_count(unsigned n);

/* Under the same graph policy mutex as submit/push/pop/do_schedule: report checkpointable writers
 * (stderr), then insert up to starpu_graph_sched_set_checkpoint_count automatic _ckp tasks.
 * sched_ctx_id 0 = current context. Call after submit_hook has registered the DAG. */
void starpu_graph_sched_apply_auto_checkpoints(unsigned sched_ctx_id);

/* Non-zero iff the task’s buffers match automatic checkpoint rematerialization rules:
 * exactly one pure STARPU_W and all other buffers pure STARPU_R. */
int starpu_graph_sched_task_ok_for_checkpoint(struct starpu_task *task);

/* After tasks are visible to the graph policy (submit_hook):
 * *out_wrr_chains — W→R→R checkpoint *candidates* (handles not yet checkpointed in this ctx), after
 *   dropping writers with no positive StarPU expected duration on STARPU_GRAPH_SCHED_WORKER_ID
 *   (same rule as _ckp rematerialization timing).
 * *out_auto_compatible — how many of those have a writer satisfying task_ok_for_checkpoint.
 * Either output pointer may be NULL if not needed. sched_ctx_id 0 = current context. */
void starpu_graph_sched_get_checkpoint_eligibility(unsigned sched_ctx_id,
    unsigned *out_wrr_chains,
    unsigned *out_auto_compatible);

#ifdef __cplusplus
}
#endif

#endif /* GRAPH_SCHED_H */
