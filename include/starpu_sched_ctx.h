/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010 - 2012  INRIA
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __STARPU_SCHED_CTX_H__
#define __STARPU_SCHED_CTX_H__

#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

/*
 * MANAGEMENT OF SCHEDULING CONTEXTS
 */

/* create a context indicating the scheduling policy, the workers it should have and a potential name */
unsigned starpu_sched_ctx_create(const char *policy_name, int *workerids_ctx, int nworkers_ctx, const char *sched_ctx_name);

/* create a context indicating an approximate interval of resources */
unsigned starpu_sched_ctx_create_inside_interval(const char *policy_name, const char *sched_name,
						 int min_ncpus, int max_ncpus, int min_ngpus, int max_ngpus,
						 unsigned allow_overlap);

/* add workers to a context */
void starpu_sched_ctx_add_workers(int *workerids_ctx, int nworkers_ctx, unsigned sched_ctx_id);

/* remove workers from a context */
void starpu_sched_ctx_remove_workers(int *workerids_ctx, int nworkers_ctx, unsigned sched_ctx_id);

/* delete a certain context */
void starpu_sched_ctx_delete(unsigned sched_ctx_id);

/* indicate which context whill inherit the resources of this context when he will be deleted */
void starpu_sched_ctx_set_inheritor(unsigned sched_ctx_id, unsigned inheritor);

/* indicate that the current thread is submitting only to the current context */
void starpu_sched_ctx_set_context(unsigned *sched_ctx_id);

/* find out to which context is submitting the current thread */
unsigned starpu_sched_ctx_get_context(void);

/* stop submitting tasks from the empty context list until the next time the context has
   time to check the empty context list*/
void starpu_sched_ctx_stop_task_submission(void);

/* indicate starpu that hte application finished submitting to this context in order to
   move the workers to the inheritor as soon as possible */
void starpu_sched_ctx_finished_submit(unsigned sched_ctx_id);


/*
 * CONNECTION WITH THE HYPERVISOR
 */

/* performance counters used by the starpu to indicate the hypervisor 
   how the application and the resources are executing */
struct starpu_sched_ctx_performance_counters
{
	/* tell the hypervisor for how long a worker was idle in a certain context */ 
	void (*notify_idle_cycle)(unsigned sched_ctx_id, int worker, double idle_time);
	/* tell the hypervisor when a worker stoped being idle in a certain context */ 
	void (*notify_idle_end)(unsigned sched_ctx_id, int worker);
	/* tell the hypervisor when a task was pushed on a worker in a certain context */ 
	void (*notify_pushed_task)(unsigned sched_ctx_id, int worker);
	/* tell the hypervisor when a task was poped from a worker in a certain context */ 
	void (*notify_poped_task)(unsigned sched_ctx_id, int worker, struct starpu_task *task, size_t data_size, uint32_t footprint);
	/* tell the hypervisor when a task finished executing in a certain context */
	void (*notify_post_exec_hook)(unsigned sched_ctx_id, int taskid);
	/* tell the hypervisor when a task was submitted to a certain context */
	void (*notify_submitted_job)(struct starpu_task *task, uint32_t footprint);
	/* tell the hypervisor when a context was deleted */
	void (*notify_delete_context)(unsigned sched_ctx);
};

#ifdef STARPU_USE_SC_HYPERVISOR
/* indicates to starpu the pointer to the performance counte */
void starpu_sched_ctx_set_perf_counters(unsigned sched_ctx_id, struct starpu_sched_ctx_performance_counters *perf_counters);
/* callback that lets the scheduling policy tell the hypervisor that a task was pushed on a worker */
void starpu_sched_ctx_call_pushed_task_cb(int workerid, unsigned sched_ctx_id);
#endif //STARPU_USE_SC_HYPERVISOR

/* allow the hypervisor to let starpu know he's initialised */
void starpu_sched_ctx_notify_hypervisor_exists(void);

/* ask starpu if he is informed if the hypervisor is initialised */
unsigned starpu_sched_ctx_check_if_hypervisor_exists(void);

/*
 * POLICY DATA 
*/
/* allow the scheduling policy to have its own data in a context, like a private list of tasks, mutexes, conds, etc. */
void starpu_sched_ctx_set_policy_data(unsigned sched_ctx_id, void *policy_data);

/* return the scheduling policy private data */
void* starpu_sched_ctx_get_policy_data(unsigned sched_ctx_id);


/*
 * WORKERS IN CONTEXT 
*/
/* create a worker collection for a context, the type can be only STARPU_WORKER_LIST for now, which corresponds to a simple list */
struct starpu_worker_collection* starpu_sched_ctx_create_worker_collection(unsigned sched_ctx_id, int type);

/* free the worker collection when removing the context */
void starpu_sched_ctx_delete_worker_collection(unsigned sched_ctx_id);

/*return the worker collection */
struct starpu_worker_collection* starpu_sched_ctx_get_worker_collection(unsigned sched_ctx_id);

/* return the number of workers in the sched_ctx's collection */
unsigned starpu_sched_ctx_get_nworkers(unsigned sched_ctx_id);

/* return the number of shared workers in the sched_ctx's collection */
unsigned starpu_sched_ctx_get_nshared_workers(unsigned sched_ctx_id, unsigned sched_ctx_id2);

/* return 1 if the worker belongs to the context and 0 otherwise */
unsigned starpu_sched_ctx_contains_worker(int workerid, unsigned sched_ctx_id);

/* check if a worker is shared between several contexts */
unsigned starpu_sched_ctx_overlapping_ctxs_on_worker(int workerid);

/* manage sharing of resources between contexts: checkOB which ctx has its turn to pop */
unsigned starpu_sched_ctx_is_ctxs_turn(int workerid, unsigned sched_ctx_id);

/* manage sharing of resources between contexts: by default a round_robin strategy
   is executed but the user can interfere to tell which ctx has its turn to pop */
void starpu_sched_ctx_set_turn_to_other_ctx(int workerid, unsigned sched_ctx_id);

/* time sharing a resources, indicate how long a worker has been active in
   the current sched_ctx */
double starpu_sched_ctx_get_max_time_worker_on_ctx(void);

/*
 *	Priorities
 */

/* get min priority for the scheduler of the global context */
int starpu_sched_get_min_priority(void);

/* get max priority for the scheduler of the global context */
int starpu_sched_get_max_priority(void);

/* set min priority for the scheduler of the global context */
int starpu_sched_set_min_priority(int min_prio);

/* set max priority for the scheduler of the global context */
int starpu_sched_set_max_priority(int max_prio);

/* get min priority for the scheduler of the scheduling context indicated */
int starpu_sched_ctx_get_min_priority(unsigned sched_ctx_id);

/* get max priority for the scheduler of the scheduling context indicated */
int starpu_sched_ctx_get_max_priority(unsigned sched_ctx_id);

/* set min priority for the scheduler of the scheduling context indicated */
int starpu_sched_ctx_set_min_priority(unsigned sched_ctx_id, int min_prio);

/* set max priority for the scheduler of the scheduling context indicated */
int starpu_sched_ctx_set_max_priority(unsigned sched_ctx_id, int max_prio);

/* Provided for legacy reasons */
#define STARPU_MIN_PRIO		(starpu_sched_get_min_priority())
#define STARPU_MAX_PRIO		(starpu_sched_get_max_priority())

/* By convention, the default priority level should be 0 so that we can
 * statically allocate tasks with a default priority. */
#define STARPU_DEFAULT_PRIO	0

/* execute any parallel code on the workers of the sched_ctx (workers are blocked) */
void* starpu_sched_ctx_exec_parallel_code(void* (*func)(void* param), void* param, unsigned sched_ctx_id);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHED_CTX_H__ */
