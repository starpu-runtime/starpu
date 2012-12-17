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

#ifdef STARPU_DEVEL
#  warning rename all objects to start with starpu_sched_ctx
#endif

//struct starpu_iterator;
struct starpu_iterator
{
	int cursor;
};


/* generic structure used by the scheduling contexts to iterate the workers */
struct starpu_sched_ctx_worker_collection
{
	/* hidden data structure used to memorize the workers */
	void *workerids;
	/* the number of workers in the collection */
	unsigned nworkers;
	/* the type of structure (WORKER_LIST,...) */
	int type;
	/* checks if there is another element in collection */
	unsigned (*has_next)(struct starpu_sched_ctx_worker_collection *workers, struct starpu_iterator *it);
	/* return the next element in the collection */
	int (*get_next)(struct starpu_sched_ctx_worker_collection *workers, struct starpu_iterator *it);
	/* add a new element in the collection */
	int (*add)(struct starpu_sched_ctx_worker_collection *workers, int worker);
	/* remove an element from the collection */
	int (*remove)(struct starpu_sched_ctx_worker_collection *workers, int worker);
	/* initialize the structure */
	void* (*init)(struct starpu_sched_ctx_worker_collection *workers);
	/* free the structure */
	void (*deinit)(struct starpu_sched_ctx_worker_collection *workers);
	/* initialize the cursor if there is one */
	void (*init_iterator)(struct starpu_sched_ctx_worker_collection *workers, struct starpu_iterator *it);
};

/* types of structures the worker collection can implement */
#define WORKER_LIST 0

struct starpu_performance_counters
{
	void (*notify_idle_cycle)(unsigned sched_ctx_id, int worker, double idle_time);
	void (*notify_idle_end)(unsigned sched_ctx_id, int worker);
	void (*notify_pushed_task)(unsigned sched_ctx_id, int worker);
	void (*notify_poped_task)(unsigned sched_ctx_id, int worker, double flops);
	void (*notify_post_exec_hook)(unsigned sched_ctx_id, int taskid);
	void (*notify_submitted_job)(struct starpu_task *task, uint32_t footprint);
};

#ifdef STARPU_USE_SCHED_CTX_HYPERVISOR
void starpu_set_perf_counters(unsigned sched_ctx_id, struct starpu_performance_counters *perf_counters);
void starpu_call_poped_task_cb(int workerid, unsigned sched_ctx_id, double flops);
void starpu_call_pushed_task_cb(int workerid, unsigned sched_ctx_id);
#endif //STARPU_USE_SCHED_CTX_HYPERVISOR

unsigned starpu_sched_ctx_create(const char *policy_name, int *workerids_ctx, int nworkers_ctx, const char *sched_ctx_name);

unsigned starpu_sched_ctx_create_inside_interval(const char *policy_name, const char *sched_name,
						 int min_ncpus, int max_ncpus, int min_ngpus, int max_ngpus,
						 unsigned allow_overlap);

void starpu_sched_ctx_delete(unsigned sched_ctx_id);

void starpu_sched_ctx_add_workers(int *workerids_ctx, int nworkers_ctx, unsigned sched_ctx_id);

void starpu_sched_ctx_remove_workers(int *workerids_ctx, int nworkers_ctx, unsigned sched_ctx_id);

void starpu_sched_ctx_set_policy_data(unsigned sched_ctx_id, void *policy_data);

void* starpu_sched_ctx_get_policy_data(unsigned sched_ctx_id);

/* When there is no available task for a worker, StarPU blocks this worker on a
condition variable. This function specifies which condition variable (and the
associated mutex) should be used to block (and to wake up) a worker. Note that
multiple workers may use the same condition variable. For instance, in the case
of a scheduling strategy with a single task queue, the same condition variable
would be used to block and wake up all workers.  The initialization method of a
scheduling strategy (init_sched) must call this function once per worker. */
#if !defined(_MSC_VER) && !defined(STARPU_SIMGRID)
#ifdef STARPU_DEVEL
#warning do we really need both starpu_sched_ctx_set_worker_mutex_and_cond and starpu_sched_ctx_init_worker_mutex_and_cond functions
#endif

void starpu_sched_ctx_set_worker_mutex_and_cond(unsigned sched_ctx_id, int workerid, pthread_mutex_t *sched_mutex, pthread_cond_t *sched_cond);

void starpu_sched_ctx_get_worker_mutex_and_cond(unsigned sched_ctx_id, int workerid, pthread_mutex_t **sched_mutex, pthread_cond_t **sched_cond);
#endif

void starpu_sched_ctx_init_worker_mutex_and_cond(unsigned sched_ctx_id, int workerid);

void starpu_sched_ctx_deinit_worker_mutex_and_cond(unsigned sched_ctx_id, int workerid);

struct starpu_sched_ctx_worker_collection* starpu_sched_ctx_create_worker_collection(unsigned sched_ctx_id, int type);

void starpu_sched_ctx_delete_worker_collection(unsigned sched_ctx_id);

struct starpu_sched_ctx_worker_collection* starpu_sched_ctx_get_worker_collection(unsigned sched_ctx_id);

#if !defined(_MSC_VER) && !defined(STARPU_SIMGRID)
pthread_mutex_t* starpu_get_changing_ctx_mutex(unsigned sched_ctx_id);
#endif

void starpu_task_set_context(unsigned *sched_ctx_id);

unsigned starpu_task_get_context(void);

void starpu_notify_hypervisor_exists(void);

unsigned starpu_check_if_hypervisor_exists(void);

unsigned starpu_sched_ctx_get_nworkers(unsigned sched_ctx_id);

unsigned starpu_sched_ctx_get_nshared_workers(unsigned sched_ctx_id, unsigned sched_ctx_id2);

unsigned starpu_sched_ctx_contains_worker(int workerid, unsigned sched_ctx_id);

unsigned starpu_sched_ctx_overlapping_ctxs_on_worker(int workerid);

unsigned starpu_is_ctxs_turn(int workerid, unsigned sched_ctx_id);

void starpu_set_turn_to_other_ctx(int workerid, unsigned sched_ctx_id);

double starpu_get_max_time_worker_on_ctx(void);

void starpu_stop_task_submission(void);

void starpu_sched_ctx_set_inheritor(unsigned sched_ctx_id, unsigned inheritor);

void starpu_sched_ctx_finished_submit(unsigned sched_ctx_id);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHED_CTX_H__ */
