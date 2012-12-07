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

/* generic structure used by the scheduling contexts to iterate the workers */
struct starpu_sched_ctx_worker_collection
{
	/* hidden data structure used to memorize the workers */
	void *workerids;
	/* the number of workers in the collection */
	unsigned nworkers;
	/* the current cursor of the collection*/
	pthread_key_t cursor_key;
	/* the type of structure (WORKER_LIST,...) */
	int type;
	/* checks if there is another element in collection */
	unsigned (*has_next)(struct starpu_sched_ctx_worker_collection *workers);
	/* return the next element in the collection */
	int (*get_next)(struct starpu_sched_ctx_worker_collection *workers);
	/* add a new element in the collection */
	int (*add)(struct starpu_sched_ctx_worker_collection *workers, int worker);
	/* remove an element from the collection */
	int (*remove)(struct starpu_sched_ctx_worker_collection *workers, int worker);
	/* initialize the structure */
	void* (*init)(struct starpu_sched_ctx_worker_collection *workers);
	/* free the structure */
	void (*deinit)(struct starpu_sched_ctx_worker_collection *workers);
	/* initialize the cursor if there is one */
	void (*init_cursor)(struct starpu_sched_ctx_worker_collection *workers);
	/* free the cursor if there is one */
	void (*deinit_cursor)(struct starpu_sched_ctx_worker_collection *workers);
};

/* types of structures the worker collection can implement */
#define WORKER_LIST 0

struct starpu_performance_counters
{
	void (*notify_idle_cycle)(unsigned sched_ctx, int worker, double idle_time);
	void (*notify_idle_end)(unsigned sched_ctx, int worker);
	void (*notify_pushed_task)(unsigned sched_ctx, int worker);
	void (*notify_poped_task)(unsigned sched_ctx, int worker, double flops);
	void (*notify_post_exec_hook)(unsigned sched_ctx, int taskid);
	void (*notify_submitted_job)(struct starpu_task *task, uint32_t footprint);
};

#ifdef STARPU_USE_SCHED_CTX_HYPERVISOR
void starpu_set_perf_counters(unsigned sched_ctx_id, struct starpu_performance_counters *perf_counters);
void starpu_call_poped_task_cb(int workerid, unsigned sched_ctx_id, double flops);
void starpu_call_pushed_task_cb(int workerid, unsigned sched_ctx_id);
#endif //STARPU_USE_SCHED_CTX_HYPERVISOR

unsigned starpu_create_sched_ctx(const char *policy_name, int *workerids_ctx, int nworkers_ctx, const char *sched_name);

unsigned starpu_create_sched_ctx_inside_interval(const char *policy_name, const char *sched_name,
						 int min_ncpus, int max_ncpus, int min_ngpus, int max_ngpus,
						 unsigned allow_overlap);

void starpu_delete_sched_ctx(unsigned sched_ctx_id, unsigned inheritor_sched_ctx_id);

void starpu_add_workers_to_sched_ctx(int *workerids_ctx, int nworkers_ctx, unsigned sched_ctx);

void starpu_remove_workers_from_sched_ctx(int *workerids_ctx, int nworkers_ctx, unsigned sched_ctx);

void starpu_set_sched_ctx_policy_data(unsigned sched_ctx, void* policy_data);

void* starpu_get_sched_ctx_policy_data(unsigned sched_ctx);

void starpu_worker_set_sched_condition(unsigned sched_ctx, int workerid, pthread_mutex_t *sched_mutex, pthread_cond_t *sched_cond);

void starpu_worker_get_sched_condition(unsigned sched_ctx, int workerid, pthread_mutex_t **sched_mutex, pthread_cond_t **sched_cond);

void starpu_worker_init_sched_condition(unsigned sched_ctx, int workerid);

void starpu_worker_deinit_sched_condition(unsigned sched_ctx, int workerid);

struct starpu_sched_ctx_worker_collection* starpu_create_worker_collection_for_sched_ctx(unsigned sched_ctx_id, int type);

void starpu_delete_worker_collection_for_sched_ctx(unsigned sched_ctx_id);

struct starpu_sched_ctx_worker_collection* starpu_get_worker_collection_of_sched_ctx(unsigned sched_ctx_id);

pthread_mutex_t* starpu_get_changing_ctx_mutex(unsigned sched_ctx_id);

void starpu_set_sched_ctx(unsigned *sched_ctx);

unsigned starpu_get_sched_ctx(void);

void starpu_notify_hypervisor_exists(void);

unsigned starpu_check_if_hypervisor_exists(void);

unsigned starpu_get_nworkers_of_sched_ctx(unsigned sched_ctx);

unsigned starpu_get_nshared_workers(unsigned sched_ctx_id, unsigned sched_ctx_id2);

unsigned starpu_worker_belongs_to_sched_ctx(int workerid, unsigned sched_ctx_id);

unsigned starpu_are_overlapping_ctxs_on_worker(int workerid);

unsigned starpu_is_ctxs_turn(int workerid, unsigned sched_ctx_id);

void starpu_set_turn_to_other_ctx(int workerid, unsigned sched_ctx_id);

double starpu_get_max_time_worker_on_ctx(void);

void starpu_stop_task_submission(void);

void starpu_sched_ctx_set_inheritor(unsigned sched_ctx, unsigned inheritor);

void starpu_sched_ctx_finished_submit(unsigned sched_ctx_id);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHED_CTX_H__ */
