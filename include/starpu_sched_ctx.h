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
 *	Priorities
 */
int starpu_sched_get_min_priority(void);
int starpu_sched_get_max_priority(void);

int starpu_sched_set_min_priority(int min_prio);
int starpu_sched_set_max_priority(int max_prio);

int starpu_sched_ctx_get_min_priority(unsigned sched_ctx_id);
int starpu_sched_ctx_get_max_priority(unsigned sched_ctx_id);

int starpu_sched_ctx_set_min_priority(unsigned sched_ctx_id, int min_prio);
int starpu_sched_ctx_set_max_priority(unsigned sched_ctx_id, int max_prio);

/* Provided for legacy reasons */
#define STARPU_MIN_PRIO		(starpu_sched_get_min_priority())
#define STARPU_MAX_PRIO		(starpu_sched_get_max_priority())

/* By convention, the default priority level should be 0 so that we can
 * statically allocate tasks with a default priority. */
#define STARPU_DEFAULT_PRIO	0


struct starpu_sched_ctx_performance_counters
{
	void (*notify_idle_cycle)(unsigned sched_ctx_id, int worker, double idle_time);
	void (*notify_idle_end)(unsigned sched_ctx_id, int worker);
	void (*notify_pushed_task)(unsigned sched_ctx_id, int worker);
	void (*notify_poped_task)(unsigned sched_ctx_id, int worker, struct starpu_task *task, size_t data_size, uint32_t footprint);
	void (*notify_post_exec_hook)(unsigned sched_ctx_id, int taskid);
	void (*notify_submitted_job)(struct starpu_task *task, uint32_t footprint);
	void (*notify_delete_context)(unsigned sched_ctx);
};

#ifdef STARPU_USE_SCHED_CTX_HYPERVISOR
void starpu_sched_ctx_set_perf_counters(unsigned sched_ctx_id, struct starpu_sched_ctx_performance_counters *perf_counters);
void starpu_sched_ctx_call_pushed_task_cb(int workerid, unsigned sched_ctx_id);
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

struct starpu_worker_collection* starpu_sched_ctx_create_worker_collection(unsigned sched_ctx_id, int type);

void starpu_sched_ctx_delete_worker_collection(unsigned sched_ctx_id);

struct starpu_worker_collection* starpu_sched_ctx_get_worker_collection(unsigned sched_ctx_id);

starpu_pthread_mutex_t* starpu_sched_ctx_get_changing_ctx_mutex(unsigned sched_ctx_id);

void starpu_sched_ctx_set_context(unsigned *sched_ctx_id);

unsigned starpu_sched_ctx_get_context(void);

void starpu_sched_ctx_notify_hypervisor_exists(void);

unsigned starpu_sched_ctx_check_if_hypervisor_exists(void);

unsigned starpu_sched_ctx_get_nworkers(unsigned sched_ctx_id);

unsigned starpu_sched_ctx_get_nshared_workers(unsigned sched_ctx_id, unsigned sched_ctx_id2);

unsigned starpu_sched_ctx_contains_worker(int workerid, unsigned sched_ctx_id);

unsigned starpu_sched_ctx_overlapping_ctxs_on_worker(int workerid);

unsigned starpu_sched_ctx_is_ctxs_turn(int workerid, unsigned sched_ctx_id);

void starpu_sched_ctx_set_turn_to_other_ctx(int workerid, unsigned sched_ctx_id);

double starpu_sched_ctx_get_max_time_worker_on_ctx(void);

void starpu_sched_ctx_stop_task_submission(void);

void starpu_sched_ctx_set_inheritor(unsigned sched_ctx_id, unsigned inheritor);

void starpu_sched_ctx_finished_submit(unsigned sched_ctx_id);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_SCHED_CTX_H__ */
