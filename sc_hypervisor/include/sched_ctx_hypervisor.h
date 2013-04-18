/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011 - 2013  INRIA
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

#ifndef SCHED_CTX_HYPERVISOR_H
#define SCHED_CTX_HYPERVISOR_H

#include <starpu.h>
#include <sched_ctx_hypervisor_config.h>
#include <sched_ctx_hypervisor_monitoring.h>

#ifdef __cplusplus
extern "C"
{
#endif

/* synchronise the hypervisor when several workers try to update its information */
starpu_pthread_mutex_t act_hypervisor_mutex;


/* Forward declaration of an internal data structure
 * FIXME: Remove when no longer exposed.  */
/* the resizing is not done instantly, a request is kept and executed 
   when available */
struct resize_request_entry;

/* platform of resizing contexts */
struct sched_ctx_hypervisor_policy
{
	/* name of the strategy */
	const char* name;

	/* indicate if it is a policiy create by the user or not */
	unsigned custom;

	/* if knwing the future the hypervisor can find the good 
	   distribution of workers on contexts even at the begining of the program */
	void (*size_ctxs)(int *sched_ctxs, int nsched_ctxs , int *workers, int nworkers);

	/* the hypervisor takes a decision when the worker was idle for another cyle in this ctx */
	void (*handle_idle_cycle)(unsigned sched_ctx, int worker);

	/* the hypervisor takes a decision when another task was pushed on this worker in this ctx */
	void (*handle_pushed_task)(unsigned sched_ctx, int worker);

	/* the hypervisor takes a decision when another task was poped from this worker in this ctx */
	void (*handle_poped_task)(unsigned sched_ctx, int worker,struct starpu_task *task, uint32_t footprint);

	/* the hypervisor takes a decision when the worker stoped being idle in this ctx */
	void (*handle_idle_end)(unsigned sched_ctx, int worker);

	/* the hypervisor takes a decision when a certain task finished executing in this ctx */
	void (*handle_post_exec_hook)(unsigned sched_ctx, int task_tag);

	/* the hypervisor takes a decision when a job was submitted in this ctx */
	void (*handle_submitted_job)(struct starpu_codelet *cl, unsigned sched_ctx, uint32_t footprint);
	
	/* the hypervisor takes a decision when a certain ctx was deleted */
	void (*end_ctx)(unsigned sched_ctx);
};

/* start the hypervisor indicating the resizing policy to user */
struct starpu_sched_ctx_performance_counters *sched_ctx_hypervisor_init(struct sched_ctx_hypervisor_policy *policy);

/* shutdown the hypervisor */
void sched_ctx_hypervisor_shutdown(void);

/* only registered contexts are resized by the hypervisor */
void sched_ctx_hypervisor_register_ctx(unsigned sched_ctx, double total_flops);

/* remove a worker from the hypervisor's list */
void sched_ctx_hypervisor_unregister_ctx(unsigned sched_ctx);

/* submit a requirement of resizing when a task taged with task_tag is executed */
void sched_ctx_hypervisor_resize(unsigned sched_ctx, int task_tag);

/* don't allow the hypervisor to resize a context */
void sched_ctx_hypervisor_stop_resize(unsigned sched_ctx);

/* allow the hypervisor to resize a context */
void sched_ctx_hypervisor_start_resize(unsigned sched_ctx);

/* check out the current policy of the hypervisor */
const char *sched_ctx_hypervisor_get_policy();

/* ask the hypervisor to add workers to a sched_ctx */
void sched_ctx_hypervisor_add_workers_to_sched_ctx(int* workers_to_add, unsigned nworkers_to_add, unsigned sched_ctx);

/* ask the hypervisor to remove workers from a sched_ctx */
void sched_ctx_hypervisor_remove_workers_from_sched_ctx(int* workers_to_remove, unsigned nworkers_to_remove, unsigned sched_ctx, unsigned now);

/* ask the hypervisor to move workers from one context to another */
void sched_ctx_hypervisor_move_workers(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, int *workers_to_move, unsigned nworkers_to_move, unsigned now);

/* ask the hypervisor to chose a distribution of workers in the required contexts */
void sched_ctx_hypervisor_size_ctxs(int *sched_ctxs, int nsched_ctxs, int *workers, int nworkers);

/* check if there are pending demands of resizing */
unsigned sched_ctx_hypervisor_get_size_req(int **sched_ctxs, int* nsched_ctxs, int **workers, int *nworkers);

/* save a demand of resizing */
void sched_ctx_hypervisor_save_size_req(int *sched_ctxs, int nsched_ctxs, int *workers, int nworkers);

/* clear the list of pending demands of resizing */
void sched_ctx_hypervisor_free_size_req(void);

/* check out if a context can be resized */
unsigned sched_ctx_hypervisor_can_resize(unsigned sched_ctx);

/* indicate the types of tasks a context will execute in order to better decide the sizing of ctxs */
void sched_ctx_hypervisor_set_type_of_task(struct starpu_codelet *cl, unsigned sched_ctx, uint32_t footprint);

#ifdef __cplusplus
}
#endif

#endif
