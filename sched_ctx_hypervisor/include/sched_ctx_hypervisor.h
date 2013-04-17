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

starpu_pthread_mutex_t act_hypervisor_mutex;


/* Forward declaration of an internal data structure
 * FIXME: Remove when no longer exposed.  */
struct resize_request_entry;

struct sched_ctx_hypervisor_policy
{
	const char* name;
	unsigned custom;
	void (*size_ctxs)(int *sched_ctxs, int nsched_ctxs , int *workers, int nworkers);
	void (*handle_idle_cycle)(unsigned sched_ctx, int worker);
	void (*handle_pushed_task)(unsigned sched_ctx, int worker);
	void (*handle_poped_task)(unsigned sched_ctx, int worker,struct starpu_task *task, uint32_t footprint);
	void (*handle_idle_end)(unsigned sched_ctx, int worker);

	void (*handle_post_exec_hook)(unsigned sched_ctx, int task_tag);

	void (*handle_submitted_job)(struct starpu_codelet *cl, unsigned sched_ctx, uint32_t footprint);
	
	void (*end_ctx)(unsigned sched_ctx);
};

struct starpu_sched_ctx_performance_counters *sched_ctx_hypervisor_init(struct sched_ctx_hypervisor_policy *policy);

void sched_ctx_hypervisor_shutdown(void);

void sched_ctx_hypervisor_register_ctx(unsigned sched_ctx, double total_flops);

void sched_ctx_hypervisor_unregister_ctx(unsigned sched_ctx);

void sched_ctx_hypervisor_resize(unsigned sched_ctx, int task_tag);

void sched_ctx_hypervisor_move_workers(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, int *workers_to_move, unsigned nworkers_to_move, unsigned now);

void sched_ctx_hypervisor_stop_resize(unsigned sched_ctx);

void sched_ctx_hypervisor_start_resize(unsigned sched_ctx);

const char *sched_ctx_hypervisor_get_policy();

void sched_ctx_hypervisor_add_workers_to_sched_ctx(int* workers_to_add, unsigned nworkers_to_add, unsigned sched_ctx);

void sched_ctx_hypervisor_remove_workers_from_sched_ctx(int* workers_to_remove, unsigned nworkers_to_remove, unsigned sched_ctx, unsigned now);

void sched_ctx_hypervisor_size_ctxs(int *sched_ctxs, int nsched_ctxs, int *workers, int nworkers);

unsigned sched_ctx_hypervisor_get_size_req(int **sched_ctxs, int* nsched_ctxs, int **workers, int *nworkers);

void sched_ctx_hypervisor_save_size_req(int *sched_ctxs, int nsched_ctxs, int *workers, int nworkers);

void sched_ctx_hypervisor_free_size_req(void);

unsigned sched_ctx_hypervisor_can_resize(unsigned sched_ctx);

void sched_ctx_hypervisor_set_type_of_task(struct starpu_codelet *cl, unsigned sched_ctx, uint32_t footprint);

#ifdef __cplusplus
}
#endif

#endif
