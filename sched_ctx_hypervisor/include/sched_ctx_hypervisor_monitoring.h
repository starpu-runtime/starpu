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

#ifndef SCHED_CTX_HYPERVISOR_MONITORING_H
#define SCHED_CTX_HYPERVISOR_MONITORING_H

#include <sched_ctx_hypervisor.h>

#ifdef __cplusplus
extern "C"
{
#endif

struct sched_ctx_hypervisor_resize_ack
{
	int receiver_sched_ctx;
	int *moved_workers;
	int nmoved_workers;
	int *acked_workers;
};

/* wrapper attached to a sched_ctx storing monitoring information */
struct sched_ctx_hypervisor_wrapper
{
	/* the sched_ctx it monitors */
	unsigned sched_ctx;

	/* user configuration meant to limit resizing */
	struct sched_ctx_hypervisor_policy_config *config;

	/* idle time of workers in this context */
	double current_idle_time[STARPU_NMAXWORKERS];
	
	/* list of workers that will leave this contexts (lazy resizing process) */
	int worker_to_be_removed[STARPU_NMAXWORKERS];

	/* number of tasks pushed on each worker in this ctx */
	int pushed_tasks[STARPU_NMAXWORKERS];

	/* number of tasks poped from each worker in this ctx */
	int poped_tasks[STARPU_NMAXWORKERS];

	/* number of flops the context has to execute */
	double total_flops;

	/* number of flops executed since the biginning until now */
	double total_elapsed_flops[STARPU_NMAXWORKERS];

	/* number of flops executed since last resizing */
	double elapsed_flops[STARPU_NMAXWORKERS];

	/* data quantity executed on each worker in this ctx */
	size_t elapsed_data[STARPU_NMAXWORKERS];

	/* nr of tasks executed on each worker in this ctx */
	int elapsed_tasks[STARPU_NMAXWORKERS];

	/* the average speed of workers when they belonged to this context */
	double ref_velocity[STARPU_NMAXWORKERS];

	/* number of flops submitted to this ctx */
	double submitted_flops;

	/* number of flops that still have to be executed in this ctx */
	double remaining_flops;
	
	/* the start time of the resizing sample of this context*/
	double start_time;

	/* the first time a task was pushed to this context*/
	double real_start_time;

	/* the workers don't leave the current ctx until the receiver ctx 
	   doesn't ack the receive of these workers */
	struct sched_ctx_hypervisor_resize_ack resize_ack;

	/* mutex to protect the ack of workers */
	starpu_pthread_mutex_t mutex;
};

struct sched_ctx_hypervisor_wrapper *sched_ctx_hypervisor_get_wrapper(unsigned sched_ctx);

int *sched_ctx_hypervisor_get_sched_ctxs();

int sched_ctx_hypervisor_get_nsched_ctxs();

int sched_ctx_hypervisor_get_nworkers_ctx(unsigned sched_ctx, enum starpu_archtype arch);

double sched_ctx_hypervisor_get_elapsed_flops_per_sched_ctx(struct sched_ctx_hypervisor_wrapper *sc_w);

double sched_ctx_hypervisor_get_total_elapsed_flops_per_sched_ctx(struct sched_ctx_hypervisor_wrapper* sc_w);

/* compute an average value of the cpu/cuda velocity */
double sched_ctx_hypervisor_get_velocity_per_worker_type(struct sched_ctx_hypervisor_wrapper* sc_w, enum starpu_archtype arch);

double sched_ctx_hypervisor_get_velocity(struct sched_ctx_hypervisor_wrapper *sc_w, enum starpu_archtype arch);

#ifdef __cplusplus
}
#endif

#endif
