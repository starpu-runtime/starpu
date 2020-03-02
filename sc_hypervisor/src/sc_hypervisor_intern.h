/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <sc_hypervisor.h>
#include "uthash.h"

#define SC_SPEED_MAX_GAP_DEFAULT 50
#define SC_HYPERVISOR_DEFAULT_CPU_SPEED 5.0
#define SC_HYPERVISOR_DEFAULT_CUDA_SPEED 100.0

struct size_request
{
	int *workers;
	int nworkers;
	unsigned *sched_ctxs;
	int nsched_ctxs;
};


/* Entry in the resize request hash table.  */
struct resize_request_entry
{
	/* Key: the tag of tasks concerned by this resize request.  */
	uint32_t task_tag;

	/* Value: identifier of the scheduling context needing to be resized.
	 * The value doesn't matter since the hash table is used only to test
	 * membership of a task tag.  */
	unsigned sched_ctx;

	/* Bookkeeping.  */
	UT_hash_handle hh;
};

/* structure to indicate when the moving of workers was actually done
   (moved workers can be seen in the new ctx ) */
struct resize_ack
{
	/* receiver context */
	int receiver_sched_ctx;
	/* list of workers required to be moved */
	int *moved_workers;
	/* number of workers required to be moved */
	int nmoved_workers;
	/* list of workers that actually got in the receiver ctx */
	int *acked_workers;
};

struct configuration_entry
{
	/* Key: the tag of tasks concerned by this configuration.  */
	uint32_t task_tag;

	/* Value: configuration of the scheduling context.  */
	struct sc_hypervisor_policy_config *configuration;

	/* Bookkeeping.  */
	UT_hash_handle hh;
};

struct sc_hypervisor
{
	struct sc_hypervisor_wrapper sched_ctx_w[STARPU_NMAX_SCHED_CTXS];
	unsigned sched_ctxs[STARPU_NMAX_SCHED_CTXS];
	unsigned nsched_ctxs;
	unsigned resize[STARPU_NMAX_SCHED_CTXS];
	unsigned allow_remove[STARPU_NMAX_SCHED_CTXS];
	int min_tasks;
	struct sc_hypervisor_policy policy;

	struct configuration_entry *configurations[STARPU_NMAX_SCHED_CTXS];

	/* Set of pending resize requests for any context/tag pair.  */
	struct resize_request_entry *resize_requests[STARPU_NMAX_SCHED_CTXS];

	starpu_pthread_mutex_t conf_mut[STARPU_NMAX_SCHED_CTXS];
	starpu_pthread_mutex_t resize_mut[STARPU_NMAX_SCHED_CTXS];
	struct size_request *sr;
	int check_min_tasks[STARPU_NMAX_SCHED_CTXS];

	/* time when the hypervisor started */
	double start_executing_time;

	/* max speed diff btw ctx before triggering resizing */
	double max_speed_gap;

	/* criteria to trigger resizing */
	unsigned resize_criteria;

	/* value of the speed to compare the speed of the context to */
	double optimal_v[STARPU_NMAX_SCHED_CTXS];
};

struct sc_hypervisor_adjustment
{
	int workerids[STARPU_NMAXWORKERS];
	int nworkers;
};

extern struct sc_hypervisor hypervisor;

void _add_config(unsigned sched_ctx);

void _remove_config(unsigned sched_ctx);

double _get_max_speed_gap();

double _get_optimal_v(unsigned sched_ctx);

void _set_optimal_v(unsigned sched_ctx, double optimal_v);

int _sc_hypervisor_use_lazy_resize(void);

void _sc_hypervisor_allow_compute_idle(unsigned sched_ctx, int worker, unsigned allow);
