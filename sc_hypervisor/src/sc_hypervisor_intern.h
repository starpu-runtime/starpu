/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2012  INRIA
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

#include <sched_ctx_hypervisor.h>
#include <common/uthash.h>
struct size_request
{
	int *workers;
	int nworkers;
	int *sched_ctxs;
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

struct configuration_entry
{
	/* Key: the tag of tasks concerned by this configuration.  */
	uint32_t task_tag;

	/* Value: configuration of the scheduling context.  */
	struct sched_ctx_hypervisor_policy_config *configuration;

	/* Bookkeeping.  */
	UT_hash_handle hh;
};

struct sched_ctx_hypervisor
{
	struct sched_ctx_hypervisor_wrapper sched_ctx_w[STARPU_NMAX_SCHED_CTXS];
	int sched_ctxs[STARPU_NMAX_SCHED_CTXS];
	unsigned nsched_ctxs;
	unsigned resize[STARPU_NMAX_SCHED_CTXS];
	unsigned allow_remove[STARPU_NMAX_SCHED_CTXS];
	int min_tasks;
	struct sched_ctx_hypervisor_policy policy;

	struct configuration_entry *configurations[STARPU_NMAX_SCHED_CTXS];

	/* Set of pending resize requests for any context/tag pair.  */
	struct resize_request_entry *resize_requests[STARPU_NMAX_SCHED_CTXS];

	starpu_pthread_mutex_t conf_mut[STARPU_NMAX_SCHED_CTXS];
	starpu_pthread_mutex_t resize_mut[STARPU_NMAX_SCHED_CTXS];
	struct size_request *sr;
	int check_min_tasks[STARPU_NMAX_SCHED_CTXS];

	/* time when the hypervisor started */
	double start_executing_time;
};

struct sched_ctx_hypervisor_adjustment
{
	int workerids[STARPU_NMAXWORKERS];
	int nworkers;
};

struct sched_ctx_hypervisor hypervisor;


void _add_config(unsigned sched_ctx);

void _remove_config(unsigned sched_ctx);
