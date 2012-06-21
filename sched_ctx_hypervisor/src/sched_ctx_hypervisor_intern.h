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
#include <common/htable32.h>

struct size_request {
	int *workers;
	int nworkers;
	int *sched_ctxs;
	int nsched_ctxs;
};

struct sched_ctx_hypervisor {
	struct sched_ctx_wrapper sched_ctx_w[STARPU_NMAX_SCHED_CTXS];
	int sched_ctxs[STARPU_NMAX_SCHED_CTXS];
	unsigned nsched_ctxs;
	unsigned resize[STARPU_NMAX_SCHED_CTXS];
	int min_tasks;
	struct hypervisor_policy policy;
	struct starpu_htbl32_node *configurations[STARPU_NMAX_SCHED_CTXS];
	struct starpu_htbl32_node *resize_requests[STARPU_NMAX_SCHED_CTXS];
	struct size_request *sr;
	int check_min_tasks[STARPU_NMAX_SCHED_CTXS];
};

struct sched_ctx_hypervisor_adjustment {
	int workerids[STARPU_NMAXWORKERS];
	int nworkers;
};

struct sched_ctx_hypervisor hypervisor;


void _add_config(unsigned sched_ctx);

void _remove_config(unsigned sched_ctx);
