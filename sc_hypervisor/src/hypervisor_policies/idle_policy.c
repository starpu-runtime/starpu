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

#include "sched_ctx_hypervisor_policy.h"

unsigned worker_belong_to_other_sched_ctx(unsigned sched_ctx, int worker)
{
	int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();

	int i;
	for(i = 0; i < nsched_ctxs; i++)
		if(sched_ctxs[i] != sched_ctx && starpu_sched_ctx_contains_worker(worker, sched_ctxs[i]))
			return 1;
	return 0;
}

void idle_handle_idle_cycle(unsigned sched_ctx, int worker)
{
	struct sched_ctx_hypervisor_wrapper* sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctx);
	struct sched_ctx_hypervisor_policy_config *config = sc_w->config;
	if(config != NULL &&  sc_w->current_idle_time[worker] > config->max_idle[worker])
	{
		if(worker_belong_to_other_sched_ctx(sched_ctx, worker))
			sched_ctx_hypervisor_remove_workers_from_sched_ctx(&worker, 1, sched_ctx, 1);
		else
			_resize_to_unknown_receiver(sched_ctx, 0);
	}
}

struct sched_ctx_hypervisor_policy idle_policy =
{
	.size_ctxs = NULL,
	.handle_poped_task = NULL,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = idle_handle_idle_cycle,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = NULL,
	.end_ctx = NULL,
	.custom = 0,
	.name = "idle"
};
