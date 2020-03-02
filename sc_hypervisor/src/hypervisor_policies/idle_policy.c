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

#include "sc_hypervisor_policy.h"

unsigned worker_belong_to_other_sched_ctx(unsigned sched_ctx, int worker)
{
	unsigned *sched_ctxs = sc_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sc_hypervisor_get_nsched_ctxs();

	int i;
	for(i = 0; i < nsched_ctxs; i++)
		if(sched_ctxs[i] != sched_ctx && starpu_sched_ctx_contains_worker(worker, sched_ctxs[i]))
			return 1;
	return 0;
}

void idle_handle_idle_cycle(unsigned sched_ctx, int worker)
{
	if(sc_hypervisor_criteria_fulfilled(sched_ctx, worker))
	{
		if(worker_belong_to_other_sched_ctx(sched_ctx, worker))
			sc_hypervisor_remove_workers_from_sched_ctx(&worker, 1, sched_ctx, 1);
		else
			sc_hypervisor_policy_resize_to_unknown_receiver(sched_ctx, 0);
	}
}

struct sc_hypervisor_policy idle_policy =
{
	.size_ctxs = NULL,
	.handle_poped_task = NULL,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = idle_handle_idle_cycle,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = NULL,
	.end_ctx = NULL,
	.init_worker = NULL,
	.custom = 0,
	.name = "idle"
};
