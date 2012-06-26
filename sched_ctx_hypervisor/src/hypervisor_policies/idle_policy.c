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

#include "policy_tools.h"

void idle_handle_idle_cycle(unsigned sched_ctx, int worker)
{
	struct sched_ctx_wrapper* sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctx);
	struct policy_config *config = sc_w->config;
	if(config != NULL &&  sc_w->current_idle_time[worker] > config->max_idle[worker])
		_resize_to_unknown_receiver(sched_ctx, 0);
}

struct hypervisor_policy idle_policy = {
	.size_ctxs = NULL,
	.handle_poped_task = NULL,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = idle_handle_idle_cycle,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = NULL,
	.custom = 0,
	.name = "idle"
};
