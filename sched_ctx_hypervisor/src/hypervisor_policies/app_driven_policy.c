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

void app_driven_handle_post_exec_hook(unsigned sched_ctx, struct starpu_htbl32_node_s* resize_requests, int task_tag)
{
	void* sched_ctx_pt =  _starpu_htbl_search_32(resize_requests, (uint32_t)task_tag);
	if(sched_ctx_pt && sched_ctx_pt != resize_requests)
	{
		_resize_to_unknown_receiver(sched_ctx);
		_starpu_htbl_insert_32(&resize_requests, (uint32_t)task_tag, NULL);
	}

}

struct hypervisor_policy app_driven_policy = {
	.size_ctxs = NULL,
	.handle_poped_task = NULL,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = app_driven_handle_post_exec_hook,
	.handle_submitted_job = NULL,
	.custom = 0,
	.name = "app_driven"
};
