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

#include <sched_ctx_hypervisor_intern.h>

static void app_driven_handle_post_exec_hook(unsigned sched_ctx, int task_tag)
{
	_resize_to_unknown_receiver(sched_ctx, 1);
}

struct sched_ctx_hypervisor_policy app_driven_policy =
{
	.size_ctxs = NULL,
	.handle_poped_task = NULL,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = app_driven_handle_post_exec_hook,
	.handle_submitted_job = NULL,
	.end_ctx = NULL,
	.custom = 0,
	.name = "app_driven"
};
