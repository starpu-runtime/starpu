/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_sched_component.h>
#include <starpu_scheduler.h>
#include <limits.h>

static void starpu_initialize_prio_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_eager_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_WORKERS |
			STARPU_SCHED_SIMPLE_FIFO_ABOVE |
			STARPU_SCHED_SIMPLE_FIFO_ABOVE_PRIO |
			STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
}

struct starpu_sched_policy _starpu_sched_modular_prio_policy =
{
	.init_sched = starpu_initialize_prio_center_policy,
	.deinit_sched = starpu_sched_tree_deinitialize,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.policy_name = "modular-prio",
	.policy_description = "prio modular policy",
	.worker_type = STARPU_WORKER_LIST,
};
