/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Simon Archipoff
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

#define _STARPU_SCHED_NTASKS_THRESHOLD_DEFAULT 2
#define _STARPU_SCHED_EXP_LEN_THRESHOLD_DEFAULT 1000000000.0

static void initialize_eager_prefetching_center_policy(unsigned sched_ctx_id)
{
	_STARPU_DISP("Warning: you are running the default modular-eager-prefetching scheduler, which is not a very smart scheduler. Make sure to read the StarPU documentation about adding performance models in order to be able to use the modular-heft scheduler instead.\n");

	struct starpu_sched_tree *t;
	struct starpu_sched_component * eager_component;

	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);

	t = starpu_sched_tree_create(sched_ctx_id);
 	t->root = starpu_sched_component_fifo_create(t, NULL);
	eager_component = starpu_sched_component_eager_create(t, NULL);

	starpu_sched_component_connect(t->root, eager_component);

	struct starpu_sched_component_fifo_data fifo_data =
		{
			.ntasks_threshold = starpu_get_env_number_default("STARPU_NTASKS_THRESHOLD", _STARPU_SCHED_NTASKS_THRESHOLD_DEFAULT),
			.exp_len_threshold = starpu_get_env_float_default("STARPU_EXP_LEN_THRESHOLD", _STARPU_SCHED_EXP_LEN_THRESHOLD_DEFAULT),
		};

	unsigned i;
	for(i = 0; i < starpu_worker_get_count() + starpu_combined_worker_get_count(); i++)
	{
		struct starpu_sched_component * worker_component = starpu_sched_component_worker_get(sched_ctx_id, i);
		struct starpu_sched_component * fifo_component = starpu_sched_component_fifo_create(t, &fifo_data);

		starpu_sched_component_connect(fifo_component, worker_component);
		starpu_sched_component_connect(eager_component, fifo_component);
	}
	starpu_sched_tree_update_workers(t);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)t);
}

static void deinitialize_eager_prefetching_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}

struct starpu_sched_policy _starpu_sched_modular_eager_prefetching_policy =
{
	.init_sched = initialize_eager_prefetching_center_policy,
	.deinit_sched = deinitialize_eager_prefetching_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "modular-eager-prefetching",
	.policy_description = "eager with prefetching modular policy"
};
