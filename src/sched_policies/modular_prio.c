/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void starpu_initialize_prio_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *t;
	struct starpu_sched_component * eager_component;

	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);

	t = starpu_sched_tree_create(sched_ctx_id);
 	t->root = starpu_sched_component_prio_create(t, NULL);
	eager_component = starpu_sched_component_eager_create(t, NULL);

	starpu_sched_component_connect(t->root, eager_component);

	unsigned i;
	for(i = 0; i < starpu_worker_get_count() + starpu_combined_worker_get_count(); i++)
		starpu_sched_component_connect(eager_component, starpu_sched_component_worker_get(sched_ctx_id, i));

	starpu_sched_tree_update_workers(t);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)t);

	/* The application may use any integer */
	if (starpu_sched_ctx_min_priority_is_set(sched_ctx_id) == 0)
		starpu_sched_ctx_set_min_priority(sched_ctx_id, INT_MIN);
	if (starpu_sched_ctx_max_priority_is_set(sched_ctx_id) == 0)
		starpu_sched_ctx_set_max_priority(sched_ctx_id, INT_MAX);
}

static void deinitialize_prio_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}

struct starpu_sched_policy _starpu_sched_modular_prio_policy =
{
	.init_sched = starpu_initialize_prio_center_policy,
	.deinit_sched = deinitialize_prio_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "modular-prio",
	.policy_description = "prio modular policy"
};
