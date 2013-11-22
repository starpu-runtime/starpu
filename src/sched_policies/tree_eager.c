/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Simon Archipoff
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

#include <starpu_sched_node.h>
#include <starpu_scheduler.h>

static void initialize_eager_center_policy(unsigned sched_ctx_id)
{
	_STARPU_DISP("Warning: you are running the default tree-eager scheduler, which is not very smart. Make sure to read the StarPU documentation about adding performance models in order to be able to use the tree-heft scheduler instead.\n");

	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct starpu_sched_tree *t = starpu_sched_tree_create(sched_ctx_id);
 	t->root = starpu_sched_node_fifo_create(NULL);
	struct starpu_sched_node * eager_node = starpu_sched_node_eager_create(NULL);
	t->root->add_child(t->root, eager_node);
	starpu_sched_node_set_father(eager_node, t->root, sched_ctx_id);

	unsigned i;
	for(i = 0; i < starpu_worker_get_count() + starpu_combined_worker_get_count(); i++)
	{
		struct starpu_sched_node * worker_node = starpu_sched_node_worker_get(i);
		STARPU_ASSERT(worker_node);

		eager_node->add_child(eager_node, worker_node);
		starpu_sched_node_set_father(worker_node, eager_node, sched_ctx_id);
	}
	starpu_sched_tree_update_workers(t);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)t);
}

static void deinitialize_eager_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}

struct starpu_sched_policy _starpu_sched_tree_eager_policy =
{
	.init_sched = initialize_eager_center_policy,
	.deinit_sched = deinitialize_eager_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_node_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_node_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "tree-eager",
	.policy_description = "eager tree policy"
};
