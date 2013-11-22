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

#define _STARPU_SCHED_NTASKS_THRESHOLD_DEFAULT 4
#define _STARPU_SCHED_EXP_LEN_THRESHOLD_DEFAULT 1000000000.0
static unsigned ntasks_threshold = _STARPU_SCHED_NTASKS_THRESHOLD_DEFAULT;
static double exp_len_threshold = _STARPU_SCHED_EXP_LEN_THRESHOLD_DEFAULT;

/* Random scheduler with fifo queues for its scheduling window and its workers. */

static void initialize_random_fifo_prefetching_center_policy(unsigned sched_ctx_id)
{
	const char *strval_ntasks_threshold = getenv("STARPU_NTASKS_THRESHOLD");
	if (strval_ntasks_threshold)
		ntasks_threshold = atof(strval_ntasks_threshold);

	const char *strval_exp_len_threshold = getenv("STARPU_EXP_LEN_THRESHOLD");
	if (strval_exp_len_threshold)
		exp_len_threshold = atof(strval_exp_len_threshold);

	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct starpu_sched_tree *t = starpu_sched_tree_create(sched_ctx_id);
 	t->root = starpu_sched_node_fifo_create(NULL);
	struct starpu_sched_node * random_node = starpu_sched_node_random_create(NULL);
	t->root->add_child(t->root, random_node);
	starpu_sched_node_set_father(random_node, t->root, sched_ctx_id);

	struct starpu_fifo_data fifo_data =
		{
			.ntasks_threshold = ntasks_threshold,
			.exp_len_threshold = exp_len_threshold,
		};

	unsigned i;
	for(i = 0; i < starpu_worker_get_count() + starpu_combined_worker_get_count(); i++)
	{
		struct starpu_sched_node * worker_node = starpu_sched_node_worker_get(i);
		STARPU_ASSERT(worker_node);

		struct starpu_sched_node * fifo_node = starpu_sched_node_fifo_create(&fifo_data);
		fifo_node->add_child(fifo_node, worker_node);
		starpu_sched_node_set_father(worker_node, fifo_node, sched_ctx_id);

		random_node->add_child(random_node, fifo_node);
		starpu_sched_node_set_father(fifo_node, random_node, sched_ctx_id);
	}
	starpu_sched_tree_update_workers(t);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)t);
}

static void deinitialize_random_fifo_prefetching_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}

struct starpu_sched_policy _starpu_sched_tree_random_prefetching_policy =
{
	.init_sched = initialize_random_fifo_prefetching_center_policy,
	.deinit_sched = deinitialize_random_fifo_prefetching_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "tree-random-prefetching",
	.policy_description = "random prefetching tree policy"
};

/* Random scheduler with priority queues for its scheduling window and its workers. */

static void initialize_random_prio_prefetching_center_policy(unsigned sched_ctx_id)
{
	const char *strval_ntasks_threshold = getenv("STARPU_NTASKS_THRESHOLD");
	if (strval_ntasks_threshold)
		ntasks_threshold = atof(strval_ntasks_threshold);

	const char *strval_exp_len_threshold = getenv("STARPU_EXP_LEN_THRESHOLD");
	if (strval_exp_len_threshold)
		exp_len_threshold = atof(strval_exp_len_threshold);

	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct starpu_sched_tree *t = starpu_sched_tree_create(sched_ctx_id);
 	t->root = starpu_sched_node_prio_create(NULL);
	struct starpu_sched_node * random_node = starpu_sched_node_random_create(NULL);
	t->root->add_child(t->root, random_node);
	starpu_sched_node_set_father(random_node, t->root, sched_ctx_id);

	struct starpu_prio_data prio_data =
		{
			.ntasks_threshold = ntasks_threshold,
			.exp_len_threshold = exp_len_threshold,
		};

	unsigned i;
	for(i = 0; i < starpu_worker_get_count() + starpu_combined_worker_get_count(); i++)
	{
		struct starpu_sched_node * worker_node = starpu_sched_node_worker_get(i);
		STARPU_ASSERT(worker_node);

		struct starpu_sched_node * prio_node = starpu_sched_node_prio_create(&prio_data);
		prio_node->add_child(prio_node, worker_node);
		starpu_sched_node_set_father(worker_node, prio_node, sched_ctx_id);

		random_node->add_child(random_node, prio_node);
		starpu_sched_node_set_father(prio_node, random_node, sched_ctx_id);
	}
	starpu_sched_tree_update_workers(t);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)t);
}

static void deinitialize_random_prio_prefetching_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}

struct starpu_sched_policy _starpu_sched_tree_random_prio_prefetching_policy =
{
	.init_sched = initialize_random_prio_prefetching_center_policy,
	.deinit_sched = deinitialize_random_prio_prefetching_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "tree-random-prio-prefetching",
	.policy_description = "random-prio prefetching tree policy"
};
