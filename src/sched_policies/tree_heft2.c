/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Université de Bordeaux 1
 * Copyright (C) 2013  INRIA
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
#include <float.h>

/* The two thresolds concerns the prio nodes, which contains queues
 * who can handle the priority of StarPU tasks. You can tune your
 * scheduling by benching those values and choose which one is the
 * best for your current application. 
 * The current value of the ntasks_threshold is the best we found
 * so far across several types of applications (cholesky, LU, stencil).
 */
#define _STARPU_SCHED_NTASKS_THRESHOLD_DEFAULT 30
#define _STARPU_SCHED_EXP_LEN_THRESHOLD_DEFAULT 1000000000.0

static void initialize_heft2_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);

	unsigned ntasks_threshold = _STARPU_SCHED_NTASKS_THRESHOLD_DEFAULT;
	double exp_len_threshold = _STARPU_SCHED_EXP_LEN_THRESHOLD_DEFAULT;

	const char *strval_ntasks_threshold = getenv("STARPU_NTASKS_THRESHOLD");
	if (strval_ntasks_threshold)
		ntasks_threshold = atof(strval_ntasks_threshold);

	const char *strval_exp_len_threshold = getenv("STARPU_EXP_LEN_THRESHOLD");
	if (strval_exp_len_threshold)
		exp_len_threshold = atof(strval_exp_len_threshold);


	struct starpu_sched_tree * t = starpu_sched_tree_create(sched_ctx_id);

	struct starpu_sched_node * perfmodel_node = starpu_sched_node_heft_create(NULL);
	struct starpu_sched_node * no_perfmodel_node = starpu_sched_node_eager_create(NULL);
	struct starpu_sched_node * calibrator_node = starpu_sched_node_eager_create(NULL);
	
	struct starpu_perfmodel_select_data perfmodel_select_data =
		{
			.calibrator_node = calibrator_node,
			.no_perfmodel_node = no_perfmodel_node,
			.perfmodel_node = perfmodel_node,
		};

	struct starpu_sched_node * perfmodel_select_node = starpu_sched_node_perfmodel_select_create(&perfmodel_select_data);
	t->root = perfmodel_select_node;

	perfmodel_select_node->add_child(perfmodel_select_node, calibrator_node);
	starpu_sched_node_set_father(calibrator_node, perfmodel_select_node, sched_ctx_id);
	perfmodel_select_node->add_child(perfmodel_select_node, perfmodel_node);
	starpu_sched_node_set_father(perfmodel_node, perfmodel_select_node, sched_ctx_id);
	perfmodel_select_node->add_child(perfmodel_select_node, no_perfmodel_node);
	starpu_sched_node_set_father(no_perfmodel_node, perfmodel_select_node, sched_ctx_id);

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

		struct starpu_sched_node * prio = starpu_sched_node_prio_create(&prio_data);
		prio->add_child(prio, worker_node);
		starpu_sched_node_set_father(worker_node, prio, sched_ctx_id);

		struct starpu_sched_node * impl_node = starpu_sched_node_best_implementation_create(NULL);
		impl_node->add_child(impl_node, prio);
		starpu_sched_node_set_father(prio, impl_node, sched_ctx_id);

		perfmodel_node->add_child(perfmodel_node, impl_node);
		starpu_sched_node_set_father(impl_node, perfmodel_node, sched_ctx_id);
	}

	starpu_sched_tree_update_workers(t);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)t);
}

static void deinitialize_heft2_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *t = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(t);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}

struct starpu_sched_policy _starpu_sched_tree_heft2_policy =
{
	.init_sched = initialize_heft2_center_policy,
	.deinit_sched = deinitialize_heft2_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_node_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_node_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "tree-heft2",
	.policy_description = "heft tree2 policy"
};
