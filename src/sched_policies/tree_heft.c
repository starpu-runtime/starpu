/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Marc Sergent 
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

/* Alpha, Beta and Gamma are heft-specific values, which allows the
 * user to set more precisely the weight of each computing value.
 * Beta, for example, controls the weight of communications between
 * memories for the computation of the best node to choose. 
 */
/* The two thresolds concerns the prio nodes, which contains queues
 * who can handle the priority of StarPU tasks. You can tune your
 * scheduling by benching those values and choose which one is the
 * best for your current application. 
 * The current value of the ntasks_threshold is the best we found
 * so far across several types of applications (cholesky, LU, stencil).
 */
#define _STARPU_SCHED_ALPHA_DEFAULT 1.0
#define _STARPU_SCHED_BETA_DEFAULT 1.0
#define _STARPU_SCHED_GAMMA_DEFAULT 1000.0
#define _STARPU_SCHED_NTASKS_THRESHOLD_DEFAULT 30
#define _STARPU_SCHED_EXP_LEN_THRESHOLD_DEFAULT 1000000000.0
static double alpha = _STARPU_SCHED_ALPHA_DEFAULT;
static double beta = _STARPU_SCHED_BETA_DEFAULT;
static double _gamma = _STARPU_SCHED_GAMMA_DEFAULT;
static unsigned ntasks_threshold = _STARPU_SCHED_NTASKS_THRESHOLD_DEFAULT;
static double exp_len_threshold = _STARPU_SCHED_EXP_LEN_THRESHOLD_DEFAULT;

#ifdef STARPU_USE_TOP
static const float alpha_minimum=0;
static const float alpha_maximum=10.0;
static const float beta_minimum=0;
static const float beta_maximum=10.0;
static const float gamma_minimum=0;
static const float gamma_maximum=10000.0;
static const float idle_power_minimum=0;
static const float idle_power_maximum=10000.0;
#endif /* !STARPU_USE_TOP */

static double idle_power = 0.0;

#ifdef STARPU_USE_TOP
static void param_modified(struct starpu_top_param* d)
{
#ifdef STARPU_DEVEL
#warning FIXME: get sched ctx to get alpha/beta/gamma/idle values
#endif
	/* Just to show parameter modification. */
	fprintf(stderr,
		"%s has been modified : "
		"alpha=%f|beta=%f|gamma=%f|idle_power=%f !\n",
		d->name, alpha,beta,_gamma, idle_power);
}
#endif /* !STARPU_USE_TOP */

static void initialize_heft_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	const char *strval_alpha = getenv("STARPU_SCHED_ALPHA");
	if (strval_alpha)
		alpha = atof(strval_alpha);

	const char *strval_beta = getenv("STARPU_SCHED_BETA");
	if (strval_beta)
		beta = atof(strval_beta);

	const char *strval_gamma = getenv("STARPU_SCHED_GAMMA");
	if (strval_gamma)
		_gamma = atof(strval_gamma);

	const char *strval_idle_power = getenv("STARPU_IDLE_POWER");
	if (strval_idle_power)
		idle_power = atof(strval_idle_power);

	const char *strval_ntasks_threshold = getenv("STARPU_NTASKS_THRESHOLD");
	if (strval_ntasks_threshold)
		ntasks_threshold = atof(strval_ntasks_threshold);

	const char *strval_exp_len_threshold = getenv("STARPU_EXP_LEN_THRESHOLD");
	if (strval_exp_len_threshold)
		exp_len_threshold = atof(strval_exp_len_threshold);

#ifdef STARPU_USE_TOP
	starpu_top_register_parameter_float("HEFT_ALPHA", &alpha,
					    alpha_minimum, alpha_maximum, param_modified);
	starpu_top_register_parameter_float("HEFT_BETA", &beta,
					    beta_minimum, beta_maximum, param_modified);
	starpu_top_register_parameter_float("HEFT_GAMMA", &_gamma,
					    gamma_minimum, gamma_maximum, param_modified);
	starpu_top_register_parameter_float("HEFT_IDLE_POWER", &idle_power,
					    idle_power_minimum, idle_power_maximum, param_modified);
#endif /* !STARPU_USE_TOP */

	
/* The scheduling strategy look like this :
 *
 *									  |
 *								window_node
 *									  |
 * perfmodel_node <--push-- perfmodel_select_node --push--> eager_node
 *			|						   						     |
 *			|						   						     |
 *			>----------------------------------------------------<
 *					  |								|
 *				best_impl_node					best_impl_node
 *					  |								|
 *				  prio_node						prio_node
 *					  |								|
 *				 worker_node				   worker_node
 *
 * A window contain the tasks that failed to be pushed, so as when the prio_nodes reclaim
 * tasks by calling room to their father (classically, just after a successful pop have
 * been made by its associated worker_node), this call goes up to the window_node which
 * pops a task from its local queue and try to schedule it by pushing it to the
 * decision_node. 
 * The decision node takes care of the scheduling of tasks which are not
 * calibrated, or tasks which don't have a performance model, because the scheduling
 * architecture of this scheduler for tasks with no performance model is exactly
 * the same as the tree-prio scheduler.
 * Tasks with a perfmodel are pushed to the perfmodel_node, which takes care of the
 * scheduling of those tasks on the correct worker_node.
 * Finally, the task will be pushed to the prio_node which is the direct
 * father in the tree of the worker_node the task has been scheduled on. This
 * node will push the task on its local queue if no one of the two thresholds
 * have been reached for it, or send a push_error signal to its father.
 */
	struct starpu_sched_tree * t = starpu_sched_tree_create(sched_ctx_id);

	struct starpu_sched_node * window_node = starpu_sched_node_prio_create(NULL);
	t->root = window_node;

	struct starpu_mct_data mct_data =
		{
			.alpha = alpha,
			.beta = beta,
			.gamma = _gamma,
			.idle_power = idle_power,
		};

	struct starpu_sched_node * perfmodel_node = starpu_sched_node_mct_create(&mct_data);
	struct starpu_sched_node * no_perfmodel_node = starpu_sched_node_eager_create(NULL);
	struct starpu_sched_node * calibrator_node = starpu_sched_node_eager_create(NULL);
	
	struct starpu_perfmodel_select_data perfmodel_select_data =
		{
			.calibrator_node = calibrator_node,
			.no_perfmodel_node = no_perfmodel_node,
			.perfmodel_node = perfmodel_node,
		};

	struct starpu_sched_node * perfmodel_select_node = starpu_sched_node_perfmodel_select_create(&perfmodel_select_data);
	window_node->add_child(window_node, perfmodel_select_node);
	starpu_sched_node_set_father(perfmodel_select_node, window_node, sched_ctx_id);

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

		perfmodel_select_node->add_child(perfmodel_select_node, impl_node);
		starpu_sched_node_set_father(impl_node, perfmodel_select_node, sched_ctx_id);
	}

	starpu_sched_tree_update_workers(t);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)t);
}

static void deinitialize_heft_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *t = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(t);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}

struct starpu_sched_policy _starpu_sched_tree_heft_policy =
{
	.init_sched = initialize_heft_center_policy,
	.deinit_sched = deinitialize_heft_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_node_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_node_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "tree-heft",
	.policy_description = "heft tree policy"
};
