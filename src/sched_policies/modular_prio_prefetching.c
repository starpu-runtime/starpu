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

/* Just as documentation example, here is the detailed equivalent of the
 * starpu_sched_component_initialize_simple_scheduler call below */
#if 0
static void initialize_prio_prefetching_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *t;
	struct starpu_sched_component * eager_component;

	t = starpu_sched_tree_create(sched_ctx_id);
 	t->root = starpu_sched_component_prio_create(t, NULL);
	eager_component = starpu_sched_component_eager_create(t, NULL);

	starpu_sched_component_connect(t->root, eager_component);

	struct starpu_sched_component_prio_data prio_data =
		{
			.ntasks_threshold = starpu_get_env_number_default("STARPU_NTASKS_THRESHOLD", _STARPU_SCHED_NTASKS_THRESHOLD_DEFAULT),
			.exp_len_threshold = starpu_get_env_float_default("STARPU_EXP_LEN_THRESHOLD", _STARPU_SCHED_EXP_LEN_THRESHOLD_DEFAULT),
		};

	unsigned i;
	for(i = 0; i < starpu_worker_get_count() + starpu_combined_worker_get_count(); i++)
	{
		struct starpu_sched_component * worker_component = starpu_sched_component_worker_new(sched_ctx_id, i);
		struct starpu_sched_component * prio_component = starpu_sched_component_prio_create(t, &prio_data);

		starpu_sched_component_connect(prio_component, worker_component);
		starpu_sched_component_connect(eager_component, prio_component);
	}
	starpu_sched_tree_update_workers(t);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)t);

	/* The application may use any integer */
	if (starpu_sched_ctx_min_priority_is_set(sched_ctx_id) == 0)
		starpu_sched_ctx_set_min_priority(sched_ctx_id, INT_MIN);
	if (starpu_sched_ctx_max_priority_is_set(sched_ctx_id) == 0)
		starpu_sched_ctx_set_max_priority(sched_ctx_id, INT_MAX);
}
#endif

static void initialize_prio_prefetching_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_component_initialize_simple_scheduler((starpu_sched_component_create_t) starpu_sched_component_eager_create, NULL,
			STARPU_SCHED_SIMPLE_DECIDE_WORKERS |
			STARPU_SCHED_SIMPLE_FIFO_ABOVE |
			STARPU_SCHED_SIMPLE_FIFO_ABOVE_PRIO |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_PRIO |
			STARPU_SCHED_SIMPLE_FIFOS_BELOW_READY |
			STARPU_SCHED_SIMPLE_IMPL, sched_ctx_id);
}

static void deinitialize_prio_prefetching_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
}

struct starpu_sched_policy _starpu_sched_modular_prio_prefetching_policy =
{
	.init_sched = initialize_prio_prefetching_center_policy,
	.deinit_sched = deinitialize_prio_prefetching_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_component_worker_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "modular-prio-prefetching",
	.policy_description = "prio prefetching modular policy",
	.worker_type = STARPU_WORKER_LIST,
};
