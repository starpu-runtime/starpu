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
#include <core/workers.h>

static struct  starpu_sched_component_composed_recipe *  recipe_for_worker(enum starpu_worker_archtype a STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_component_composed_recipe * r = starpu_sched_component_composed_recipe_create();
	starpu_sched_component_composed_recipe_add(r, (starpu_sched_component_create_t) starpu_sched_component_best_implementation_create, NULL);
	starpu_sched_component_composed_recipe_add(r, (starpu_sched_component_create_t) starpu_sched_component_fifo_create, NULL);
	return r;
}




static void initialize_heft_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_component_specs specs;
	memset(&specs,0,sizeof(specs));


	struct starpu_sched_component_mct_data heft_data =
	{
		.alpha = 1.0,
		.beta = 1.0,
		._gamma = 0.0,
		.idle_power = 0.0,
		/*
		.no_perf_model_component_create = starpu_sched_component_random_create,
		.arg_no_perf_model = NULL,
		.calibrating_component_create = starpu_sched_component_random_create,
		.arg_calibrating_component = NULL,
		*/
	};
	struct starpu_sched_component_composed_recipe * r = starpu_sched_component_composed_recipe_create();
	/* FIXME: add perfmodel_select component */
	starpu_sched_component_composed_recipe_add(r, (starpu_sched_component_create_t) starpu_sched_component_heft_create,&heft_data);
	specs.hwloc_machine_composed_sched_component = r;

	r = starpu_sched_component_composed_recipe_create();
	starpu_sched_component_composed_recipe_add(r, (starpu_sched_component_create_t) starpu_sched_component_best_implementation_create, NULL);
	starpu_sched_component_composed_recipe_add(r, (starpu_sched_component_create_t) starpu_sched_component_fifo_create ,NULL);

	specs.hwloc_component_composed_sched_component = r;
	specs.worker_composed_sched_component = recipe_for_worker;

	struct starpu_sched_tree *t = starpu_sched_component_make_scheduler(sched_ctx_id, specs);

	starpu_sched_component_composed_recipe_destroy(specs.hwloc_machine_composed_sched_component);


	starpu_sched_tree_update_workers(t);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)t);
}

static void deinitialize_heft_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *t = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	starpu_sched_tree_destroy(t);
}





struct starpu_sched_policy _starpu_sched_tree_heft_hierarchical_policy =
{
	.init_sched = initialize_heft_center_policy,
	.deinit_sched = deinitialize_heft_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "modular-heft-hierarchical",
	.policy_description = "hierarchical heft tree policy",
	.worker_type = STARPU_WORKER_LIST,
};
