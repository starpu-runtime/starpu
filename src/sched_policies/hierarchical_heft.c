#include <starpu_sched_node.h>
#include <core/workers.h>
#include "scheduler_maker.h"

static struct  starpu_sched_node_composed_recipe *  recipe_for_worker(enum starpu_worker_archtype a STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_node_composed_recipe * r = starpu_sched_node_create_recipe();
	starpu_sched_recipe_add_node(r, starpu_sched_node_best_implementation_create, NULL);
	starpu_sched_recipe_add_node(r, starpu_sched_node_fifo_create, NULL);
	return r;
}




static void initialize_heft_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);

	struct starpu_sched_specs specs;
	memset(&specs,0,sizeof(specs));

	struct starpu_heft_data heft_data =
	{
		.alpha = 1.0,
		.beta = 2.0,
		.gamma = 0.0,
		.idle_power = 0.0,
		.no_perf_model_node_create = starpu_sched_node_random_create,
		.arg_no_perf_model = NULL,
		.calibrating_node_create = starpu_sched_node_random_create,
		.arg_calibrating_node = NULL,
	};
	struct starpu_sched_node_composed_recipe * r = starpu_sched_node_create_recipe();
	starpu_sched_recipe_add_node(r,(struct starpu_sched_node * (*)(void*))starpu_sched_node_heft_create,&heft_data);
	specs.hwloc_machine_composed_sched_node = r;

	r = starpu_sched_node_create_recipe();
	starpu_sched_recipe_add_node(r, starpu_sched_node_best_implementation_create, NULL);
	starpu_sched_recipe_add_node(r, starpu_sched_node_fifo_create ,NULL);

	specs.hwloc_node_composed_sched_node = r;
	specs.worker_composed_sched_node = recipe_for_worker;

	struct starpu_sched_tree *t = _starpu_make_scheduler(sched_ctx_id, specs);

	starpu_destroy_composed_sched_node_recipe(specs.hwloc_machine_composed_sched_node);


	starpu_sched_tree_update_workers(t);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)t);
}

static void deinitialize_heft_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *t = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	starpu_sched_tree_destroy(t);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
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
	.policy_name = "tree-heft-hierarchical",
	.policy_description = "hierarchical heft tree policy"
};
