#include "node_sched.h"
#include <core/workers.h>
#include "scheduler_maker.h"

static struct  _starpu_composed_sched_node_recipe *  recipe_for_worker(enum starpu_worker_archtype a STARPU_ATTRIBUTE_UNUSED)
{
	struct _starpu_composed_sched_node_recipe * r = _starpu_sched_node_create_recipe();
	_starpu_sched_recipe_add_node(r, _starpu_sched_node_fifo_create, NULL);
	return NULL; r;
}

static void initialize_heft_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	
	struct _starpu_sched_specs specs;
	memset(&specs,0,sizeof(specs));
	
	specs.hwloc_machine_composed_sched_node = ({
			struct _starpu_composed_sched_node_recipe * r = _starpu_sched_node_create_recipe();
			_starpu_sched_recipe_add_node(r, _starpu_sched_node_heft_create,NULL);
			r;});

	specs.hwloc_node_composed_sched_node = ({
			struct _starpu_composed_sched_node_recipe * r = _starpu_sched_node_create_recipe();
			_starpu_sched_recipe_add_node(r, _starpu_sched_node_fifo_create,NULL);
			r;});
	specs.worker_composed_sched_node = recipe_for_worker;

	struct _starpu_sched_tree *data = _starpu_make_scheduler(sched_ctx_id, specs);

	_starpu_destroy_composed_sched_node_recipe(specs.hwloc_machine_composed_sched_node);

	

	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);


}

static void deinitialize_heft_center_policy(unsigned sched_ctx_id)
{
	struct _starpu_sched_tree *t = (struct _starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	_starpu_bitmap_destroy(t->workers);
	_starpu_tree_destroy(t, sched_ctx_id);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}





struct starpu_sched_policy _starpu_sched_tree_heft_policy =
{
	.init_sched = initialize_heft_center_policy,
	.deinit_sched = deinitialize_heft_center_policy,
	.add_workers = _starpu_tree_add_workers,
	.remove_workers = _starpu_tree_remove_workers,
	.push_task = _starpu_tree_push_task,
	.pop_task = _starpu_tree_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "tree-heft",
	.policy_description = "heft tree policy"
};
