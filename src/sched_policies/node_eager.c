#include <common/thread.h>
#include <core/sched_policy.h>
#include "node_sched.h"
#include "fifo_queues.h"



static void initialize_eager_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct _starpu_sched_tree *data = malloc(sizeof(struct _starpu_sched_tree));
	STARPU_PTHREAD_RWLOCK_INIT(&data->lock,NULL);
 	data->root = _starpu_sched_node_fifo_create(NULL);
	data->workers = _starpu_bitmap_create();
	unsigned i;
	for(i = 0; i < starpu_worker_get_count(); i++)
	{
		struct _starpu_sched_node * node = _starpu_sched_node_worker_get(i);
		if(!node)
			continue;
		node->fathers[sched_ctx_id] = data->root;
		_starpu_sched_node_add_child(data->root, node);
	}
	_starpu_set_workers_bitmaps();
	_starpu_tree_call_init_data(data);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);
}

static void deinitialize_eager_center_policy(unsigned sched_ctx_id)
{
	struct _starpu_sched_tree *tree = (struct _starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	_starpu_sched_tree_destroy(tree, sched_ctx_id);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}



struct starpu_sched_policy _starpu_sched_tree_eager_policy =
{
	.init_sched = initialize_eager_center_policy,
	.deinit_sched = deinitialize_eager_center_policy,
	.add_workers = _starpu_tree_add_workers,
	.remove_workers = _starpu_tree_remove_workers,
	.push_task = _starpu_tree_push_task,
	.pop_task = _starpu_tree_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,//pop_every_task_eager_policy,
	.policy_name = "tree",
	.policy_description = "test tree policy"
};
