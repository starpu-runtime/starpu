#include <common/thread.h>
#include <core/sched_policy.h>
#include "node_sched.h"
#include "fifo_queues.h"



static void initialize_eager_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct _starpu_sched_tree *data = malloc(sizeof(struct _starpu_sched_tree));
	STARPU_PTHREAD_RWLOCK_INIT(&data->mutex,NULL);
 	data->root = _starpu_sched_node_fifo_create();
	
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);
}

static void deinitialize_eager_center_policy(unsigned sched_ctx_id)
{
	struct _starpu_sched_tree *tree = (struct _starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	_starpu_tree_destroy(tree, sched_ctx_id);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}

static void add_worker_eager(unsigned sched_ctx_id, int * workerids, unsigned nworkers)
{
	struct _starpu_sched_tree *t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned i;
	for(i = 0; i < nworkers; i++)
	{
		t->root->add_child(t->root, _starpu_sched_node_worker_get(workerids[i]), sched_ctx_id);
		_starpu_sched_node_worker_get(workerids[i])->fathers[sched_ctx_id] = t->root;
	}
}

static void remove_worker_eager(unsigned sched_ctx_id, int * workerids, unsigned nworkers)
{
	struct _starpu_sched_tree *t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned i;
	for(i = 0; i < nworkers; i++)
	{
		t->root->remove_child(t->root, _starpu_sched_node_worker_get(workerids[i]), sched_ctx_id);
		_starpu_sched_node_worker_get(workerids[i])->fathers[sched_ctx_id] = NULL;
	}
}



struct starpu_sched_policy _starpu_sched_tree_eager_policy =
{
	.init_sched = initialize_eager_center_policy,
	.deinit_sched = deinitialize_eager_center_policy,
	.add_workers = add_worker_eager,
	.remove_workers = remove_worker_eager,
	.push_task = _starpu_tree_push_task,
	.pop_task = _starpu_tree_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,//pop_every_task_eager_policy,
	.policy_name = "tree",
	.policy_description = "test tree policy"
};
