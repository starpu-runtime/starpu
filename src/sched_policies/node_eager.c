#include "node_sched.h"
#include <common/thread.h>
#include <core/sched_policy.h>
/*
static void _starpu_wake_all_interested_workers(struct starpu_task * task){
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(task->sched_ctx);
	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
		workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		unsigned worker = workers->get_next(workers, &it);
		int nimpl;
		for(nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
			if(starpu_worker_can_execute_task(worker, task, nimpl))
			{
				starpu_pthread_mutex_t *sched_mutex;
				starpu_pthread_cond_t *sched_cond;
				starpu_worker_get_sched_condition(worker, &sched_mutex, &sched_cond);
				_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
				_STARPU_PTHREAD_COND_SIGNAL(sched_cond);
				_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
				break;
			}
	}


}
*/

static int _starpu_eager_push_task(struct _starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(node->push_task == _starpu_eager_push_task);
	int ret_val = -1;
	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	ret_val = _starpu_fifo_push_task(node->data, task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
	node->available(node);
	return ret_val;
}

static struct starpu_task * _starpu_eager_pop_task(struct _starpu_sched_node *node,
						   unsigned sched_ctx_id)
{
	STARPU_ASSERT(node->pop_task == _starpu_eager_pop_task);

	int workerid = starpu_worker_get_id();
	
	/* Tell helgrind that it's fine to check for empty fifo without actual
	 * mutex (it's just a pointer) */
	/* block until some event happens */
	if (_starpu_fifo_empty(node->data))
	{
		VALGRIND_HG_MUTEX_UNLOCK_PRE(&node->mutex);
		VALGRIND_HG_MUTEX_UNLOCK_POST(&node->mutex);
		return NULL;
	}
	VALGRIND_HG_MUTEX_UNLOCK_PRE(&node->mutex);
	VALGRIND_HG_MUTEX_UNLOCK_POST(&node->mutex);

	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	struct starpu_task* task = _starpu_fifo_pop_task(node->data, workerid);
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
	if(!task )
	{
		struct _starpu_sched_node * father = node->fathers[sched_ctx_id];
		if(father)
			task = father->pop_task(father, sched_ctx_id);
	}
	return task;
}

void _starpu_destroy_eager(struct _starpu_sched_node * node)
{
	_starpu_destroy_fifo(node->data);
	_starpu_sched_node_destroy(node);
}


struct _starpu_sched_node * _starpu_sched_node_eager_create(void)
{
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	node->data = _starpu_create_fifo();
	node->push_task = _starpu_eager_push_task;
	node->pop_task = _starpu_eager_pop_task;
	node->childs = NULL;
	node->destroy_node = _starpu_destroy_eager;
	return node;
}



static void initialize_eager_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct _starpu_sched_tree *data = malloc(sizeof(struct _starpu_sched_tree));
	STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
 	data->root = _starpu_sched_node_eager_create();
	
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
		_starpu_sched_node_add_child(t->root,
					     _starpu_sched_node_worker_get(workerids[i]),
					     sched_ctx_id);
}

static void remove_worker_eager(unsigned sched_ctx_id, int * workerids, unsigned nworkers)
{
	struct _starpu_sched_tree *t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned i;
	for(i = 0; i < nworkers; i++)
		_starpu_sched_node_remove_child(t->root,
						_starpu_sched_node_worker_get(workerids[i]),
						sched_ctx_id);
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
