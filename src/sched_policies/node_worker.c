#include "node_sched.h"
#include <core/workers.h>

static struct _starpu_sched_node * _worker_nodes[STARPU_NMAXWORKERS];

static struct _starpu_sched_node  * _starpu_sched_node_worker_create(int workerid);
struct _starpu_sched_node * _starpu_sched_node_worker_get(int workerid)
{
	STARPU_ASSERT(workerid >= 0 && workerid < STARPU_NMAXWORKERS);
	if(_worker_nodes[workerid])
		return _worker_nodes[workerid];
	else
		return _worker_nodes[workerid] = _starpu_sched_node_worker_create(workerid);
}



int _starpu_sched_node_worker_push_task(struct _starpu_sched_node * node, struct starpu_task *task)
{
	
	return _starpu_push_local_task(node->data, task, task->priority);

/*	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	int ret_val = _starpu_fifo_push_sorted_task(node->fifo, task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
	node->available(node);
	return ret_val;
*/
}

struct starpu_task * _starpu_sched_node_worker_pop_task(struct _starpu_sched_node *node,unsigned sched_ctx_id)
{
/*	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	struct starpu_task * task = _starpu_fifo_pop_local_task(node->fifo);
	if(task)
	{      
		STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
		return task;
	}
*/	struct _starpu_sched_node *father = node->fathers[sched_ctx_id];
//	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
	if(father == NULL)
		return NULL;
	else
		return father->pop_task(father,sched_ctx_id);
}
void _starpu_sched_node_worker_destroy(struct _starpu_sched_node *node)
{
	struct _starpu_worker * worker = node->data;
	unsigned id = worker->workerid;
	assert(_worker_nodes[id] == node);
	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS ; i++)
		if(node->fathers[i] != NULL)
			return;//this node is shared between several contexts
//	_starpu_destroy_fifo(node->fifo);
	_starpu_sched_node_destroy(node);
	_worker_nodes[id] = NULL;
}

static void available(struct _starpu_sched_node * worker_node)
{
	struct _starpu_worker * w = worker_node->data;
	starpu_pthread_mutex_t *sched_mutex = &w->sched_mutex;
	starpu_pthread_cond_t *sched_cond = &w->sched_cond;

	STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
	STARPU_PTHREAD_COND_SIGNAL(sched_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
}


static struct _starpu_sched_node  * _starpu_sched_node_worker_create(int workerid)
{
	STARPU_ASSERT(workerid >= 0 && workerid <  (int) starpu_worker_get_count());

	if(_worker_nodes[workerid])
		return _worker_nodes[workerid];

	struct _starpu_worker * worker = _starpu_get_worker_struct(workerid);
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	node->data = worker;
	//node->fifo = _starpu_create_fifo(),
	node->push_task = _starpu_sched_node_worker_push_task;
	node->pop_task = _starpu_sched_node_worker_pop_task;
	node->destroy_node = _starpu_sched_node_worker_destroy;
	node->available = available;
	node->workerids[0] = workerid;
	node->nworkers = 1;
	_worker_nodes[workerid] = node;
	return node;
}

int _starpu_sched_node_is_worker(struct _starpu_sched_node * node)
{
	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
		if(_worker_nodes[i] == node)
			return 1;
	return 0;
}

#ifndef STARPU_NO_ASSERT
static int _worker_consistant(struct _starpu_sched_node * node)
{
	int is_a_worker = 0;
	int i;
	for(i = 0; i<STARPU_NMAXWORKERS; i++)
		if(_worker_nodes[i] == node)
			is_a_worker = 1;
	if(!is_a_worker)
		return 0;
	struct _starpu_worker * worker = node->data;
	int id = worker->workerid;
	int father = 1;
	for(i = 0; i<STARPU_NMAX_SCHED_CTXS; i++)
		if(node->fathers[i] != NULL)
			return 1;
		else
			father = 0;
	return  father
		&& (_worker_nodes[id] == node)
		&&  node->nchilds == 0;
}
#endif

int _starpu_sched_node_worker_get_workerid(struct _starpu_sched_node * worker_node)
{
#ifndef STARPU_NO_ASSERT
	STARPU_ASSERT(_worker_consistant(worker_node));
#endif
	struct _starpu_worker * worker = worker_node->data;
	return worker->workerid;
}
