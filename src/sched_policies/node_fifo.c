#include "node_sched.h"
#include "fifo_queues.h"

static int push_task(struct _starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	int ret = _starpu_fifo_push_sorted_task(node->data, task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
	return ret;
}

static struct starpu_task *  pop_task(struct _starpu_sched_node * node, unsigned sched_ctx_id)
{
	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	struct starpu_task * task  = _starpu_fifo_pop_task(node->data, starpu_worker_get_id());
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
	if(task)
		return task;
	struct _starpu_sched_node * father = node->fathers[sched_ctx_id];
	if(father)
		return father->pop_task(father,sched_ctx_id);
	return NULL;
}


struct _starpu_sched_node * _starpu_sched_node_fifo_create(void)
{
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	node->data = _starpu_create_fifo();
	node->push_task = push_task;
	node->pop_task = pop_task;
	return node;
}


struct _starpu_fifo_taskq *  _starpu_node_fifo_get_fifo(struct _starpu_sched_node * node)
{
	STARPU_ASSERT(node->push_task == push_task);
	return node->data;
}
