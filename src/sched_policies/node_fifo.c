#include "node_sched.h"
#include "fifo_queues.h"
#include <starpu_scheduler.h>

static double estimated_finish_time(struct _starpu_sched_node * node)
{
	struct _starpu_fifo_taskq * fifo = node->data;
	return fifo->exp_end;
}

static double estimated_load(struct _starpu_sched_node * node)
{
	double relative_speedup = 0.0;
	int i;
	int nworkers = node->is_homogeneous ? 1 : node->nworkers;
	for(i = 0; i < nworkers; i++)
		relative_speedup += starpu_worker_get_relative_speedup(starpu_worker_get_perf_archtype(node->workerids[i]));
	relative_speedup /= nworkers;
	struct _starpu_fifo_taskq * fifo = node->data;
	STARPU_ASSERT(!_STARPU_IS_ZERO(relative_speedup));
	double load = fifo->ntasks / relative_speedup; 
	for(i = 0; i < node->nchilds; i++)
	{
		struct _starpu_sched_node * c = node->childs[i];
		load += c->estimated_load(c);
	}
	return load;
}
static int push_task(struct _starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_PTHREAD_RWLOCK_WRLOCK(&node->mutex);
	struct _starpu_fifo_taskq * fifo = node->data;
	int ret = _starpu_fifo_push_sorted_task(fifo, task);
	fifo->exp_end += task->predicted;
	STARPU_PTHREAD_RWLOCK_UNLOCK(&node->mutex);
	node->available(node);
	return ret;
}

static struct starpu_task * pop_task(struct _starpu_sched_node * node, unsigned sched_ctx_id)
{
	struct _starpu_fifo_taskq * fifo = node->data;
	STARPU_PTHREAD_RWLOCK_WRLOCK(&node->mutex);
	struct starpu_task * task  = _starpu_fifo_pop_task(fifo, starpu_worker_get_id());
	if(task)
		fifo->exp_start = starpu_timing_now() + task->predicted;
	STARPU_PTHREAD_RWLOCK_UNLOCK(&node->mutex);
	if(task)
		return task;
	struct _starpu_sched_node * father = node->fathers[sched_ctx_id];
	if(father)
		return father->pop_task(father,sched_ctx_id);
	return NULL;
}

int _starpu_sched_node_is_fifo(struct _starpu_sched_node * node)
{
	return node->estimated_finish_time == estimated_finish_time
		|| node->estimated_load == estimated_load
		|| node->push_task == node->push_task
		|| node->pop_task == node->pop_task;
}

struct _starpu_sched_node * _starpu_sched_node_fifo_create(void)
{
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	node->data = _starpu_create_fifo();
	node->estimated_finish_time = estimated_finish_time;
	node->estimated_load = estimated_load;
	node->push_task = push_task;
	node->pop_task = pop_task;
	return node;
}


struct _starpu_fifo_taskq *  _starpu_sched_node_fifo_get_fifo(struct _starpu_sched_node * node)
{
	STARPU_ASSERT(node->push_task == push_task);
	return node->data;
}
