#include "node_sched.h"
#include "fifo_queues.h"
#include <starpu_scheduler.h>


struct _starpu_fifo_data
{
	struct _starpu_fifo_taskq * fifo;
	starpu_pthread_mutex_t mutex;
};


static struct _starpu_task_execute_preds estimated_execute_preds(struct _starpu_sched_node * node,
								 struct starpu_task * task)
{
	if(node->nchilds == 0)
	{
		struct _starpu_task_execute_preds p = { CANNOT_EXECUTE };
		return p;
	}
	
	struct _starpu_task_execute_preds preds = node->childs[0]->estimated_execute_preds(node->childs[0],task);

	if(preds.state == PERF_MODEL)
	{
		struct _starpu_fifo_data * data = node->data;
		struct _starpu_fifo_taskq * fifo = data->fifo;
		starpu_pthread_mutex_t * mutex = &data->mutex;
		STARPU_PTHREAD_MUTEX_LOCK(mutex);
		preds.expected_finish_time = _starpu_compute_expected_time(fifo->exp_start,
									   preds.expected_finish_time + fifo->exp_end,
									   preds.expected_length + fifo->exp_len,
									   preds.expected_transfer_length);
		STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
	}
	return preds;
}

static double estimated_load(struct _starpu_sched_node * node)
{
	double relative_speedup = 0.0;
	int i;
	STARPU_ASSERT(node->nworkers > 0);
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
	STARPU_ASSERT(node->nworkers > 0);
	struct _starpu_fifo_data * data = node->data;
	struct _starpu_fifo_taskq * fifo = data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	STARPU_PTHREAD_MUTEX_LOCK(mutex);
	STARPU_ASSERT(!isnan(fifo->exp_end));
	STARPU_ASSERT(!isnan(fifo->exp_len));
	STARPU_ASSERT(!isnan(fifo->exp_start));
	int ret = _starpu_fifo_push_sorted_task(fifo, task);
	if(!isnan(task->predicted))
	{
		fifo->exp_len += task->predicted/node->nworkers;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;
	}
	STARPU_ASSERT(!isnan(fifo->exp_end));
	STARPU_ASSERT(!isnan(fifo->exp_len));
	STARPU_ASSERT(!isnan(fifo->exp_start));
	STARPU_PTHREAD_MUTEX_UNLOCK(mutex);

	node->available(node);
	return ret;
}

static struct starpu_task * pop_task(struct _starpu_sched_node * node, unsigned sched_ctx_id)
{
	struct _starpu_fifo_data * data = node->data;
	struct _starpu_fifo_taskq * fifo = data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	STARPU_PTHREAD_MUTEX_LOCK(mutex);
	STARPU_ASSERT(!isnan(fifo->exp_end));
	STARPU_ASSERT(!isnan(fifo->exp_len));
	STARPU_ASSERT(!isnan(fifo->exp_start));
	struct starpu_task * task  = _starpu_fifo_pop_task(fifo, starpu_worker_get_id());
	if(task)
	{
		fifo->exp_start = starpu_timing_now();
		STARPU_ASSERT(node->nworkers > 0);
		if(!isnan(task->predicted))
			fifo->exp_len -= task->predicted/node->nworkers;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;
	}
	STARPU_ASSERT(!isnan(fifo->exp_end));
	STARPU_ASSERT(!isnan(fifo->exp_len));
	STARPU_ASSERT(!isnan(fifo->exp_start));
	STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
	if(task)
		return task;
	struct _starpu_sched_node * father = node->fathers[sched_ctx_id];
	if(father)
		return father->pop_task(father,sched_ctx_id);
	return NULL;
}

struct starpu_task_list  _starpu_sched_node_fifo_get_non_executable_tasks(struct _starpu_sched_node * node)
{
	struct starpu_task_list list;
	starpu_task_list_init(&list);
	struct _starpu_fifo_data * data = node->data;
	struct _starpu_fifo_taskq * fifo = data->fifo;
	struct starpu_task * task;
	for (task  = starpu_task_list_begin(&fifo->taskq);
	     task != starpu_task_list_end(&fifo->taskq);
	     task  = starpu_task_list_next(task))
	{
		STARPU_ASSERT(task);
		if(!_starpu_sched_node_can_execute_task(node, task))
		{
			starpu_task_list_erase(&fifo->taskq, task);
			starpu_task_list_push_front(&list, task);
			fifo->ntasks--;
		}
	}
	return list;
}

int _starpu_sched_node_is_fifo(struct _starpu_sched_node * node)
{
	return node->estimated_execute_preds == estimated_execute_preds
		|| node->estimated_load == estimated_load
		|| node->push_task == push_task
		|| node->pop_task == pop_task;
}

struct _starpu_sched_node * _starpu_sched_node_fifo_create(void)
{
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	struct _starpu_fifo_data * data = malloc(sizeof(*data));
	data->fifo = _starpu_create_fifo();
	STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
	node->data = data;
	node->estimated_execute_preds = estimated_execute_preds;
	node->estimated_load = estimated_load;
	node->push_task = push_task;
	node->pop_task = pop_task;
	return node;
}
