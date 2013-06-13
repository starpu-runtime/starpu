#include "node_sched.h"
#include "prio_deque.h"
#include <starpu_scheduler.h>


struct _starpu_fifo_data
{
	struct _starpu_prio_deque fifo;
	starpu_pthread_mutex_t mutex;
};


static struct _starpu_task_execute_preds estimated_execute_preds(struct _starpu_sched_node * node,
								 struct starpu_task * task)
{
	struct _starpu_fifo_data * data = node->data;
	struct _starpu_prio_deque * fifo = &data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	if(node->nchilds == 0)
	{
		struct _starpu_task_execute_preds p = { CANNOT_EXECUTE };
		return p;
	}
	
	if(!node->is_homogeneous)
	{
		struct _starpu_task_execute_preds preds = _starpu_sched_node_average_estimated_execute_preds(node, task);
		STARPU_PTHREAD_MUTEX_LOCK(mutex);
		preds.expected_finish_time = _starpu_compute_expected_time(fifo->exp_start,
									   preds.expected_finish_time + fifo->exp_len / _starpu_bitmap_cardinal(node->workers),
									   preds.state == PERF_MODEL ? preds.expected_length + fifo->exp_len : fifo->exp_len,
									   preds.expected_transfer_length);
		STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
		return preds;
	}
	
	struct _starpu_task_execute_preds preds = node->childs[0]->estimated_execute_preds(node->childs[0],task);

	if(preds.state == PERF_MODEL)
	{
		STARPU_PTHREAD_MUTEX_LOCK(mutex);
		preds.expected_finish_time = _starpu_compute_expected_time(fifo->exp_start,
									   preds.expected_finish_time + fifo->exp_len / _starpu_bitmap_cardinal(node->workers),
									   preds.expected_length + fifo->exp_len,
									   preds.expected_transfer_length);
		STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
	}
	return preds;
}

static double estimated_load(struct _starpu_sched_node * node)
{
	struct _starpu_fifo_data * data = node->data;
	struct _starpu_prio_deque * fifo = &data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	double relative_speedup = 0.0;
	double load;

	if(node->is_homogeneous)
	{
		relative_speedup = starpu_worker_get_relative_speedup(starpu_worker_get_perf_archtype(_starpu_bitmap_first(node->workers)));
		STARPU_PTHREAD_MUTEX_LOCK(mutex);
		load = fifo->ntasks / relative_speedup;
		STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
		return load;
	}
	else
	{
		int i;
		for(i = _starpu_bitmap_first(node->workers);
		    i != -1;
		    i = _starpu_bitmap_next(node->workers, i))
			relative_speedup += starpu_worker_get_relative_speedup(starpu_worker_get_perf_archtype(i));
		relative_speedup /= _starpu_bitmap_cardinal(node->workers);
			STARPU_ASSERT(!_STARPU_IS_ZERO(relative_speedup));
			STARPU_PTHREAD_MUTEX_LOCK(mutex);
			load = fifo->ntasks / relative_speedup;
			STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
	}
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		struct _starpu_sched_node * c = node->childs[i];
		load += c->estimated_load(c);
	}
	return load;
}

static int push_task(struct _starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(_starpu_sched_node_can_execute_task(node,task));
	struct _starpu_fifo_data * data = node->data;
	struct _starpu_prio_deque * fifo = &data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	STARPU_PTHREAD_MUTEX_LOCK(mutex);
	STARPU_ASSERT(!isnan(fifo->exp_end));
	STARPU_ASSERT(!isnan(fifo->exp_len));
	STARPU_ASSERT(!isnan(fifo->exp_start));
	int ret = _starpu_prio_deque_push_task(fifo, task);
	if(!isnan(task->predicted))
	{
//		task->predicted /= _starpu_bitmap_cardinal(node->workers);
		fifo->exp_len += task->predicted;
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
	struct _starpu_prio_deque * fifo = &data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	STARPU_PTHREAD_MUTEX_LOCK(mutex);
	STARPU_ASSERT(!isnan(fifo->exp_end));
	STARPU_ASSERT(!isnan(fifo->exp_len));
	STARPU_ASSERT(!isnan(fifo->exp_start));
	struct starpu_task * task  = node->is_homogeneous ?
		_starpu_prio_deque_pop_task(fifo):
		_starpu_prio_deque_pop_task_for_worker(fifo, starpu_worker_get_id());
	if(task)
	{
		fifo->exp_start = starpu_timing_now();
		if(!isnan(task->predicted))
			fifo->exp_len -= task->predicted;
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
/*
struct starpu_task_list  _starpu_sched_node_fifo_get_non_executable_tasks(struct _starpu_sched_node * node)
{
	struct starpu_task_list list;
	starpu_task_list_init(&list);
	struct _starpu_fifo_data * data = node->data;
	struct _starpu_prio_deque * fifo = data->fifo;
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
*/
void init_fifo_data(struct _starpu_sched_node * node)
{
	STARPU_ASSERT(_starpu_sched_node_is_fifo(node));
	struct _starpu_fifo_data * data = malloc(sizeof(*data));
	_starpu_prio_deque_init(&data->fifo);
	STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
	node->data = data;
}
void deinit_fifo_data(struct _starpu_sched_node * node)
{
	struct _starpu_fifo_data * data = node->data;
	STARPU_PTHREAD_MUTEX_DESTROY(&data->mutex);
	_starpu_prio_deque_destroy(&data->fifo);
	free(data);
}


int _starpu_sched_node_is_fifo(struct _starpu_sched_node * node)
{
	return node->init_data == init_fifo_data;
}

struct _starpu_sched_node * _starpu_sched_node_fifo_create(void * arg STARPU_ATTRIBUTE_UNUSED)
{
	struct _starpu_sched_node * node = _starpu_sched_node_create();
	node->estimated_execute_preds = estimated_execute_preds;
	node->estimated_load = estimated_load;
	node->init_data = init_fifo_data;
	node->deinit_data = deinit_fifo_data;
	node->push_task = push_task;
	node->pop_task = pop_task;
	return node;
}
