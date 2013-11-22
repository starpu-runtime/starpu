/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Marc Sergent
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu_sched_node.h>
#include <starpu_scheduler.h>

#include "prio_deque.h"

#define STARPU_TRACE_SCHED_NODE_PUSH_PRIO(node,ntasks,exp_len) do {                                 \
	int workerid = STARPU_NMAXWORKERS + 1;									\
	if((node->nchilds == 1) && starpu_sched_node_is_worker(node->childs[0])) \
		workerid = starpu_sched_node_worker_get_workerid(node->childs[0]); \
	_STARPU_TRACE_SCHED_NODE_PUSH_PRIO(workerid, ntasks, exp_len); \
} while (0)

#define STARPU_TRACE_SCHED_NODE_POP_PRIO(node,ntasks,exp_len) do {                                 \
	int workerid = STARPU_NMAXWORKERS + 1;									\
	if((node->nchilds == 1) && starpu_sched_node_is_worker(node->childs[0])) \
		workerid = starpu_sched_node_worker_get_workerid(node->childs[0]); \
	_STARPU_TRACE_SCHED_NODE_POP_PRIO(workerid, ntasks, exp_len); \
} while (0)


struct _starpu_prio_data
{
	struct _starpu_prio_deque prio;
	starpu_pthread_mutex_t mutex;
	unsigned ntasks_threshold;
	double exp_len_threshold;
};

void prio_node_deinit_data(struct starpu_sched_node * node)
{
	STARPU_ASSERT(node && node->data);
	struct _starpu_prio_data * f = node->data;
	_starpu_prio_deque_destroy(&f->prio);
	STARPU_PTHREAD_MUTEX_DESTROY(&f->mutex);
	free(f);
}

static double prio_estimated_end(struct starpu_sched_node * node)
{
	STARPU_ASSERT(node && node->data);
	struct _starpu_prio_data * data = node->data;
	struct _starpu_prio_deque * prio = &data->prio;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	int card = starpu_bitmap_cardinal(node->workers_in_ctx);
	STARPU_ASSERT(card != 0);
	STARPU_PTHREAD_MUTEX_LOCK(mutex);
	prio->exp_start = STARPU_MAX(prio->exp_start, starpu_timing_now());
	double estimated_end = prio->exp_start + prio->exp_len / card;
	STARPU_PTHREAD_MUTEX_UNLOCK(mutex);

	return estimated_end;
}

static double prio_estimated_load(struct starpu_sched_node * node)
{
	STARPU_ASSERT(node && node->data);
	STARPU_ASSERT(starpu_bitmap_cardinal(node->workers_in_ctx) != 0);
	struct _starpu_prio_data * data = node->data;
	struct _starpu_prio_deque * prio = &data->prio;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	double relative_speedup = 0.0;
	double load;
	if(STARPU_SCHED_NODE_IS_HOMOGENEOUS(node))
	{		
		int first_worker = starpu_bitmap_first(node->workers_in_ctx);
		relative_speedup = starpu_worker_get_relative_speedup(starpu_worker_get_perf_archtype(first_worker));
		STARPU_PTHREAD_MUTEX_LOCK(mutex);
		load = prio->ntasks / relative_speedup;
		STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
		return load;
	}
	else
	{
		int i;
		for(i = starpu_bitmap_first(node->workers_in_ctx);
		    i != -1;
		    i = starpu_bitmap_next(node->workers_in_ctx, i))
			relative_speedup += starpu_worker_get_relative_speedup(starpu_worker_get_perf_archtype(i));
		relative_speedup /= starpu_bitmap_cardinal(node->workers_in_ctx);
		STARPU_ASSERT(!_STARPU_IS_ZERO(relative_speedup));
		STARPU_PTHREAD_MUTEX_LOCK(mutex);
		load = prio->ntasks / relative_speedup;
		STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
	}
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		struct starpu_sched_node * c = node->childs[i];
		load += c->estimated_load(c);
	}
	return load;
}

static int prio_push_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(node && node->data && task);
	STARPU_ASSERT(starpu_sched_node_can_execute_task(node,task));
	struct _starpu_prio_data * data = node->data;
	struct _starpu_prio_deque * prio = &data->prio;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	int ret;
	
	STARPU_ASSERT(node->nchilds == 1);
	struct starpu_sched_node * child = node->childs[0];
	if(starpu_sched_node_is_worker(child))
		ret = 1;
	else
		ret = child->push_task(child,task);

	if(ret)
	{
		STARPU_PTHREAD_MUTEX_LOCK(mutex);
		double exp_len;
		if(!isnan(task->predicted))
			exp_len = prio->exp_len + task->predicted;
		else
			exp_len = prio->exp_len;

		if((data->ntasks_threshold != 0) && (data->exp_len_threshold != 0.0) && 
				((prio->ntasks >= data->ntasks_threshold) || (exp_len >= data->exp_len_threshold)))
		{
			static int warned;
			if(task->predicted > data->exp_len_threshold && !warned)
			{
				_STARPU_DISP("Warning : a predicted task length (%lf) exceeds the expected length threshold (%lf) of a prio node queue, you should reconsider the value of this threshold. This message will not be printed again for further thresholds exceeding.\n",task->predicted,data->exp_len_threshold);
				warned = 1;
			}
			ret = 1;
			STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
		}
		else
		{
			starpu_sched_node_prefetch_on_node(node, task);
			ret = _starpu_prio_deque_push_task(prio,task);
			STARPU_TRACE_SCHED_NODE_PUSH_PRIO(node, prio->ntasks, exp_len);
			if(!isnan(task->predicted))
			{
				prio->exp_len = exp_len;
				prio->exp_end = prio->exp_start + prio->exp_len;
			}
			STARPU_ASSERT(!isnan(prio->exp_end));
			STARPU_ASSERT(!isnan(prio->exp_len));
			STARPU_ASSERT(!isnan(prio->exp_start));
			STARPU_PTHREAD_MUTEX_UNLOCK(mutex);

			// When a task is pushed onto the local queue, we signify to our children
			// that a task has been pushed, and that if everyone is sleeping, someone
			// needs to wake up to come and take it.
			node->avail(node);
		}
	}
	return ret;
}

static int prio_push_back_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(node && node->data && task);
	STARPU_ASSERT(starpu_sched_node_can_execute_task(node,task));
	struct _starpu_prio_data * data = node->data;
	struct _starpu_prio_deque * prio = &data->prio;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	int ret;

	STARPU_ASSERT(node->nchilds == 1);
	struct starpu_sched_node * child = node->childs[0];
	if(starpu_sched_node_is_worker(child))
		ret = 1;
	else
		ret = child->push_task(child,task);

	if(ret)
	{
		STARPU_PTHREAD_MUTEX_LOCK(mutex);
		double exp_len;
		if(!isnan(task->predicted))
			exp_len = prio->exp_len + task->predicted;
		else
			exp_len = prio->exp_len;

		_starpu_prio_deque_push_back_task(prio,task);
		STARPU_TRACE_SCHED_NODE_PUSH_PRIO(node, prio->ntasks, exp_len);
		if(!isnan(task->predicted))
		{
			prio->exp_len = exp_len;
			prio->exp_end = prio->exp_start + exp_len;
		}
		STARPU_ASSERT(!isnan(prio->exp_end));
		STARPU_ASSERT(!isnan(prio->exp_len));
		STARPU_ASSERT(!isnan(prio->exp_start));
		STARPU_PTHREAD_MUTEX_UNLOCK(mutex);

		// When a task is pushed onto the local queue, we signify to our children
		// that a task has been pushed, and that if everyone is sleeping, someone
		// needs to wake up to come and take it.
		node->avail(node);
	}
	return ret;
}

int starpu_sched_node_is_prio(struct starpu_sched_node * node)
{
	return node->push_task == prio_push_task;
}

static struct starpu_task * prio_pop_task(struct starpu_sched_node * node, unsigned sched_ctx_id)
{
	STARPU_ASSERT(node && node->data);
	struct _starpu_prio_data * data = node->data;
	struct _starpu_prio_deque * prio = &data->prio;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	STARPU_PTHREAD_MUTEX_LOCK(mutex);
	struct starpu_task * task = _starpu_prio_deque_pop_task(prio);
	if(task)
	{
		if(!isnan(task->predicted))
		{
			prio->exp_start = starpu_timing_now() + task->predicted;
			prio->exp_len -= task->predicted;
		}
		prio->exp_end = prio->exp_start + prio->exp_len;
		if(prio->ntasks == 0)
			prio->exp_len = 0.0;
		
		STARPU_TRACE_SCHED_NODE_POP_PRIO(node, prio->ntasks, prio->exp_len);
	}
	STARPU_ASSERT(!isnan(prio->exp_end));
	STARPU_ASSERT(!isnan(prio->exp_len));
	STARPU_ASSERT(!isnan(prio->exp_start));
	STARPU_PTHREAD_MUTEX_UNLOCK(mutex);

	// When a pop is called, a room is called for pushing tasks onto
	// the empty place of the queue left by the popped task.
	struct starpu_sched_node * father = node->fathers[sched_ctx_id];
	if(father != NULL)
		father->room(father, sched_ctx_id);
	
	if(task)
		return task;

	return NULL;
}

/* When a room is caught by this function, we try to pop and push
 * tasks from our local queue as much as possible, until a
 * push fails, which means that the worker prio_nodes are
 * currently "full".
 */
static void prio_room(struct starpu_sched_node * node, unsigned sched_ctx_id)
{
	STARPU_ASSERT(node && starpu_sched_node_is_prio(node));
	int ret = 0;

	struct starpu_task * task = node->pop_task(node, sched_ctx_id);
	if(task)
		ret = node->push_back_task(node,task);	
	while(task && !ret) 
	{
		task = node->pop_task(node, sched_ctx_id);
		if(task)
			ret = node->push_back_task(node,task);	
	} 
}

struct starpu_sched_node * starpu_sched_node_prio_create(struct starpu_prio_data * params)
{
	struct starpu_sched_node * node = starpu_sched_node_create();
	struct _starpu_prio_data * data = malloc(sizeof(*data));
	_starpu_prio_deque_init(&data->prio);
	STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
	node->data = data;
	node->estimated_end = prio_estimated_end;
	node->estimated_load = prio_estimated_load;
	node->push_task = prio_push_task;
	node->push_back_task = prio_push_back_task;
	node->pop_task = prio_pop_task;
	node->room = prio_room;
	node->deinit_data = prio_node_deinit_data;

	if(params)
	{
		data->ntasks_threshold=params->ntasks_threshold;
		data->exp_len_threshold=params->exp_len_threshold;
	}
	else
	{
		data->ntasks_threshold=0;
		data->exp_len_threshold=0.0;
	}

	return node;
}
