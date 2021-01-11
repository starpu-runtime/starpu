/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Simon Archipoff
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

#include <starpu_sched_component.h>
#include <starpu_scheduler.h>

#include "fifo_queues.h"

struct _starpu_fifo_data
{
	struct _starpu_fifo_taskq * fifo;
	starpu_pthread_mutex_t mutex;
	unsigned ntasks_threshold;
	double exp_len_threshold;
};

static void fifo_component_deinit_data(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component && component->data);
	struct _starpu_fifo_data * f = component->data;
	_starpu_destroy_fifo(f->fifo);
	STARPU_PTHREAD_MUTEX_DESTROY(&f->mutex);
	free(f);
}

static double fifo_estimated_end(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component && component->data);
	struct _starpu_fifo_data * data = component->data;
	struct _starpu_fifo_taskq * fifo = data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	int card = starpu_bitmap_cardinal(component->workers_in_ctx);
	STARPU_ASSERT(card != 0);
	double estimated_end = starpu_sched_component_estimated_end_average(component);
	STARPU_PTHREAD_MUTEX_LOCK(mutex);
	fifo->exp_start = STARPU_MAX(fifo->exp_start, starpu_timing_now());
	estimated_end += fifo->exp_start + fifo->exp_len / card;
	STARPU_PTHREAD_MUTEX_UNLOCK(mutex);

	return estimated_end;
}

static double fifo_estimated_load(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component && component->data);
	STARPU_ASSERT(starpu_bitmap_cardinal(component->workers_in_ctx) != 0);
	struct _starpu_fifo_data * data = component->data;
	struct _starpu_fifo_taskq * fifo = data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	double relative_speedup = 0.0;
	double load = starpu_sched_component_estimated_load(component);
	if(STARPU_SCHED_COMPONENT_IS_HOMOGENEOUS(component))
	{
		int first_worker = starpu_bitmap_first(component->workers_in_ctx);
		relative_speedup = starpu_worker_get_relative_speedup(starpu_worker_get_perf_archtype(first_worker, component->tree->sched_ctx_id));
		STARPU_PTHREAD_MUTEX_LOCK(mutex);
		load += fifo->ntasks / relative_speedup;
		STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
		return load;
	}
	else
	{
		int i;
		for(i = starpu_bitmap_first(component->workers_in_ctx);
		    i != -1;
		    i = starpu_bitmap_next(component->workers_in_ctx, i))
			relative_speedup += starpu_worker_get_relative_speedup(starpu_worker_get_perf_archtype(i, component->tree->sched_ctx_id));
		relative_speedup /= starpu_bitmap_cardinal(component->workers_in_ctx);
		STARPU_ASSERT(!_STARPU_IS_ZERO(relative_speedup));
		STARPU_PTHREAD_MUTEX_LOCK(mutex);
		load += fifo->ntasks / relative_speedup;
		STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
	}
	return load;
}

static int fifo_push_local_task(struct starpu_sched_component * component, struct starpu_task * task, unsigned is_pushback)
{
	STARPU_ASSERT(component && component->data && task);
	STARPU_ASSERT(starpu_sched_component_can_execute_task(component,task));
	struct _starpu_fifo_data * data = component->data;
	struct _starpu_fifo_taskq * fifo = data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	int ret = 0;

	STARPU_PTHREAD_MUTEX_LOCK(mutex);
	double exp_len;
	if(!isnan(task->predicted))
		exp_len = fifo->exp_len + task->predicted;
	else
		exp_len = fifo->exp_len;

	if((data->ntasks_threshold != 0) && (data->exp_len_threshold != 0.0) && 
			((fifo->ntasks >= data->ntasks_threshold) || (exp_len >= data->exp_len_threshold)))
	{
		static int warned;
		if(task->predicted > data->exp_len_threshold && !warned)
		{
			_STARPU_DISP("Warning : a predicted task length (%lf) exceeds the expected length threshold (%lf) of a prio component queue, you should reconsider the value of this threshold. This message will not be printed again for further thresholds exceeding.\n",task->predicted,data->exp_len_threshold);
			warned = 1;
		}
		STARPU_ASSERT(!is_pushback);
		ret = 1;
		STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
	}
	else
	{
		if(is_pushback)
			ret = _starpu_fifo_push_back_task(fifo,task);
		else
		{
			ret = _starpu_fifo_push_task(fifo,task);
			starpu_sched_component_prefetch_on_node(component, task);
		}

		if(!isnan(task->predicted))
		{
			fifo->exp_len = exp_len;
			fifo->exp_end = fifo->exp_start + fifo->exp_len;
		}
		STARPU_ASSERT(!isnan(fifo->exp_end));
		STARPU_ASSERT(!isnan(fifo->exp_len));
		STARPU_ASSERT(!isnan(fifo->exp_start));
		
		if(!is_pushback)
			component->can_pull(component);
		STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
	}
	
	return ret;
}

static int fifo_push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	return fifo_push_local_task(component, task, 0);
}

static struct starpu_task * fifo_pull_task(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component && component->data);
	struct _starpu_fifo_data * data = component->data;
	struct _starpu_fifo_taskq * fifo = data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	STARPU_PTHREAD_MUTEX_LOCK(mutex);
	struct starpu_task * task = _starpu_fifo_pop_task(fifo, starpu_worker_get_id_check());
	if(task)
	{
		if(!isnan(task->predicted))
		{
			fifo->exp_start = starpu_timing_now() + task->predicted;
			fifo->exp_len -= task->predicted;
		}
		fifo->exp_end = fifo->exp_start + fifo->exp_len;
		if(fifo->ntasks == 0)
			fifo->exp_len = 0.0;
	}
	STARPU_ASSERT(!isnan(fifo->exp_end));
	STARPU_ASSERT(!isnan(fifo->exp_len));
	STARPU_ASSERT(!isnan(fifo->exp_start));
	STARPU_PTHREAD_MUTEX_UNLOCK(mutex);

	// When a pop is called, a can_push is called for pushing tasks onto
	// the empty place of the queue left by the popped task.
	int i,ret;
	for(i=0; i < component->nparents; i++)
	{
		if(component->parents[i] == NULL)
			continue;
		else
		{
			ret = component->parents[i]->can_push(component->parents[i]);
			if(ret)
				break;
		}
	}
	
	if(task)
		return task;

	return NULL;
}

/* When a can_push is caught by this function, we try to pop and push
 * tasks from our local queue as much as possible, until a
 * push fails, which means that the worker fifo_components are
 * currently "full".
 */
static int fifo_can_push(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component && starpu_sched_component_is_fifo(component));
	int ret = 0;
	int res = 0;

	STARPU_ASSERT(component->nchildren == 1);
	struct starpu_sched_component * child = component->children[0];

	struct starpu_task * task = fifo_pull_task(component);
	if(task)
		ret = starpu_sched_component_push_task(NULL,child,task);	
	while(task && !ret) 
	{
		if(!res)
			res = 1;

		task = fifo_pull_task(component);
		if(task)
			ret = starpu_sched_component_push_task(NULL,child,task);	
	}
	if(task && ret)
		fifo_push_local_task(component,task,1); 

	return res;
}

int starpu_sched_component_is_fifo(struct starpu_sched_component * component)
{
	return component->push_task == fifo_push_task;
}

struct starpu_sched_component * starpu_sched_component_fifo_create(struct starpu_sched_tree *tree, struct starpu_sched_component_fifo_data * params)
{
	struct starpu_sched_component * component = starpu_sched_component_create(tree, "fifo");
	struct _starpu_fifo_data * data;
	_STARPU_MALLOC(data, sizeof(*data));
	data->fifo = _starpu_create_fifo();
	STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
	component->data = data;
	component->estimated_end = fifo_estimated_end;
	component->estimated_load = fifo_estimated_load;
	component->push_task = fifo_push_task;
	component->pull_task = fifo_pull_task;
	component->can_push = fifo_can_push;
	component->deinit_data = fifo_component_deinit_data;

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

	return component;
}
