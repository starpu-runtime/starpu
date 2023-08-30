/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <schedulers/starpu_scheduler_toolbox.h>

#include <core/workers.h>
#include <sched_policies/fifo_queues.h>

struct _starpu_fifo_data
{
	struct starpu_st_fifo_taskq fifo;
	starpu_pthread_mutex_t mutex;
	unsigned ntasks_threshold;
	double exp_len_threshold;
	int ready;
	int exp;
	int ready_first;
};

static void fifo_component_deinit_data(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component && component->data);
	struct _starpu_fifo_data * f = component->data;
	STARPU_PTHREAD_MUTEX_DESTROY(&f->mutex);
	free(f);
}

static double fifo_estimated_end(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component && component->data);
	struct _starpu_fifo_data * data = component->data;
	struct starpu_st_fifo_taskq * queue = &data->fifo;
	return starpu_sched_component_estimated_end_min_add(component, queue->exp_len);
}

static double fifo_estimated_load(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component && component->data);
	STARPU_ASSERT(starpu_bitmap_cardinal(&component->workers_in_ctx) != 0);
	struct _starpu_fifo_data * data = component->data;
	struct starpu_st_fifo_taskq * queue = &data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	double relative_speedup = 0.0;
	double load = starpu_sched_component_estimated_load(component);
	if(STARPU_SCHED_COMPONENT_IS_HOMOGENEOUS(component))
	{
		int first_worker = starpu_bitmap_first(&component->workers_in_ctx);
		relative_speedup = starpu_worker_get_relative_speedup(starpu_worker_get_perf_archtype(first_worker, component->tree->sched_ctx_id));
		STARPU_COMPONENT_MUTEX_LOCK(mutex);
		load += queue->ntasks / relative_speedup;
		STARPU_COMPONENT_MUTEX_UNLOCK(mutex);
		return load;
	}
	else
	{
		int i;
		for(i = starpu_bitmap_first(&component->workers_in_ctx);
		    i != -1;
		    i = starpu_bitmap_next(&component->workers_in_ctx, i))
			relative_speedup += starpu_worker_get_relative_speedup(starpu_worker_get_perf_archtype(i, component->tree->sched_ctx_id));
		relative_speedup /= starpu_bitmap_cardinal(&component->workers_in_ctx);
		STARPU_ASSERT(!_STARPU_IS_ZERO(relative_speedup));
		STARPU_COMPONENT_MUTEX_LOCK(mutex);
		load += queue->ntasks / relative_speedup;
		STARPU_COMPONENT_MUTEX_UNLOCK(mutex);
	}
	return load;
}

static int fifo_push_local_task(struct starpu_sched_component * component, struct starpu_task * task, unsigned is_pushback)
{
	STARPU_ASSERT(component && component->data && task);
	STARPU_ASSERT(starpu_sched_component_can_execute_task(component,task));
	struct _starpu_fifo_data * data = component->data;
	struct starpu_st_fifo_taskq * queue = &data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	int ret = 0;
	const double now = starpu_timing_now();
	STARPU_COMPONENT_MUTEX_LOCK(mutex);

	if (!is_pushback && data->ntasks_threshold != 0 && queue->ntasks >= data->ntasks_threshold)
	{
		ret = 1;
		STARPU_COMPONENT_MUTEX_UNLOCK(mutex);
	}
	else if(data->exp)
	{
		double exp_len;
		if(!isnan(task->predicted))
			exp_len = queue->exp_len + task->predicted;
		else
			exp_len = queue->exp_len;

		if (!is_pushback && data->exp_len_threshold != 0.0 && exp_len >= data->exp_len_threshold)
		{
			static int warned;
			STARPU_HG_DISABLE_CHECKING(warned);
			if(data->exp_len_threshold != 0.0 && task->predicted > data->exp_len_threshold && !warned)
			{
				_STARPU_DISP("Warning : a predicted task length (%lf) exceeds the expected length threshold (%lf) of a prio component queue, you should reconsider the value of this threshold. This message will not be printed again for further thresholds exceeding.\n",task->predicted,data->exp_len_threshold);
				warned = 1;
			}
			ret = 1;
			STARPU_COMPONENT_MUTEX_UNLOCK(mutex);
		}
		else
		{
			if(!isnan(task->predicted_transfer))
			{
				double end = fifo_estimated_end(component);
				double tfer_end = now + task->predicted_transfer;
				/* FIXME: We don't have overlap when running CPU-CPU transfers */
				if(tfer_end < end)
					task->predicted_transfer = 0.0;
				else
					task->predicted_transfer = tfer_end - end;
				exp_len += task->predicted_transfer;
			}

			if(!isnan(task->predicted))
			{
				queue->exp_len = exp_len;
				queue->exp_end = queue->exp_start + queue->exp_len;
			}
			STARPU_ASSERT(!isnan(queue->exp_end));
			STARPU_ASSERT(!isnan(queue->exp_len));
			STARPU_ASSERT(!isnan(queue->exp_start));
		}
	}

	if(!ret)
	{
		if(is_pushback)
			ret = starpu_st_fifo_taskq_push_back_task(queue,task);
		else
		{
			ret = starpu_st_fifo_taskq_push_task(queue,task);
			starpu_sched_component_prefetch_on_node(component, task);
		}
		STARPU_COMPONENT_MUTEX_UNLOCK(mutex);
		if(!is_pushback)
			component->can_pull(component);
	}

	return ret;
}

static int fifo_push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	return fifo_push_local_task(component, task, 0);
}

static struct starpu_task * fifo_pull_task(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	STARPU_ASSERT(component && component->data);
	struct _starpu_fifo_data * data = component->data;
	struct starpu_st_fifo_taskq * queue = &data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	const double now = starpu_timing_now();

	if (!STARPU_RUNNING_ON_VALGRIND && starpu_st_fifo_taskq_empty(queue))
	{
		starpu_sched_component_send_can_push_to_parents(component);
		return NULL;
	}

	STARPU_COMPONENT_MUTEX_LOCK(mutex);
	struct starpu_task * task;
	if (data->ready && to->properties & STARPU_SCHED_COMPONENT_SINGLE_MEMORY_NODE)
#ifdef STARPU_DEVEL
#warning In eager schedulers, we never write that we want to fill the fifo before picking up a task. Eager is then ineffective since in practice the fifo will not fill
#endif
	{
		if (data->ready_first)
			task = starpu_st_fifo_taskq_pop_first_ready_task_ready_first(queue, starpu_bitmap_first(&to->workers_in_ctx), -1);
		else
			task = starpu_st_fifo_taskq_pop_first_ready_task(queue, starpu_bitmap_first(&to->workers_in_ctx), -1);
	}
	else if (to->properties & STARPU_SCHED_COMPONENT_HOMOGENEOUS)
		task = starpu_st_fifo_taskq_pop_task(queue, starpu_bitmap_first(&to->workers_in_ctx));
	else
		task = starpu_st_fifo_taskq_pop_task(queue, -1);
	if(task && data->exp)
	{
		if(!isnan(task->predicted))
		{
			const double exp_len = queue->exp_len - task->predicted;
			queue->exp_start = now + task->predicted;
			if (exp_len >= 0.0)
			{
				queue->exp_len = exp_len;
			}
			else
			{
				/* exp_len can become negative due to rounding errors */
				queue->exp_len = 0.0;
			}
		}

		STARPU_ASSERT_MSG(queue->exp_len>=0, "fifo->exp_len=%lf\n",queue->exp_len);
		if(!isnan(task->predicted_transfer))
		{
			if (queue->exp_len > task->predicted_transfer)
			{
				queue->exp_start += task->predicted_transfer;
				queue->exp_len -= task->predicted_transfer;
			}
			else
			{
				queue->exp_start += queue->exp_len;
				queue->exp_len = 0;
			}
		}

		queue->exp_end = queue->exp_start + queue->exp_len;
		if(queue->ntasks == 0)
			queue->exp_len = 0.0;
	}
	STARPU_ASSERT(!isnan(queue->exp_end));
	STARPU_ASSERT(!isnan(queue->exp_len));
	STARPU_ASSERT(!isnan(queue->exp_start));
	STARPU_COMPONENT_MUTEX_UNLOCK(mutex);

	// When a pop is called, a can_push is called for pushing tasks onto
	// the empty place of the queue left by the popped task.

	starpu_sched_component_send_can_push_to_parents(component);

	if(task)
		return task;

	return NULL;
}

/* When a can_push is caught by this function, we try to pop and push
 * tasks from our local queue as much as possible, until a
 * push fails, which means that the worker fifo_components are
 * currently "full".
 */
static int fifo_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to STARPU_ATTRIBUTE_UNUSED)
{
	STARPU_ASSERT(component && starpu_sched_component_is_fifo(component));
	int res = 0;
	struct starpu_task * task;

	task = starpu_sched_component_pump_downstream(component, &res);

	if(task)
	{
		int ret = fifo_push_local_task(component,task,1);
		STARPU_ASSERT(!ret);
	}

	return res;
}

int starpu_sched_component_is_fifo(struct starpu_sched_component * component)
{
	return component->push_task == fifo_push_task;
}

struct starpu_sched_component *starpu_sched_component_fifo_create(struct starpu_sched_tree *tree, struct starpu_sched_component_fifo_data *params)
{
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "fifo");
	struct _starpu_fifo_data *data;
	_STARPU_MALLOC(data, sizeof(*data));
	starpu_st_fifo_taskq_init(&data->fifo);
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
		data->ready=params->ready;
		data->exp=params->exp;
		data->ready_first=params->ready_first;
	}
	else
	{
		data->ntasks_threshold=0;
		data->exp_len_threshold=0.0;
		data->ready=0;
		data->exp=0;
		data->ready_first=0;
	}

	return component;
}
