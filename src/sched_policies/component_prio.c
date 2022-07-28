/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <common/fxt.h>
#include <core/workers.h>
#include <sched_policies/prio_deque.h>

#ifdef STARPU_USE_FXT
#define STARPU_TRACE_SCHED_COMPONENT_PUSH_PRIO(component,ntasks,exp_len) do {                                 \
	if (fut_active) {					\
		int workerid = STARPU_NMAXWORKERS + 1;			\
		if((component->nchildren == 1) && starpu_sched_component_is_worker(component->children[0])) \
			workerid = starpu_sched_component_worker_get_workerid(component->children[0]); \
		_STARPU_TRACE_SCHED_COMPONENT_PUSH_PRIO(workerid, ntasks, exp_len); \
	}								\
} while (0)

#define STARPU_TRACE_SCHED_COMPONENT_POP_PRIO(component,ntasks,exp_len) do {                                 \
	if (fut_active) { \
		int workerid = STARPU_NMAXWORKERS + 1;			\
		if((component->nchildren == 1) && starpu_sched_component_is_worker(component->children[0])) \
			workerid = starpu_sched_component_worker_get_workerid(component->children[0]); \
		_STARPU_TRACE_SCHED_COMPONENT_POP_PRIO(workerid, ntasks, exp_len); \
	}								\
} while (0)
#else
#define STARPU_TRACE_SCHED_COMPONENT_PUSH_PRIO(component,ntasks,exp_len) do { } while (0)
#define STARPU_TRACE_SCHED_COMPONENT_POP_PRIO(component,ntasks,exp_len) do { } while (0)
#endif

struct _starpu_prio_data
{
	struct starpu_st_prio_deque prio;
	starpu_pthread_mutex_t mutex;
	unsigned ntasks_threshold;
	double exp_len_threshold;
	int ready;
	int exp;
};

static void prio_component_deinit_data(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component && component->data);
	struct _starpu_prio_data * f = component->data;
	starpu_st_prio_deque_destroy(&f->prio);
	STARPU_PTHREAD_MUTEX_DESTROY(&f->mutex);
	free(f);
}

static double prio_estimated_end(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component && component->data);
	struct _starpu_prio_data * data = component->data;
	struct starpu_st_prio_deque * queue = &data->prio;
	return starpu_sched_component_estimated_end_min_add(component, queue->exp_len);
}

static double prio_estimated_load(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component && component->data);
	STARPU_ASSERT(starpu_bitmap_cardinal(&component->workers_in_ctx) != 0);
	struct _starpu_prio_data * data = component->data;
	struct starpu_st_prio_deque * queue = &data->prio;
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

static int prio_push_local_task(struct starpu_sched_component * component, struct starpu_task * task, unsigned is_pushback)
{
	STARPU_ASSERT(component && component->data && task);
	STARPU_ASSERT(starpu_sched_component_can_execute_task(component,task));
	struct _starpu_prio_data * data = component->data;
	struct starpu_st_prio_deque * queue = &data->prio;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	int ret = 0;
	const double now = starpu_timing_now();
	STARPU_COMPONENT_MUTEX_LOCK(mutex);

	double exp_len = NAN;

	if (!is_pushback && data->ntasks_threshold != 0 && queue->ntasks >= data->ntasks_threshold)
	{
		ret = 1;
		STARPU_COMPONENT_MUTEX_UNLOCK(mutex);
	}
	else if(data->exp)
	{
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
				double end = prio_estimated_end(component);
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
			ret = starpu_st_prio_deque_push_front_task(queue,task);
		else
		{
			ret = starpu_st_prio_deque_push_back_task(queue,task);
			starpu_sched_component_prefetch_on_node(component, task);
			STARPU_TRACE_SCHED_COMPONENT_PUSH_PRIO(component, queue->ntasks, exp_len);
		}
		STARPU_COMPONENT_MUTEX_UNLOCK(mutex);
		if(!is_pushback)
			component->can_pull(component);
	}

	return ret;
}

static int prio_push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	int ret = prio_push_local_task(component, task, 0);
	return ret;
}

static struct starpu_task * prio_pull_task(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	STARPU_ASSERT(component && component->data);
	struct _starpu_prio_data * data = component->data;
	struct starpu_st_prio_deque * queue = &data->prio;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	const double now = starpu_timing_now();

	if (!STARPU_RUNNING_ON_VALGRIND && starpu_st_prio_deque_is_empty(queue))
	{
		starpu_sched_component_send_can_push_to_parents(component);
		return NULL;
	}

	STARPU_COMPONENT_MUTEX_LOCK(mutex);
	struct starpu_task * task;
	if (data->ready && to->properties & STARPU_SCHED_COMPONENT_SINGLE_MEMORY_NODE)
		task = starpu_st_prio_deque_deque_first_ready_task(queue, starpu_bitmap_first(&to->workers_in_ctx));
	else
		task = starpu_st_prio_deque_pop_task(queue);
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

		STARPU_ASSERT_MSG(queue->exp_len>=0, "prio->exp_len=%lf\n",queue->exp_len);
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
	if(task)
		STARPU_TRACE_SCHED_COMPONENT_POP_PRIO(component, queue->ntasks, queue->exp_len);
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
 * push fails, which means that the worker prio_components are
 * currently "full".
 */
static int prio_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to STARPU_ATTRIBUTE_UNUSED)
{
	STARPU_ASSERT(component && starpu_sched_component_is_prio(component));
	int res = 0;
	struct starpu_task * task;

	task = starpu_sched_component_pump_downstream(component, &res);

	if(task)
	{
		int ret = prio_push_local_task(component,task,1);
		STARPU_ASSERT(!ret);
	}

	return res;
}

int starpu_sched_component_is_prio(struct starpu_sched_component * component)
{
	return component->push_task == prio_push_task;
}

struct starpu_sched_component * starpu_sched_component_prio_create(struct starpu_sched_tree *tree, struct starpu_sched_component_prio_data * params)
{
	struct starpu_sched_component * component = starpu_sched_component_create(tree, "prio");
	struct _starpu_prio_data *data;
	_STARPU_MALLOC(data, sizeof(*data));
	starpu_st_prio_deque_init(&data->prio);
	STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
	component->data = data;
	component->estimated_end = prio_estimated_end;
	component->estimated_load = prio_estimated_load;
	component->push_task = prio_push_task;
	component->pull_task = prio_pull_task;
	component->can_push = prio_can_push;
	component->deinit_data = prio_component_deinit_data;

	if(params)
	{
		data->ntasks_threshold=params->ntasks_threshold;
		data->exp_len_threshold=params->exp_len_threshold;
		data->ready=params->ready;
		data->exp=params->exp;
	}
	else
	{
		data->ntasks_threshold=0;
		data->exp_len_threshold=0.0;
		data->ready=0;
		data->exp=0;
	}

	return component;
}
