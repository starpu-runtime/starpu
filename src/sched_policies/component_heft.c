/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* HEFT variant which tries to schedule a given number of tasks instead of just
 * the first of its scheduling window, and actually schedule the task for which
 * the most benefit is achieved.  */

#include <starpu_sched_component.h>
#include "prio_deque.h"
#include <starpu_perfmodel.h>
#include "helper_mct.h"
#include <float.h>
#include <core/sched_policy.h>
#include <core/task.h>

#define NTASKS 5

struct _starpu_heft_data
{
	struct _starpu_prio_deque prio;
	starpu_pthread_mutex_t mutex;
	struct _starpu_mct_data *mct_data;
};

static int heft_progress_one(struct starpu_sched_component *component)
{
	struct _starpu_heft_data * data = component->data;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	struct _starpu_prio_deque * prio = &data->prio;
	struct starpu_task * (tasks[NTASKS]);
	unsigned ntasks = 0;

	STARPU_COMPONENT_MUTEX_LOCK(mutex);
	tasks[0] = _starpu_prio_deque_pop_task(prio);
	if (tasks[0])
	{
		int priority = tasks[0]->priority;
		/* Try to look at NTASKS from the queue */
		for (ntasks = 1; ntasks < NTASKS; ntasks++)
		{
			tasks[ntasks] = _starpu_prio_deque_highest_task(prio);
 			if (!tasks[ntasks] || tasks[ntasks]->priority < priority)
 				break;
 			_starpu_prio_deque_pop_task(prio);		
		}
	}
	STARPU_COMPONENT_MUTEX_UNLOCK(mutex);

	if (!ntasks)
	{
		return 1;
	}

	{
		struct _starpu_mct_data * d = data->mct_data;
		struct starpu_sched_component * best_component;
		unsigned n;

		/* Estimated task duration for each child */
		double estimated_lengths[component->nchildren * ntasks];
		/* Estimated transfer duration for each child */
		double estimated_transfer_length[component->nchildren * ntasks];
		/* Estimated transfer+task termination for each child */
		double estimated_ends_with_task[component->nchildren * ntasks];

		/* Minimum transfer+task termination on all children */
		double min_exp_end_with_task[ntasks];
		/* Maximum transfer+task termination on all children */
		double max_exp_end_with_task[ntasks];

		unsigned suitable_components[component->nchildren * ntasks];

		unsigned nsuitable_components[ntasks];

		/* Estimate durations */
		for (n = 0; n < ntasks; n++)
		{
			unsigned offset = component->nchildren * n;

			nsuitable_components[n] = starpu_mct_compute_execution_times(component, tasks[n],
					estimated_lengths + offset,
					estimated_transfer_length + offset,
					suitable_components + offset);

			starpu_mct_compute_expected_times(component, tasks[n],
					estimated_lengths + offset,
					estimated_transfer_length + offset,
					estimated_ends_with_task + offset,
					&min_exp_end_with_task[n], &max_exp_end_with_task[n],
							  suitable_components + offset, nsuitable_components[n]);
		}

		/* best_task is the task that will finish first among the ntasks, while best_benefit is its expected execution time*/
		int best_task = 0;
		double best_benefit = min_exp_end_with_task[0];

		/* Find the task which provides the most computation time benefit */
		for (n = 1; n < ntasks; n++)
		{
			if (best_benefit > min_exp_end_with_task[n])
			{
				best_benefit =  min_exp_end_with_task[n];
				best_task = n;
			}
		}

		STARPU_ASSERT(best_task >= 0);

		/* Push back the other tasks */
		STARPU_COMPONENT_MUTEX_LOCK(mutex);
		for (n = ntasks - 1; n < ntasks; n--)
			if ((int) n != best_task)
				_starpu_prio_deque_push_front_task(prio, tasks[n]);
		STARPU_COMPONENT_MUTEX_UNLOCK(mutex);

		unsigned offset = component->nchildren * best_task;

		int best_icomponent = starpu_mct_get_best_component(d, tasks[best_task], estimated_lengths + offset, estimated_transfer_length + offset, estimated_ends_with_task + offset, min_exp_end_with_task[best_task], max_exp_end_with_task[best_task], suitable_components + offset, nsuitable_components[best_task]);

		STARPU_ASSERT(best_icomponent != -1);
		best_component = component->children[best_icomponent];

		if(starpu_sched_component_is_worker(best_component))
		{
			best_component->can_pull(best_component);
			return 1;
		}

		starpu_sched_task_break(tasks[best_task]);
		int ret = starpu_sched_component_push_task(component, best_component, tasks[best_task]);

		if (ret)
		{
			/* Could not push to child actually, push that one back too */
			STARPU_COMPONENT_MUTEX_LOCK(mutex);
			_starpu_prio_deque_push_front_task(prio, tasks[best_task]);
			STARPU_COMPONENT_MUTEX_UNLOCK(mutex);
			return 1;
		}
		else
			return 0;
	}
}

/* Try to push some tasks below */
static void heft_progress(struct starpu_sched_component *component)
{
	STARPU_ASSERT(component && starpu_sched_component_is_heft(component));
	while (!heft_progress_one(component))
		;
}

static int heft_push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	STARPU_ASSERT(component && task && starpu_sched_component_is_heft(component));
	struct _starpu_heft_data * data = component->data;
	struct _starpu_prio_deque * prio = &data->prio;
	starpu_pthread_mutex_t * mutex = &data->mutex;

	STARPU_COMPONENT_MUTEX_LOCK(mutex);
	_starpu_prio_deque_push_back_task(prio,task);
	STARPU_COMPONENT_MUTEX_UNLOCK(mutex);

	heft_progress(component);

	return 0;
}

static int heft_can_push(struct starpu_sched_component *component, struct starpu_sched_component * to STARPU_ATTRIBUTE_UNUSED)
{
	heft_progress(component);
	int ret = 0;
	unsigned j;
	for(j=0; j < component->nparents; j++)
	{
		if(component->parents[j] == NULL)
			continue;
		else
		{
			ret = component->parents[j]->can_push(component->parents[j], component);
			if(ret)
				break;
		}
	}
	return ret;
}

static void heft_component_deinit_data(struct starpu_sched_component * component)
{
	STARPU_ASSERT(starpu_sched_component_is_heft(component));
	struct _starpu_heft_data * d = component->data;
	struct _starpu_mct_data * mct_d = d->mct_data;
	_starpu_prio_deque_destroy(&d->prio);
	free(mct_d);
	free(d);
}

int starpu_sched_component_is_heft(struct starpu_sched_component * component)
{
	return component->push_task == heft_push_task;
}

struct starpu_sched_component * starpu_sched_component_heft_create(struct starpu_sched_tree *tree, struct starpu_sched_component_mct_data * params)
{
	struct starpu_sched_component * component = starpu_sched_component_create(tree, "heft");
	struct _starpu_mct_data *mct_data = starpu_mct_init_parameters(params);
	struct _starpu_heft_data *data;
	_STARPU_MALLOC(data, sizeof(*data));

	_starpu_prio_deque_init(&data->prio);
	STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
	data->mct_data = mct_data;
	component->data = data;

	component->push_task = heft_push_task;
	component->can_push = heft_can_push;
	component->deinit_data = heft_component_deinit_data;

	return component;
}
