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

/* eager component which has its own priority queue. It can thus eagerly push
 * tasks to lower queues without having to wait for being pulled from.  */

#include <starpu_sched_component.h>
#include "prio_deque.h"
#include <starpu_perfmodel.h>
#include <float.h>
#include <core/sched_policy.h>
#include <core/task.h>

struct _starpu_eager_prio_data
{
	struct _starpu_prio_deque prio;
	starpu_pthread_mutex_t mutex;
};

static int eager_prio_progress_one(struct starpu_sched_component *component)
{
	struct _starpu_eager_prio_data * data = component->data;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	struct _starpu_prio_deque * prio = &data->prio;
	struct starpu_task *task;
	int ret;

	STARPU_COMPONENT_MUTEX_LOCK(mutex);
	task = _starpu_prio_deque_pop_task(prio);
	STARPU_COMPONENT_MUTEX_UNLOCK(mutex);

	if (!task)
	{
		return 1;
	}

	/* FIXME: should rather just loop over children before looping over its workers */
	int workerid;
	for(workerid = starpu_bitmap_first(component->workers_in_ctx);
	    workerid != -1;
	    workerid = starpu_bitmap_next(component->workers_in_ctx, workerid))
	{
		int nimpl;
		for(nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			/* FIXME: use starpu_worker_can_execute_task_first_impl instead */
			if(starpu_worker_can_execute_task(workerid,task,nimpl)
			   || starpu_combined_worker_can_execute_task(workerid, task, nimpl))
			{
				unsigned i;
				for (i = 0; i < component->nchildren; i++)
				{
					int idworker;
					for(idworker = starpu_bitmap_first(component->children[i]->workers);
						idworker != -1;
						idworker = starpu_bitmap_next(component->children[i]->workers, idworker))
					{
						if (idworker == workerid)
						{
							STARPU_ASSERT(!starpu_sched_component_is_worker(component->children[i]));
							ret = starpu_sched_component_push_task(component,component->children[i],task);
							if (!ret)
								return 0;
						}
					}
				}
			}
		}
	}

	/* Could not push to child actually, push that one back too */
	STARPU_COMPONENT_MUTEX_LOCK(mutex);
	_starpu_prio_deque_push_front_task(prio, task);
	STARPU_COMPONENT_MUTEX_UNLOCK(mutex);
	return 1;
}

/* Try to push some tasks below */
static void eager_prio_progress(struct starpu_sched_component *component)
{
	STARPU_ASSERT(component && starpu_sched_component_is_eager_prio(component));
	while (!eager_prio_progress_one(component))
		;
}

static int eager_prio_push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	STARPU_ASSERT(component && task && starpu_sched_component_is_eager_prio(component));
	struct _starpu_eager_prio_data * data = component->data;
	struct _starpu_prio_deque * prio = &data->prio;
	starpu_pthread_mutex_t * mutex = &data->mutex;

	STARPU_COMPONENT_MUTEX_LOCK(mutex);
	_starpu_prio_deque_push_back_task(prio,task);
	STARPU_COMPONENT_MUTEX_UNLOCK(mutex);

	eager_prio_progress(component);

	return 0;
}

static int eager_prio_can_push(struct starpu_sched_component *component, struct starpu_sched_component * to STARPU_ATTRIBUTE_UNUSED)
{
	eager_prio_progress(component);
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

static void eager_prio_component_deinit_data(struct starpu_sched_component * component)
{
	STARPU_ASSERT(starpu_sched_component_is_eager_prio(component));
	struct _starpu_eager_prio_data * d = component->data;
	_starpu_prio_deque_destroy(&d->prio);
	free(d);
}

int starpu_sched_component_is_eager_prio(struct starpu_sched_component * component)
{
	return component->push_task == eager_prio_push_task;
}

struct starpu_sched_component * starpu_sched_component_eager_prio_create(struct starpu_sched_tree *tree, void *arg)
{
	(void)arg;
	struct starpu_sched_component * component = starpu_sched_component_create(tree, "eager_prio");
	struct _starpu_eager_prio_data *data;
	_STARPU_MALLOC(data, sizeof(*data));

	_starpu_prio_deque_init(&data->prio);
	STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
	component->data = data;

	component->push_task = eager_prio_push_task;
	component->can_push = eager_prio_can_push;
	component->deinit_data = eager_prio_component_deinit_data;

	return component;
}
