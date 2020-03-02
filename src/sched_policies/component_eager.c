/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

struct _starpu_eager_data
{
	struct starpu_sched_component *target;
	starpu_pthread_mutex_t scheduling_mutex;
};

static int eager_push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	int ret;
	STARPU_ASSERT(component && task && starpu_sched_component_is_eager(component));
	STARPU_ASSERT(starpu_sched_component_can_execute_task(component,task));
	struct _starpu_eager_data *d = component->data;
	struct starpu_sched_component *target;

	if ((target = d->target))
	{
		/* target told us we could push to it, try to */
		int idworker;
		for(idworker = starpu_bitmap_first(target->workers);
			idworker != -1;
			idworker = starpu_bitmap_next(target->workers, idworker))
		{
			int nimpl;
			for(nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
			{
				if(starpu_worker_can_execute_task(idworker,task,nimpl)
				   || starpu_combined_worker_can_execute_task(idworker, task, nimpl))
				{
					ret = starpu_sched_component_push_task(component,target,task);
					if (!ret)
						return 0;
				}
			}
		}
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
							if(starpu_sched_component_is_worker(component->children[i]))
							{
								if (component->children[i]->can_pull(component->children[i]))
									return 1;
							}
							else
							{
								ret = starpu_sched_component_push_task(component,component->children[i],task);
								if (!ret)
									return 0;
							}
						}
					}
				}
			}
		}
	}
	return 1;
}

/* Note: we can't use starpu_sched_component_pump_to because if a fifo below
 * refuses a task, we have no way to push it back to a fifo above. */
static int eager_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to)
{
	int success;
	struct _starpu_eager_data *d = component->data;
	STARPU_COMPONENT_MUTEX_LOCK(&d->scheduling_mutex);
	/* Target flow of tasks to this child */
	d->target = to;
	success = starpu_sched_component_can_push(component, to);
	d->target = NULL;
	STARPU_COMPONENT_MUTEX_UNLOCK(&d->scheduling_mutex);
	return success;
}

static void eager_deinit_data(struct starpu_sched_component *component)
{
	STARPU_ASSERT(starpu_sched_component_is_eager(component));
	struct _starpu_eager_data *d = component->data;
	STARPU_PTHREAD_MUTEX_DESTROY(&d->scheduling_mutex);
	free(d);
}

int starpu_sched_component_is_eager(struct starpu_sched_component * component)
{
	return component->push_task == eager_push_task;
}

struct starpu_sched_component * starpu_sched_component_eager_create(struct starpu_sched_tree *tree, void *arg)
{
	(void)arg;
	struct starpu_sched_component * component = starpu_sched_component_create(tree, "eager");
	struct _starpu_eager_data *data;
	_STARPU_MALLOC(data, sizeof(*data));
	data->target = NULL;
	STARPU_PTHREAD_MUTEX_INIT(&data->scheduling_mutex, NULL);

	component->data = data;
	component->push_task = eager_push_task;
	component->can_push = eager_can_push;
	component->can_pull = starpu_sched_component_can_pull_all;
	component->deinit_data = eager_deinit_data;

	return component;
}
