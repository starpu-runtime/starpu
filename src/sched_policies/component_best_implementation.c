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

#include <float.h>

#include <starpu_sched_component.h>
#include <starpu_scheduler.h>
#include <core/workers.h>

/* return true if workerid can execute task, and fill task->predicted and task->predicted_transfer
 *  according to best implementation predictions
 */
static int find_best_impl(unsigned sched_ctx_id, struct starpu_task * task, int workerid)
{
	double len = DBL_MAX;
	int best_impl = -1;
	unsigned impl;
	if (!task->cl->model)
	{
		/* No perfmodel, first available will be fine */
		starpu_worker_can_execute_task_first_impl(workerid, task, &impl);
		best_impl = impl;
		len = 0.0;
	}
	else
	{
		struct starpu_perfmodel_arch* archtype = starpu_worker_get_perf_archtype(workerid, sched_ctx_id);
		for(impl = 0; impl < STARPU_MAXIMPLEMENTATIONS; impl++)
		{
			if(starpu_worker_can_execute_task(workerid, task, impl))
			{
				double d = starpu_task_expected_length(task, archtype, impl);
				if(isnan(d))
				{
					best_impl = impl;
					len = 0.0;
					break;
				}
				if(d < len)
				{
					len = d;
					best_impl = impl;
				}
			}
		}
	}
	if(best_impl == -1)
		return 0;

	task->predicted = len;
	task->predicted_transfer = starpu_task_expected_data_transfer_time_for(task, workerid);
	starpu_task_set_implementation(task, best_impl);
	return 1;
}


/* set implementation, task->predicted and task->predicted_transfer with the first worker of workers that can execute that task
 * or have to be calibrated
 */
static void select_best_implementation_and_set_preds(unsigned sched_ctx_id, struct starpu_bitmap * workers, struct starpu_task * task)
{
	int workerid;
	for(workerid = starpu_bitmap_first(workers);
	    -1 != workerid;
	    workerid = starpu_bitmap_next(workers, workerid))
		if(find_best_impl(sched_ctx_id, task, workerid))
			break;
}

static int best_implementation_push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	STARPU_ASSERT(component->nchildren == 1);
	select_best_implementation_and_set_preds(component->tree->sched_ctx_id, component->workers_in_ctx, task);
	return starpu_sched_component_push_task(component,component->children[0],task);
}

int starpu_sched_component_is_best_implementation(struct starpu_sched_component * component)
{
	return component->push_task == best_implementation_push_task;
}

static struct starpu_task * best_implementation_pull_task(struct starpu_sched_component * component, struct starpu_sched_component * from STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_task * task = NULL;
	unsigned i;
	for(i=0; i < component->nparents; i++)
	{
		if(component->parents[i] == NULL)
			continue;
		else
		{
			task = starpu_sched_component_pull_task(component->parents[i], component);
			if(task)
				break;
		}
	}
	if(task)
		/* this worker can execute this task as it was returned by a pop*/
		(void)find_best_impl(component->tree->sched_ctx_id, task, starpu_worker_get_id_check());
	return task;
}

struct starpu_sched_component * starpu_sched_component_best_implementation_create(struct starpu_sched_tree *tree, void *arg)
{
	(void)arg;
	struct starpu_sched_component * component = starpu_sched_component_create(tree, "best_impl");
	component->push_task = best_implementation_push_task;
	component->pull_task = best_implementation_pull_task;
	return component;
}
