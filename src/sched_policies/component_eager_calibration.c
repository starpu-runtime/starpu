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

static int eager_calibration_push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	STARPU_ASSERT(component && task && starpu_sched_component_is_eager_calibration(component));
	STARPU_ASSERT(starpu_sched_component_can_execute_task(component,task));
	
	starpu_task_bundle_t bundle = task->bundle;

	int workerid;
	for(workerid = starpu_bitmap_first(component->workers_in_ctx);
	    workerid != -1;
	    workerid = starpu_bitmap_next(component->workers_in_ctx, workerid))
	{
		struct starpu_perfmodel_arch* archtype = starpu_worker_get_perf_archtype(workerid, component->tree->sched_ctx_id);
		int nimpl;
		for(nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if(starpu_worker_can_execute_task(workerid,task,nimpl)
			   || starpu_combined_worker_can_execute_task(workerid, task, nimpl))
			{
				double d;

				if(bundle)
					d = starpu_task_bundle_expected_length(bundle, archtype, nimpl);
				else
					d = starpu_task_expected_length(task, archtype, nimpl);

				if(isnan(d))
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
								return starpu_sched_component_push_task(component,component->children[i],task);
							}
						}
					}
				}
			}
		}
	}
	return 1;
}

int starpu_sched_component_is_eager_calibration(struct starpu_sched_component * component)
{
	return component->push_task == eager_calibration_push_task;
}

struct starpu_sched_component * starpu_sched_component_eager_calibration_create(struct starpu_sched_tree *tree, void *arg)
{
	(void)arg;
	struct starpu_sched_component * component = starpu_sched_component_create(tree, "eager_calibration");
	component->push_task = eager_calibration_push_task;

	return component;
}
