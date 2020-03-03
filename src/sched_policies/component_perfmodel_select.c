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

/* The decision component takes care of the scheduling of tasks which are not
 * calibrated, or tasks which don't have a performance model, because the scheduling
 * architecture of this scheduler for tasks with no performance model is exactly
 * the same as the tree-prio scheduler.
 * Tasks with a perfmodel are pushed to the perfmodel_component, which takes care of the
 * scheduling of those tasks on the correct worker_component.
 */

struct _starpu_perfmodel_select_data
{
	struct starpu_sched_component * calibrator_component;
	struct starpu_sched_component * no_perfmodel_component;
	struct starpu_sched_component * perfmodel_component;
};

static int perfmodel_select_push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	STARPU_ASSERT(component && component->data && task && starpu_sched_component_is_perfmodel_select(component));
	STARPU_ASSERT(starpu_sched_component_can_execute_task(component,task));

	struct _starpu_perfmodel_select_data * data = component->data;
	double length;
	int can_execute = starpu_sched_component_execute_preds(component,task,&length);

	if(can_execute)
	{
		if(isnan(length))
		{
			static int warned;
			if (!warned)
			{
				warned = 1;
				_STARPU_DISP("Warning: performance model for %s not finished calibrating, using a dumb scheduling heuristic for now\n",starpu_task_get_name(task));
			}
			return starpu_sched_component_push_task(component,data->calibrator_component,task);
		}
		if(_STARPU_IS_ZERO(length))
			return starpu_sched_component_push_task(component,data->no_perfmodel_component,task);
		return starpu_sched_component_push_task(component,data->perfmodel_component,task);
	}
	else
		return 1;

}

static struct starpu_task * perfmodel_select_pull_task(struct starpu_sched_component * component STARPU_ATTRIBUTE_UNUSED, struct starpu_sched_component * to STARPU_ATTRIBUTE_UNUSED)
{
	/* We don't want to pull tasks blindly, only let them go through push, so we push to the right component. */
	return NULL;
}

static void perfmodel_select_component_deinit_data(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component && component->data);
	struct _starpu_perfmodel_select_data * d = component->data;
	free(d);
}

int starpu_sched_component_is_perfmodel_select(struct starpu_sched_component * component)
{
	return component->push_task == perfmodel_select_push_task;
}

struct starpu_sched_component * starpu_sched_component_perfmodel_select_create(struct starpu_sched_tree *tree, struct starpu_sched_component_perfmodel_select_data * params)
{
	STARPU_ASSERT(params);
	STARPU_ASSERT(params->calibrator_component && params->no_perfmodel_component && params->perfmodel_component);
	struct starpu_sched_component * component = starpu_sched_component_create(tree, "perfmodel_selector");

	struct _starpu_perfmodel_select_data *data;
	_STARPU_MALLOC(data, sizeof(*data));

	data->calibrator_component = params->calibrator_component;
	data->no_perfmodel_component = params->no_perfmodel_component;
	data->perfmodel_component = params->perfmodel_component;

	component->data = data;
	component->can_pull = starpu_sched_component_send_can_push_to_parents;
	component->push_task = perfmodel_select_push_task;
	component->pull_task = perfmodel_select_pull_task;
	component->deinit_data = perfmodel_select_component_deinit_data;
	component->estimated_end = starpu_sched_component_estimated_end_min;

	return component;
}
