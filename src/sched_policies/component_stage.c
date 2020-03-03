/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* This component takes tasks from its parents in the parent order.
 * It can be useful to make scheduling stages, pushing tasks of different stages
 * to different schedulers, and this component will pick them up in the right
 * order. */

#include <starpu_sched_component.h>
#include <starpu_scheduler.h>

static int stage_push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	_STARPU_DISP("stage component is not supposed to be pushed to...\n");
	STARPU_ASSERT(component->nchildren == 1);
	return starpu_sched_component_push_task(component, component->children[0], task);
}

static int stage_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to STARPU_ATTRIBUTE_UNUSED)
{
	_STARPU_DISP("stage component is not supposed to be pushed to...\n");
	return starpu_sched_component_can_push(component, to);
}

static struct starpu_task * stage_pull_task(struct starpu_sched_component * component, struct starpu_sched_component * to STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_task *task;
	task = starpu_sched_component_parents_pull_task(component, to);
	return task;
}

int starpu_sched_component_is_stage(struct starpu_sched_component * component)
{
	return component->push_task == stage_push_task;
}

struct starpu_sched_component * starpu_sched_component_stage_create(struct starpu_sched_tree *tree, void *args STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "stage");
	component->push_task = stage_push_task;
	/* The default implementation happens to be doing staged pull from parents */
	component->pull_task = stage_pull_task;
	component->can_push = stage_can_push;

	return component;
}
