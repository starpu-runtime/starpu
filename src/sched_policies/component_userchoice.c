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

/* This component uses (uintptr_t) tasks->sched_data as the child number it
 * should push its tasks to. It can thus be used to let the user choose which
 * scheduler a task should go to. */

#include <starpu_sched_component.h>
#include <starpu_scheduler.h>

static int userchoice_push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	unsigned target = (uintptr_t) task->sched_data;
	STARPU_ASSERT(target < component->nchildren);
	return starpu_sched_component_push_task(component, component->children[target], task);
}

static struct starpu_task * userchoice_pull_task(struct starpu_sched_component * component, struct starpu_sched_component * to STARPU_ATTRIBUTE_UNUSED)
{
	_STARPU_DISP("stage component is not supposed to be pull from...\n");
	return starpu_sched_component_parents_pull_task(component, to);
}

int userchoice_can_pull(struct starpu_sched_component * component)
{
	_STARPU_DISP("stage component is not supposed to be pull from...\n");
	return starpu_sched_component_can_pull(component);
}

int starpu_sched_component_is_userchoice(struct starpu_sched_component * component)
{
	return component->push_task == userchoice_push_task;
}

struct starpu_sched_component * starpu_sched_component_userchoice_create(struct starpu_sched_tree *tree, void *args STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "userchoice");
	component->push_task = userchoice_push_task;
	component->pull_task = userchoice_pull_task;
	component->can_pull = userchoice_can_pull;

	return component;
}
