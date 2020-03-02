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

#include <starpu_sched_component.h>
#include <core/workers.h>
#include <core/sched_policy.h>
#include <core/task.h>

static double compute_relative_speedup(struct starpu_sched_component * component)
{
	double sum = 0.0;
	int id;
	for(id = starpu_bitmap_first(component->workers_in_ctx);
	    id != -1;
	    id = starpu_bitmap_next(component->workers_in_ctx, id))
	{
		struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(id, component->tree->sched_ctx_id);
		sum += starpu_worker_get_relative_speedup(perf_arch);

	}
	STARPU_ASSERT(sum != 0.0);
	return sum;
}

static int random_push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	STARPU_ASSERT(component->nchildren > 0);

	/* indexes_components and size are used to memoize component that can execute tasks
	 * during the first phase of algorithm, it contain the size indexes of the components
	 * that can execute task.
	 */
	int indexes_components[component->nchildren];
	unsigned size=0;

	/* speedup[i] is revelant only if i is in the size firsts elements of
	 * indexes_components
	 */
	double speedup[component->nchildren];

	double alpha_sum = 0.0;

	unsigned i;
	for(i = 0; i < component->nchildren ; i++)
	{
		if(starpu_sched_component_can_execute_task(component->children[i],task))
		{
			speedup[size] = compute_relative_speedup(component->children[i]);
			alpha_sum += speedup[size];
			indexes_components[size] = i;
			size++;
		}
	}
	if(size == 0)
		return -ENODEV;

	/* not fully sure that this code is correct
	 * because of bad properties of double arithmetic
	 */
	double random = starpu_drand48()*alpha_sum;
	double alpha = 0.0;
	struct starpu_sched_component * select  = NULL;

	for(i = 0; i < size ; i++)
	{
		int index = indexes_components[i];
		if(alpha + speedup[i] >= random)
		{
			select = component->children[index];
			break;
		}
		alpha += speedup[i];
	}
	STARPU_ASSERT(select != NULL);
	if(starpu_sched_component_is_worker(select))
	{
		select->can_pull(select);
		return 1;
	}

	starpu_sched_task_break(task);
	int ret_val = starpu_sched_component_push_task(component,select,task);
	return ret_val;
}

int starpu_sched_component_is_random(struct starpu_sched_component *component)
{
	return component->push_task == random_push_task;
}

struct starpu_sched_component * starpu_sched_component_random_create(struct starpu_sched_tree *tree, void *arg)
{
	(void)arg;
	struct starpu_sched_component * component = starpu_sched_component_create(tree, "random");
	component->push_task = random_push_task;
	return component;
}
