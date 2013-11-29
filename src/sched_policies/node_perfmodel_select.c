/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  INRIA
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

#include <starpu_sched_node.h>
#include <starpu_scheduler.h>

/* The decision node takes care of the scheduling of tasks which are not
 * calibrated, or tasks which don't have a performance model, because the scheduling
 * architecture of this scheduler for tasks with no performance model is exactly
 * the same as the tree-prio scheduler.
 * Tasks with a perfmodel are pushed to the perfmodel_node, which takes care of the
 * scheduling of those tasks on the correct worker_node.
 */

struct _starpu_perfmodel_select_data
{
	struct starpu_sched_node * calibrator_node;
	struct starpu_sched_node * no_perfmodel_node;
	struct starpu_sched_node * perfmodel_node;
};

static int perfmodel_select_push_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(node && node->data && task && starpu_sched_node_is_perfmodel_select(node));
	STARPU_ASSERT(starpu_sched_node_can_execute_task(node,task));

	struct _starpu_perfmodel_select_data * data = node->data;
	double length;
	int can_execute = starpu_sched_node_execute_preds(node,task,&length);
	
	if(can_execute)
	{
		if(isnan(length))
			return data->calibrator_node->push_task(data->calibrator_node,task);
		if(_STARPU_IS_ZERO(length))
			return data->no_perfmodel_node->push_task(data->no_perfmodel_node,task);
		return data->perfmodel_node->push_task(data->perfmodel_node,task);
	}
	else
		return 1;

}

int starpu_sched_node_is_perfmodel_select(struct starpu_sched_node * node)
{
	return node->push_task == perfmodel_select_push_task;
}

static void perfmodel_select_notify_change_in_workers(struct starpu_sched_node * node)
{
	STARPU_ASSERT(starpu_sched_node_is_perfmodel_select(node));
	struct _starpu_perfmodel_select_data * data = node->data;

	starpu_bitmap_unset_all(data->no_perfmodel_node->workers_in_ctx);
	starpu_bitmap_unset_all(data->no_perfmodel_node->workers);
	starpu_bitmap_or(data->no_perfmodel_node->workers_in_ctx, node->workers_in_ctx);
	starpu_bitmap_or(data->no_perfmodel_node->workers, node->workers);
	data->no_perfmodel_node->properties = node->properties;

	starpu_bitmap_unset_all(data->perfmodel_node->workers_in_ctx);
	starpu_bitmap_unset_all(data->perfmodel_node->workers);
	starpu_bitmap_or(data->perfmodel_node->workers_in_ctx, node->workers_in_ctx);
	starpu_bitmap_or(data->perfmodel_node->workers, node->workers);
	data->perfmodel_node->properties = node->properties;

	starpu_bitmap_unset_all(data->calibrator_node->workers_in_ctx);
	starpu_bitmap_unset_all(data->calibrator_node->workers);
	starpu_bitmap_or(data->calibrator_node->workers_in_ctx, node->workers_in_ctx);
	starpu_bitmap_or(data->calibrator_node->workers, node->workers);
	data->calibrator_node->properties = node->properties;
}

void perfmodel_select_node_deinit_data(struct starpu_sched_node * node)

{
	STARPU_ASSERT(node && node->data);
	struct _starpu_perfmodel_select_data * d = node->data;
	free(d);
}

struct starpu_sched_node * starpu_sched_node_perfmodel_select_create(struct starpu_perfmodel_select_data * params)
{
	STARPU_ASSERT(params);
	STARPU_ASSERT(params->calibrator_node && params->no_perfmodel_node && params->perfmodel_node);
	struct starpu_sched_node * node = starpu_sched_node_create();

	struct _starpu_perfmodel_select_data * data = malloc(sizeof(*data));
	data->calibrator_node = params->calibrator_node;
	data->no_perfmodel_node = params->no_perfmodel_node;
	data->perfmodel_node = params->perfmodel_node;
	
	node->data = data;
	node->push_task = perfmodel_select_push_task;
	node->deinit_data = perfmodel_select_node_deinit_data;
	node->notify_change_workers = perfmodel_select_notify_change_in_workers;

	return node;
}
