/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Marc Sergent
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

struct _starpu_calibrator_data
{
	struct starpu_sched_node * no_perf_model_node;
	struct starpu_sched_node * next_node;
};

static int calibrator_push_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(node && node->data && task && starpu_sched_node_is_calibrator(node));
	STARPU_ASSERT(starpu_sched_node_can_execute_task(node,task));

	struct _starpu_calibrator_data * data = node->data;
	starpu_task_bundle_t bundle = task->bundle;
	

	int workerid;
	for(workerid = starpu_bitmap_first(node->workers_in_ctx);
	    workerid != -1;
	    workerid = starpu_bitmap_next(node->workers_in_ctx, workerid))
	{
		struct starpu_perfmodel_arch* archtype = starpu_worker_get_perf_archtype(workerid);
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
					int i;
					for (i = 0; i < node->nchilds; i++)
					{
						int idworker;
						for(idworker = starpu_bitmap_first(node->childs[i]->workers);
							idworker != -1;
							idworker = starpu_bitmap_next(node->childs[i]->workers, idworker))
						{
							if (idworker == workerid)
								return node->childs[i]->push_task(node->childs[i],task);
						}
					}
				}
				if(_STARPU_IS_ZERO(d))
					return data->no_perf_model_node->push_task(data->no_perf_model_node,task);
			}
		}
	}

	return data->next_node->push_task(data->next_node,task);
}

int starpu_sched_node_is_calibrator(struct starpu_sched_node * node)
{
	return node->push_task == calibrator_push_task;
}

void calibrator_add_child(struct starpu_sched_node * node, struct starpu_sched_node * child)
{
	STARPU_ASSERT(starpu_sched_node_is_calibrator(node));
	starpu_sched_node_add_child(node, child);
	struct _starpu_calibrator_data * data = node->data;
	starpu_sched_node_add_child(data->no_perf_model_node,child);
	starpu_sched_node_add_child(data->next_node, child);
}

void calibrator_remove_child(struct starpu_sched_node * node, struct starpu_sched_node * child)
{

	STARPU_ASSERT(starpu_sched_node_is_calibrator(node));
	starpu_sched_node_remove_child(node, child);
	struct _starpu_calibrator_data * data = node->data;
	starpu_sched_node_remove_child(data->no_perf_model_node,child);
	starpu_sched_node_remove_child(data->next_node, child);
}

static void calibrator_notify_change_in_workers(struct starpu_sched_node * node)
{
	STARPU_ASSERT(starpu_sched_node_is_calibrator(node));
	struct _starpu_calibrator_data * data = node->data;
	starpu_bitmap_unset_all(data->no_perf_model_node->workers_in_ctx);
	starpu_bitmap_unset_all(data->no_perf_model_node->workers);

	starpu_bitmap_or(data->no_perf_model_node->workers_in_ctx, node->workers_in_ctx);
	starpu_bitmap_or(data->no_perf_model_node->workers, node->workers);

	data->no_perf_model_node->properties = node->properties;

	starpu_bitmap_unset_all(data->next_node->workers_in_ctx);
	starpu_bitmap_unset_all(data->next_node->workers);

	starpu_bitmap_or(data->next_node->workers_in_ctx, node->workers_in_ctx);
	starpu_bitmap_or(data->next_node->workers, node->workers);

	data->next_node->properties = node->properties;
}

void calibrator_node_deinit_data(struct starpu_sched_node * node)

{
	STARPU_ASSERT(node && node->data);
	struct _starpu_calibrator_data * d = node->data;
	starpu_sched_node_destroy(d->no_perf_model_node);
	starpu_sched_node_destroy(d->next_node);
	free(d);
}

struct starpu_sched_node * starpu_sched_node_calibrator_create(struct starpu_calibrator_data * params)
{
	STARPU_ASSERT(params);
	struct starpu_sched_node * node = starpu_sched_node_create();
	struct _starpu_calibrator_data * data = malloc(sizeof(*data));
	data->no_perf_model_node = params->no_perf_model_node_create(params->arg_no_perf_model);
	data->next_node = params->next_node;
	node->data = data;
	node->push_task = calibrator_push_task;
	node->add_child = calibrator_add_child;
	node->remove_child = calibrator_remove_child;
	node->deinit_data = calibrator_node_deinit_data;
	node->notify_change_workers = calibrator_notify_change_in_workers;

	return node;
}
