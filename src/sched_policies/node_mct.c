/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Université de Bordeaux 1
 * Copyright (C) 2013  INRIA
 * Copyright (C) 2013  Simon Archipoff
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
#include "sched_node.h"
#include <starpu_perfmodel.h>
#include "helper_mct.h"
#include <float.h>

static int mct_push_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(node && task && starpu_sched_node_is_mct(node));
	struct _starpu_mct_data * d = node->data;	
	struct starpu_sched_node * best_node = NULL;

	/* Estimated task duration for each child */
	double estimated_lengths[node->nchilds];
	/* Estimated transfer duration for each child */
	double estimated_transfer_length[node->nchilds];
	/* Estimated transfer+task termination for each child */
	double estimated_ends_with_task[node->nchilds];

	/* Minimum transfer+task termination on all children */
	double min_exp_end_with_task = DBL_MAX;
	/* Maximum transfer+task termination on all children */
	double max_exp_end_with_task = 0.0;

	int suitable_nodes[node->nchilds];
	int nsuitable_nodes = 0;

	int i;

	nsuitable_nodes = starpu_mct_compute_expected_times(node, task,
			estimated_lengths, estimated_transfer_length, estimated_ends_with_task,
			&min_exp_end_with_task, &max_exp_end_with_task, suitable_nodes);

	double best_fitness = DBL_MAX;
	int best_inode = -1;
	for(i = 0; i < nsuitable_nodes; i++)
	{
		int inode = suitable_nodes[i];
#ifdef STARPU_DEVEL
#warning FIXME: take power consumption into account
#endif
		double tmp = starpu_mct_compute_fitness(d,
					     estimated_ends_with_task[inode],
					     min_exp_end_with_task,
					     max_exp_end_with_task,
					     estimated_transfer_length[inode],
					     0.0);

		if(tmp < best_fitness)
		{
			best_fitness = tmp;
			best_inode = inode;
		}
	}

	STARPU_ASSERT(best_inode != -1);
	best_node = node->childs[best_inode];

	int ret = best_node->push_task(best_node, task);

	return ret;
}

void mct_node_deinit_data(struct starpu_sched_node * node)
{
	STARPU_ASSERT(starpu_sched_node_is_mct(node));
	struct _starpu_mct_data * d = node->data;
	free(d);
}

int starpu_sched_node_is_mct(struct starpu_sched_node * node)
{
	return node->push_task == mct_push_task;
}

struct starpu_sched_node * starpu_sched_node_mct_create(struct starpu_mct_data * params)
{
	struct starpu_sched_node * node = starpu_sched_node_create();
	struct _starpu_mct_data *data = starpu_mct_init_parameters(params);

	node->data = data;

	node->push_task = mct_push_task;
	node->deinit_data = mct_node_deinit_data;

	return node;
}
