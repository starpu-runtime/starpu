/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
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
#include <float.h>

struct _starpu_mct_data
{
	double alpha;
	double beta;
	double gamma;
	double idle_power;
};

/* compute predicted_end by taking into account the case of the predicted transfer and the predicted_end overlap
 */
double compute_expected_time(double now, double predicted_end, double predicted_length, double predicted_transfer)
{
	STARPU_ASSERT(!isnan(now + predicted_end + predicted_length + predicted_transfer));
	STARPU_ASSERT(now >= 0.0 && predicted_end >= 0.0 && predicted_length >= 0.0 && predicted_transfer >= 0.0);
	if (now + predicted_transfer < predicted_end)
	{
		/* We may hope that the transfer will be finished by
		 * the start of the task. */
		predicted_transfer = 0;
	}
	else
	{
		/* The transfer will not be finished by then, take the
		 * remainder into account */
		predicted_transfer += now;
		predicted_transfer -= predicted_end;
	}
//	if(!isnan(predicted_transfer))
	{
		predicted_end += predicted_transfer;
		predicted_length += predicted_transfer;
	}

//	if(!isnan(predicted_length))
		predicted_end += predicted_length;
	return predicted_end;
}


static double compute_fitness(struct _starpu_mct_data * d, double exp_end, double best_exp_end, double max_exp_end, double transfer_len, double local_power)
{
	return d->alpha * (exp_end - best_exp_end)
		+ d->beta * transfer_len
		+ d->gamma * local_power
		+ d->gamma * d->idle_power * (exp_end - max_exp_end);
}

static int mct_push_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(node && task && starpu_sched_node_is_mct(node));
	struct _starpu_mct_data * d = node->data;	
	struct starpu_sched_node * best_node = NULL;
	double estimated_ends[node->nchilds];
	double estimated_ends_with_task[node->nchilds];
	double best_exp_end_with_task = DBL_MAX;
	double max_exp_end_with_task = 0.0;
	double estimated_lengths[node->nchilds];
	double estimated_transfer_length[node->nchilds];
	int suitable_nodes[node->nchilds];
	int nsuitable_nodes = 0;

	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		struct starpu_sched_node * c = node->childs[i];
		if(starpu_sched_node_execute_preds(c, task, estimated_lengths + i))
		{
			estimated_transfer_length[i] = starpu_sched_node_transfer_length(c, task);
			estimated_ends[i] = c->estimated_end(c);
			double now = starpu_timing_now();
			estimated_ends_with_task[i] = compute_expected_time(now,
									    estimated_ends[i],
									    estimated_lengths[i],
									    estimated_transfer_length[i]);
			if(estimated_ends_with_task[i] < best_exp_end_with_task)	
				best_exp_end_with_task = estimated_ends_with_task[i];
			if(estimated_ends_with_task[i] > max_exp_end_with_task)
				max_exp_end_with_task = estimated_ends_with_task[i];
			suitable_nodes[nsuitable_nodes++] = i;
		}
	}

	double best_fitness = DBL_MAX;
	int best_inode = -1;
	for(i = 0; i < nsuitable_nodes; i++)
	{
		int inode = suitable_nodes[i];
#ifdef STARPU_DEVEL
#warning FIXME: take power consumption into account
#endif
		double tmp = compute_fitness(d,
					     estimated_ends_with_task[inode],
					     best_exp_end_with_task,
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

	struct _starpu_mct_data * data = malloc(sizeof(*data));
	data->alpha = params->alpha;
	data->beta = params->beta;
	data->gamma = params->gamma;
	data->idle_power = params->idle_power;

	node->data = data;

	node->push_task = mct_push_task;
	node->deinit_data = mct_node_deinit_data;

	return node;
}
