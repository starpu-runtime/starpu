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

/* HEFT variant which tries to schedule a given number of tasks instead of just
 * the first of its scheduling window, and actually schedule the task for which
 * the most benefit is achieved.  */

#include <starpu_sched_node.h>
#include "prio_deque.h"
#include "sched_node.h"
#include <starpu_perfmodel.h>
#include "helper_mct.h"
#include <float.h>

#define NTASKS 5

struct _starpu_heft_data
{
	struct _starpu_prio_deque prio;
	starpu_pthread_mutex_t mutex;
	struct _starpu_mct_data *mct_data;
};

static void heft_progress(struct starpu_sched_node *node);

static int heft_push_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(node && task && starpu_sched_node_is_heft(node));
	struct _starpu_heft_data * data = node->data;
	struct _starpu_prio_deque * prio = &data->prio;
	starpu_pthread_mutex_t * mutex = &data->mutex;

	STARPU_PTHREAD_MUTEX_LOCK(mutex);
	_starpu_prio_deque_push_task(prio,task);
	STARPU_PTHREAD_MUTEX_UNLOCK(mutex);

	heft_progress(node);

	return 0;
}

static int heft_progress_one(struct starpu_sched_node *node)
{
	struct _starpu_heft_data * data = node->data;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	struct _starpu_prio_deque * prio = &data->prio;
	struct starpu_task * (tasks[NTASKS]);

	unsigned ntasks, n, i;

	STARPU_PTHREAD_MUTEX_LOCK(mutex);
	/* Try to look at NTASKS from the queue */
	for (ntasks = 0; ntasks < NTASKS; ntasks++)
	{
		tasks[ntasks] = _starpu_prio_deque_pop_task(prio);
		if (!tasks[ntasks])
			break;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(mutex);

	if (!ntasks) {
		return 1;
	}

	{
		struct _starpu_mct_data * d = data->mct_data;
		struct starpu_sched_node * best_node = NULL;

		/* Estimated task duration for each child */
		double estimated_lengths[node->nchilds * ntasks];
		/* Estimated transfer duration for each child */
		double estimated_transfer_length[node->nchilds * ntasks];
		/* Estimated transfer+task termination for each child */
		double estimated_ends_with_task[node->nchilds * ntasks];

		/* Minimum transfer+task termination on all children */
		double min_exp_end_with_task[ntasks];
		/* Maximum transfer+task termination on all children */
		double max_exp_end_with_task[ntasks];

		int suitable_nodes[node->nchilds * ntasks];

		unsigned nsuitable_nodes[ntasks];

		for (n = 0; n < ntasks; n++)
		{
			int offset = node->nchilds * n;

			min_exp_end_with_task[n] = DBL_MAX;
			max_exp_end_with_task[n] = 0.0;

			nsuitable_nodes[n] = starpu_mct_compute_expected_times(node, tasks[n],
					estimated_lengths + offset,
					estimated_transfer_length + offset,
					estimated_ends_with_task + offset,
					&min_exp_end_with_task[n], &max_exp_end_with_task[n],
					suitable_nodes + offset);
		}

		double best_fitness = DBL_MAX;
		int best_inode = -1;
		int best_task = -1;

		for (n = 0; n < ntasks; n++)
		{
			for(i = 0; i < nsuitable_nodes[n]; i++)
			{
				int offset = node->nchilds * n;
				int inode = suitable_nodes[offset + i];
#ifdef STARPU_DEVEL
#warning FIXME: take power consumption into account
#endif
				double tmp = starpu_mct_compute_fitness(d,
							     estimated_ends_with_task[offset + inode],
							     min_exp_end_with_task[n],
							     max_exp_end_with_task[n],
							     estimated_transfer_length[offset + inode],
							     0.0);

				if(tmp < best_fitness)
				{
					best_fitness = tmp;
					best_inode = inode;
					best_task = n;
				}
			}
		}

		STARPU_ASSERT(best_inode != -1);
		STARPU_ASSERT(best_task >= 0);
		best_node = node->childs[best_inode];

		/* Push back the other tasks */
		STARPU_PTHREAD_MUTEX_LOCK(mutex);
		for (n = 0; n < ntasks; n++)
			if ((int) n != best_task)
				_starpu_prio_deque_push_back_task(prio, tasks[n]);
		STARPU_PTHREAD_MUTEX_UNLOCK(mutex);

		int ret = best_node->push_task(best_node, tasks[best_task]);

		if (ret)
		{
			/* Could not push to child actually, push that one back too */
			STARPU_PTHREAD_MUTEX_LOCK(mutex);
			_starpu_prio_deque_push_back_task(prio, tasks[best_task]);
			STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
			return 1;
		}
		else
		{
			return 0;
		}
	}
}

/* Try to push some tasks below */
static void heft_progress(struct starpu_sched_node *node)
{
	STARPU_ASSERT(node && starpu_sched_node_is_heft(node));
	while (!heft_progress_one(node))
		;
}

static void heft_room(struct starpu_sched_node *node, unsigned sched_ctx_id)
{
	heft_progress(node);
}

void heft_node_deinit_data(struct starpu_sched_node * node)
{
	STARPU_ASSERT(starpu_sched_node_is_heft(node));
	struct _starpu_mct_data * d = node->data;
	free(d);
}

int starpu_sched_node_is_heft(struct starpu_sched_node * node)
{
	return node->push_task == heft_push_task;
}

struct starpu_sched_node * starpu_sched_node_heft_create(struct starpu_mct_data * params)
{
	struct starpu_sched_node * node = starpu_sched_node_create();
	struct _starpu_mct_data *mct_data = starpu_mct_init_parameters(params);
	struct _starpu_heft_data *data = malloc(sizeof(*data));

	_starpu_prio_deque_init(&data->prio);
	STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
	data->mct_data = mct_data;
	node->data = data;

	node->push_task = heft_push_task;
	node->room = heft_room;
	node->deinit_data = heft_node_deinit_data;

	return node;
}
