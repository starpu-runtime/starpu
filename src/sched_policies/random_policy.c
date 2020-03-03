/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Simon Archipoff
 * Copyright (C) 2013       Thibaut Lambert
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

/* Policy attributing tasks randomly to workers */

#include <starpu_rand.h>
#include <core/workers.h>
#include <core/sched_ctx.h>
#include <core/sched_policy.h>
#include <sched_policies/fifo_queues.h>
#include <core/debug.h>
#include <core/task.h>

static int _random_push_task(struct starpu_task *task, unsigned prio)
{
	/* find the queue */


	double alpha_sum = 0.0;

	unsigned sched_ctx_id = task->sched_ctx;
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
        int worker;
	int worker_arr[STARPU_NMAXWORKERS];
	double speedup_arr[STARPU_NMAXWORKERS];
	int size = 0;
	struct starpu_sched_ctx_iterator it;

	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
                worker = workers->get_next(workers, &it);
		unsigned impl;
		if(starpu_worker_can_execute_task_first_impl(worker, task, &impl))
		{
			struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(worker, sched_ctx_id);
			double speedup = starpu_worker_get_relative_speedup(perf_arch);
			alpha_sum += speedup;
			speedup_arr[size] = speedup;
			worker_arr[size++] = worker;
		}
	}

	double random = starpu_drand48()*alpha_sum;
	//printf("my rand is %e over %e\n", random, alpha_sum);

	if(size == 0)
		return -ENODEV;

	unsigned selected = worker_arr[size - 1];

	double alpha = 0.0;
	int i;
	for(i = 0; i < size; i++)
	{
                worker = worker_arr[i];
		double worker_alpha = speedup_arr[i];
		
		if (alpha + worker_alpha >= random)
		{
			/* we found the worker */
			selected = worker;
			break;
		}
		
		alpha += worker_alpha;
	}
	STARPU_AYU_ADDTOTASKQUEUE(starpu_task_get_job_id(task), selected);
	starpu_sched_task_break(task);
	return starpu_push_local_task(selected, task, prio);
}

static int random_push_task(struct starpu_task *task)
{
        return _random_push_task(task, !!task->priority);
}

static void initialize_random_policy(unsigned sched_ctx_id)
{
	(void) sched_ctx_id;
	starpu_srand48(time(NULL));
}

static void deinitialize_random_policy(unsigned sched_ctx_id)
{
	(void) sched_ctx_id;
}

struct starpu_sched_policy _starpu_sched_random_policy =
{
	.init_sched = initialize_random_policy,
	.add_workers = NULL,
	.remove_workers = NULL,
	.deinit_sched = deinitialize_random_policy,
	.push_task = random_push_task,
	.pop_task = NULL,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "random",
	.policy_description = "weighted random based on worker overall performance",
	.worker_type = STARPU_WORKER_LIST,
};
