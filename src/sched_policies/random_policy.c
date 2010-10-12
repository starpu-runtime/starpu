/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

/* Policy attributing tasks randomly to workers */

#include <core/workers.h>
#include <sched_policies/fifo_queues.h>

static unsigned nworkers;
static struct starpu_fifo_taskq_s *queue_array[STARPU_NMAXWORKERS];

static pthread_cond_t sched_cond[STARPU_NMAXWORKERS];
static pthread_mutex_t sched_mutex[STARPU_NMAXWORKERS];

static struct starpu_task *random_pop_task(void)
{
	struct starpu_task *task;

	int workerid = starpu_worker_get_id();

	task = _starpu_fifo_pop_task(queue_array[workerid], -1);

	return task;
}

static int _random_push_task(struct starpu_task *task, unsigned prio)
{
	/* find the queue */
	unsigned worker;

	unsigned selected = 0;

	double alpha_sum = 0.0;

	for (worker = 0; worker < nworkers; worker++)
	{
		alpha_sum += _starpu_worker_get_relative_speedup(worker);
	}

	double random = starpu_drand48()*alpha_sum;
//	_STARPU_DEBUG("my rand is %e\n", random);

	double alpha = 0.0;
	for (worker = 0; worker < nworkers; worker++)
	{
		double worker_alpha = _starpu_worker_get_relative_speedup(worker);

		if (alpha + worker_alpha > random) {
			/* we found the worker */
			selected = worker;
			break;
		}

		alpha += worker_alpha;
	}

	/* we should now have the best worker in variable "selected" */
	if (prio) {
		return _starpu_fifo_push_prio_task(queue_array[selected], &sched_mutex[selected], &sched_cond[selected], task);
	} else {
		return _starpu_fifo_push_task(queue_array[selected],&sched_mutex[selected], &sched_cond[selected],  task);
	}
}

static int random_push_prio_task(struct starpu_task *task)
{
	return _random_push_task(task, 1);
}

static int random_push_task(struct starpu_task *task)
{
	return _random_push_task(task, 0);
}

static void initialize_random_policy(struct starpu_machine_topology_s *topology, 
	 __attribute__ ((unused)) struct starpu_sched_policy_s *_policy) 
{
	starpu_srand48(time(NULL));

	nworkers = topology->nworkers;

	unsigned workerid;
	for (workerid = 0; workerid < nworkers; workerid++)
	{
		queue_array[workerid] = _starpu_create_fifo();
	
		PTHREAD_MUTEX_INIT(&sched_mutex[workerid], NULL);
		PTHREAD_COND_INIT(&sched_cond[workerid], NULL);
	
		starpu_worker_set_sched_condition(workerid, &sched_cond[workerid], &sched_mutex[workerid]);
	}
}

struct starpu_sched_policy_s _starpu_sched_random_policy = {
	.init_sched = initialize_random_policy,
	.deinit_sched = NULL,
	.push_task = random_push_task,
	.push_prio_task = random_push_prio_task,
	.pop_task = random_pop_task,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "random",
	.policy_description = "weighted random"
};
