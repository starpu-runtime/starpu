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

#include <core/policies/no_prio_policy.h>

/*
 *	This is just the trivial policy where every worker use the same
 *	JOB QUEUE.
 */

/* the former is the actual queue, the latter some container */
static struct starpu_fifo_taskq_s *fifo;

static pthread_cond_t sched_cond;
static pthread_mutex_t sched_mutex;

static void initialize_no_prio_policy(struct starpu_machine_topology_s *topology, 
	   __attribute__ ((unused)) struct starpu_sched_policy_s *_policy) 
{
	/* there is only a single queue in that trivial design */
	fifo = _starpu_create_fifo();

	PTHREAD_MUTEX_INIT(&sched_mutex, NULL);
	PTHREAD_COND_INIT(&sched_cond, NULL);

	unsigned workerid;
	for (workerid = 0; workerid < topology->nworkers; workerid++)
		starpu_worker_set_sched_condition(workerid, &sched_cond, &sched_mutex);
}

static int push_task_no_prio_policy(struct starpu_task *task)
{
        return _starpu_fifo_push_task(fifo, &sched_mutex, &sched_cond, task);
}

static struct starpu_task *pop_task_no_prio_policy(void)
{
	return _starpu_fifo_pop_task(fifo);
}

struct starpu_sched_policy_s _starpu_sched_no_prio_policy = {
	.init_sched = initialize_no_prio_policy,
	.deinit_sched = NULL,
	.push_task = push_task_no_prio_policy,
	.push_prio_task = push_task_no_prio_policy,
	.pop_task = pop_task_no_prio_policy,
	.policy_name = "no-prio",
	.policy_description = "eager without priority"
};
