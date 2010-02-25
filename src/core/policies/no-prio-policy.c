/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#include <core/policies/no-prio-policy.h>

/*
 *	This is just the trivial policy where every worker use the same
 *	JOB QUEUE.
 */

/* the former is the actual queue, the latter some container */
static struct starpu_jobq_s *jobq;

static void init_no_prio_design(void)
{
	/* there is only a single queue in that trivial design */
	jobq = _starpu_create_fifo();

	_starpu_init_fifo_queues_mechanisms();

	jobq->_starpu_push_task = _starpu_fifo_push_task;
	/* no priority in that policy, let's be stupid here */
	jobq->push_prio_task = _starpu_fifo_push_task;
	jobq->_starpu_pop_task = _starpu_fifo_pop_task;
}

static struct starpu_jobq_s *func_init_central_queue(void)
{
	/* once again, this is trivial */
	return jobq;
}

static void initialize_no_prio_policy(struct starpu_machine_config_s *config, 
	   __attribute__ ((unused)) struct starpu_sched_policy_s *_policy) 
{
	_starpu_setup_queues(init_no_prio_design, func_init_central_queue, config);
}

static struct starpu_jobq_s *get_local_queue_no_prio(struct starpu_sched_policy_s *policy 
					__attribute__ ((unused)))
{
	/* this is trivial for that strategy :) */
	return jobq;
}

struct starpu_sched_policy_s _starpu_sched_no_prio_policy = {
	.init_sched = initialize_no_prio_policy,
	.deinit_sched = NULL,
	._starpu_get_local_queue = get_local_queue_no_prio,
	.policy_name = "no-prio",
	.policy_description = "eager without priority"
};
