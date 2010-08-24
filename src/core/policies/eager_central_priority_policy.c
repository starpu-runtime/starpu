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

#include <core/policies/eager_central_priority_policy.h>

/* the former is the actual queue, the latter some container */
static struct starpu_jobq_s *jobq;

static void init_priority_queue_design(void)
{
	/* only a single queue (even though there are several internaly) */
	jobq = _starpu_create_priority_jobq();

	_starpu_init_priority_queues_mechanisms();
}

static void deinit_priority_queue_design(void)
{
	/* TODO check that there is no task left in the queue */
	_starpu_deinit_priority_queues_mechanisms();

	/* deallocate the job queue */
	_starpu_destroy_priority_jobq(jobq);
}

static struct starpu_jobq_s *func_init_priority_queue(void)
{
	return jobq;
}

static void initialize_eager_center_priority_policy(struct starpu_machine_config_s *config, 
			__attribute__ ((unused))	struct starpu_sched_policy_s *_policy) 
{
	_starpu_setup_queues(init_priority_queue_design, func_init_priority_queue, config);
}

static void deinitialize_eager_center_priority_policy(struct starpu_machine_config_s *config, 
		   __attribute__ ((unused)) struct starpu_sched_policy_s *_policy) 
{
	_starpu_deinit_queues(deinit_priority_queue_design, NULL, config);
}

static struct starpu_jobq_s *get_local_queue_eager_priority(struct starpu_sched_policy_s *policy __attribute__ ((unused)))
{
	/* this is trivial for that strategy */
	return jobq;
}

struct starpu_sched_policy_s _starpu_sched_prio_policy = {
	.init_sched = initialize_eager_center_priority_policy,
	.deinit_sched = deinitialize_eager_center_priority_policy,
	.get_local_queue = get_local_queue_eager_priority,
	/* we always use priorities in that policy */
	.push_task = _starpu_priority_push_task,
	.push_prio_task = _starpu_priority_push_task,
	.pop_task = _starpu_priority_pop_task,
	.policy_name = "prio",
	.policy_description = "eager (with priorities)"
};
