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

#include <core/policies/eager-central-priority-policy.h>

/* the former is the actual queue, the latter some container */
static struct starpu_jobq_s *jobq;

static void init_priority_queue_design(void)
{
	/* only a single queue (even though there are several internaly) */
	jobq = _starpu_create_priority_jobq();

	_starpu_init_priority_queues_mechanisms();

	/* we always use priorities in that policy */
	jobq->_starpu_push_task = _starpu_priority_push_task;
	jobq->push_prio_task = _starpu_priority_push_task;
	jobq->_starpu_pop_task = _starpu_priority_pop_task;
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

static struct starpu_jobq_s *get_local_queue_eager_priority(struct starpu_sched_policy_s *policy __attribute__ ((unused)))
{
	/* this is trivial for that strategy */
	return jobq;
}

struct starpu_sched_policy_s sched_prio_policy = {
	.init_sched = initialize_eager_center_priority_policy,
	.deinit_sched = NULL,
	._starpu_get_local_queue = get_local_queue_eager_priority,
	.policy_name = "prio",
	.policy_description = "eager (with priorities)"
};
