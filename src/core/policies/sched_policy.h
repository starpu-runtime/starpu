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

#ifndef __SCHED_POLICY_H__
#define __SCHED_POLICY_H__

#include <starpu.h>
#include <core/mechanisms/queues.h>
//#include <core/mechanisms/work_stealing_queues.h>
//#include <core/mechanisms/central_queues.h>
//#include <core/mechanisms/central_queues_priorities.h>

#include <core/workers.h>

struct machine_config_s;

struct sched_policy_s {
	/* create all the queues */
	void (*init_sched)(struct machine_config_s *, struct sched_policy_s *);

	/* cleanup method at termination */
	void (*deinit_sched)(struct machine_config_s *, struct sched_policy_s *);

	/* anyone can request which queue it is associated to */
	struct jobq_s *(*get_local_queue)(struct sched_policy_s *);

	/* name of the policy (optionnal) */
	const char *policy_name;

	/* description of the policy (optionnal) */
	const char *policy_description;

	/* some worker may block until some activity happens in the machine */
	pthread_cond_t sched_activity_cond;
	pthread_mutex_t sched_activity_mutex;

	pthread_key_t local_queue_key;
};

struct sched_policy_s *get_sched_policy(void);

void init_sched_policy(struct machine_config_s *config);
void deinit_sched_policy(struct machine_config_s *config);
//void set_local_queue(struct jobq_s *jobq);

int push_task(starpu_job_t task);
struct starpu_job_s *pop_task(void);
struct starpu_job_s *pop_task_from_queue(struct jobq_s *queue);
struct starpu_job_list_s *pop_every_task(uint32_t where);
struct starpu_job_list_s * pop_every_task_from_queue(struct jobq_s *queue, uint32_t where);

void wait_on_sched_event(void);

#endif // __SCHED_POLICY_H__
