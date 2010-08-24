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

#ifndef __SCHED_POLICY_H__
#define __SCHED_POLICY_H__

#include <starpu.h>
#include <core/mechanisms/queues.h>
//#include <core/mechanisms/work_stealing_queues.h>
//#include <core/mechanisms/central_queues.h>
//#include <core/mechanisms/central_queues_priorities.h>

#include <core/workers.h>

struct starpu_machine_config_s;

struct starpu_sched_policy_s {
	/* create all the queues */
	void (*init_sched)(struct starpu_machine_config_s *, struct starpu_sched_policy_s *);

	/* cleanup method at termination */
	void (*deinit_sched)(struct starpu_machine_config_s *, struct starpu_sched_policy_s *);

	/* anyone can request which queue it is associated to */
	struct starpu_jobq_s *(*get_local_queue)(struct starpu_sched_policy_s *);

	/* some methods to manipulate the previous queue */
	int (*push_task)(struct starpu_jobq_s *, starpu_job_t);
	int (*push_prio_task)(struct starpu_jobq_s *, starpu_job_t);
	struct starpu_job_s* (*pop_task)(struct starpu_jobq_s *);

	/* returns the number of tasks that were retrieved 
 	 * the function is reponsible for allocating the output but the driver
 	 * has to free it 
 	 *
 	 * NB : this function is non blocking
 	 * */
	struct starpu_job_list_s *(*pop_every_task)(struct starpu_jobq_s *, uint32_t);

	/* name of the policy (optionnal) */
	const char *policy_name;

	/* description of the policy (optionnal) */
	const char *policy_description;

	/* some worker may block until some activity happens in the machine */
	pthread_cond_t sched_activity_cond;
	pthread_mutex_t sched_activity_mutex;

	pthread_key_t local_queue_key;
};

struct starpu_sched_policy_s *_starpu_get_sched_policy(void);

void _starpu_init_sched_policy(struct starpu_machine_config_s *config);
void _starpu_deinit_sched_policy(struct starpu_machine_config_s *config);
//void _starpu_set_local_queue(struct starpu_jobq_s *jobq);

int _starpu_get_prefetch_flag(void);

int _starpu_push_task(starpu_job_t task, unsigned job_is_already_locked);
struct starpu_job_s *_starpu_pop_task(void);
struct starpu_job_s *_starpu_pop_task_from_queue(struct starpu_jobq_s *queue);
struct starpu_job_list_s *_starpu_pop_every_task(uint32_t where);
struct starpu_job_list_s * _starpu_pop_every_task_from_queue(struct starpu_jobq_s *queue, uint32_t where);

void _starpu_wait_on_sched_event(void);

#endif // __SCHED_POLICY_H__
