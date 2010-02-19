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

#ifndef __QUEUES_H__
#define __QUEUES_H__

#include <pthread.h>

#include <core/jobs.h>
#include <core/policies/sched_policy.h>

enum starpu_perf_archtype;

struct starpu_jobq_s {
	/* a pointer to some queue structure */
	void *queue; 

	/* some methods to manipulate the previous queue */
	int (*_starpu_push_task)(struct starpu_jobq_s *, starpu_job_t);
	int (*push_prio_task)(struct starpu_jobq_s *, starpu_job_t);
	struct starpu_job_s* (*_starpu_pop_task)(struct starpu_jobq_s *);

	/* returns the number of tasks that were retrieved 
 	 * the function is reponsible for allocating the output but the driver
 	 * has to free it 
 	 *
 	 * NB : this function is non blocking
 	 * */
	struct starpu_job_list_s *(*_starpu_pop_every_task)(struct starpu_jobq_s *, uint32_t);

	/* what are the driver that may pop job from that queue ? */
	uint32_t who;

	/* this is only relevant if there is a single worker per queue */
	uint32_t memory_node;
	enum starpu_perf_archtype arch;
	float alpha;

	/* for performance analysis purpose */
	double total_computation_time;
	double total_communication_time;
	double total_computation_time_error;
	unsigned total_job_performed;

	/* in case workers are blocked on the queue, signaling on that 
	  condition must unblock them, even if there is no available task */
	pthread_cond_t activity_cond;
	pthread_mutex_t activity_mutex;
};

struct starpu_machine_config_s;

void _starpu_setup_queues(void (*init_queue_design)(void),
                  struct starpu_jobq_s *(*func_init_queue)(void),
                  struct starpu_machine_config_s *config);

struct starpu_jobq_s *_starpu_get_local_queue(void);
void _starpu_set_local_queue(struct starpu_jobq_s *jobq);

void _starpu_jobq_lock(struct starpu_jobq_s *jobq);
void _starpu_jobq_unlock(struct starpu_jobq_s *jobq);
int _starpu_jobq_trylock(struct starpu_jobq_s *jobq);

#endif // __QUEUES_H__
