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

struct jobq_s {
	/* a pointer to some queue structure */
	void *queue; 

	/* some methods to manipulate the previous queue */
	int (*push_task)(struct jobq_s *, job_t);
	int (*push_prio_task)(struct jobq_s *, job_t);
	struct job_s* (*pop_task)(struct jobq_s *);

	/* returns the number of tasks that were retrieved 
 	 * the function is reponsible for allocating the output but the driver
 	 * has to free it 
 	 *
 	 * NB : this function is non blocking
 	 * */
	struct job_list_s *(*pop_every_task)(struct jobq_s *);

	/* what are the driver that may pop job from that queue ? */
	uint32_t who;

	/* this is only relevant if there is a single worker per queue */
	uint32_t memory_node;
	enum starpu_perf_archtype arch;
	float alpha;

	/* for performance analysis purpose */
	double total_computation_time;
	double total_communication_time;

	/* in case workers are blocked on the queue, signaling on that 
	  condition must unblock them, even if there is no available task */
	pthread_cond_t activity_cond;
	pthread_mutex_t activity_mutex;
};

struct machine_config_s;

void setup_queues(void (*init_queue_design)(void),
                  struct jobq_s *(*func_init_queue)(void),
                  struct machine_config_s *config);

struct jobq_s *get_local_queue(void);
void set_local_queue(struct jobq_s *jobq);


#endif // __QUEUES_H__
