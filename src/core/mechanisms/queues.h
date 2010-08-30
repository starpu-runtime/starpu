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

#ifndef __QUEUES_H__
#define __QUEUES_H__

#include <pthread.h>

#include <core/jobs.h>
#include <core/policies/sched_policy.h>

struct starpu_jobq_s {
	/* a pointer to some queue structure */
	void *queue; 

//	/* in case workers are blocked on the queue, signaling on that 
//	  condition must unblock them, even if there is no available task */
//	pthread_cond_t activity_cond;
//	pthread_mutex_t activity_mutex;
};

struct starpu_machine_config_s;

void _starpu_setup_queues(void (*init_queue_design)(void),
                  struct starpu_jobq_s *(*func_init_queue)(void),
                  struct starpu_machine_config_s *config);
void _starpu_deinit_queues(void (*deinit_queue_design)(void),
		  void (*func_deinit_queue)(struct starpu_jobq_s *q), 
		  struct starpu_machine_config_s *config);

#endif // __QUEUES_H__
