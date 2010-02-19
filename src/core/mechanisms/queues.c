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

#include "queues.h"

/*
 * There can be various queue designs
 * 	- trivial single list
 * 	- cilk-like 
 * 	- hierarchical (marcel-like)
 */

void setup_queues(void (*init_queue_design)(void),
		  struct jobq_s *(*func_init_queue)(void), 
		  struct starpu_machine_config_s *config) 
{
	unsigned worker;

	init_queue_design();

	for (worker = 0; worker < config->nworkers; worker++)
	{
		struct  starpu_worker_s *workerarg = &config->workers[worker];
		
		workerarg->jobq = func_init_queue();
	}
}

/* this may return NULL for an "anonymous thread" */
struct jobq_s *get_local_queue(void)
{
	struct sched_policy_s *policy = get_sched_policy();

	return pthread_getspecific(policy->local_queue_key);
}

/* XXX how to retrieve policy ? that may be given in the machine config ? */
void set_local_queue(struct jobq_s *jobq)
{
	struct sched_policy_s *policy = get_sched_policy();

	pthread_setspecific(policy->local_queue_key, jobq);
}

void jobq_lock(struct jobq_s *jobq)
{
	pthread_mutex_lock(&jobq->activity_mutex);	
}

void jobq_unlock(struct jobq_s *jobq)
{
	pthread_mutex_unlock(&jobq->activity_mutex);	
}

int jobq_trylock(struct jobq_s *jobq)
{
	return pthread_mutex_trylock(&jobq->activity_mutex);	
}
