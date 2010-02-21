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

void _starpu_setup_queues(void (*init_queue_design)(void),
		  struct starpu_jobq_s *(*func_init_queue)(void), 
		  struct starpu_machine_config_s *config) 
{
	unsigned worker;

	if (init_queue_design)
		init_queue_design();

	for (worker = 0; worker < config->nworkers; worker++)
	{
		struct  starpu_worker_s *workerarg = &config->workers[worker];
		
		if (func_init_queue)
			workerarg->jobq = func_init_queue();
	}
}

void _starpu_deinit_queues(void (*deinit_queue_design)(void),
		  void (*func_deinit_queue)(struct starpu_jobq_s *q), 
		  struct starpu_machine_config_s *config)
{
	unsigned worker;

	for (worker = 0; worker < config->nworkers; worker++)
	{
		struct  starpu_worker_s *workerarg = &config->workers[worker];
		
		if (func_deinit_queue)
			func_deinit_queue(workerarg->jobq);
	}

	if (deinit_queue_design)
		deinit_queue_design();
}



/* this may return NULL for an "anonymous thread" */
struct starpu_jobq_s *_starpu_get_local_queue(void)
{
	struct starpu_sched_policy_s *policy = _starpu_get_sched_policy();

	return pthread_getspecific(policy->local_queue_key);
}

/* XXX how to retrieve policy ? that may be given in the machine config ? */
void _starpu_set_local_queue(struct starpu_jobq_s *jobq)
{
	struct starpu_sched_policy_s *policy = _starpu_get_sched_policy();

	pthread_setspecific(policy->local_queue_key, jobq);
}

void _starpu_jobq_lock(struct starpu_jobq_s *jobq)
{
	pthread_mutex_lock(&jobq->activity_mutex);	
}

void _starpu_jobq_unlock(struct starpu_jobq_s *jobq)
{
	pthread_mutex_unlock(&jobq->activity_mutex);	
}

int _starpu_jobq_trylock(struct starpu_jobq_s *jobq)
{
	return pthread_mutex_trylock(&jobq->activity_mutex);	
}
