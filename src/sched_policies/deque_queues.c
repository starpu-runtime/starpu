/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
 * Copyright (C) 2010  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

/* Deque queues, ready for use by schedulers */

#include <starpu.h>
#include <common/config.h>
#include <core/workers.h>
#include <sched_policies/deque_queues.h>
#include <errno.h>
#include <common/utils.h>

struct starpu_deque_jobq_s *_starpu_create_deque(void)
{
	struct starpu_deque_jobq_s *deque;
	deque = (struct starpu_deque_jobq_s *) malloc(sizeof(struct starpu_deque_jobq_s));

	/* note that not all mechanisms (eg. the semaphore) have to be used */
	deque->jobq = starpu_job_list_new();
	deque->njobs = 0;
	deque->nprocessed = 0;

	deque->exp_start = starpu_timing_now();
	deque->exp_len = 0.0;
	deque->exp_end = deque->exp_start;

	return deque;
}

void _starpu_destroy_deque(struct starpu_deque_jobq_s *deque)
{
	starpu_job_list_delete(deque->jobq);
	free(deque);
}

unsigned _starpu_get_deque_njobs(struct starpu_deque_jobq_s *deque_queue)
{
	return deque_queue->njobs;
}

unsigned _starpu_get_deque_nprocessed(struct starpu_deque_jobq_s *deque_queue)
{
	return deque_queue->nprocessed;
}

struct starpu_task *_starpu_deque_pop_task(struct starpu_deque_jobq_s *deque_queue, int workerid __attribute__ ((unused)))
{
	starpu_job_t j = NULL;

	if ((deque_queue->njobs == 0) && _starpu_machine_is_running())
	{
		return NULL;
	}

	/* TODO find a task that suits workerid */
	if (deque_queue->njobs > 0) 
	{
		/* there is a task */
		j = starpu_job_list_pop_front(deque_queue->jobq);
	
		STARPU_ASSERT(j);
		deque_queue->njobs--;
		
		STARPU_TRACE_JOB_POP(j, 0);

		return j->task;
	}

	return NULL;
}

struct starpu_job_list_s *_starpu_deque_pop_every_task(struct starpu_deque_jobq_s *deque_queue, pthread_mutex_t *sched_mutex, int workerid)
{
	struct starpu_job_list_s *new_list, *old_list;

	/* block until some task is available in that queue */
	_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);

	if (deque_queue->njobs == 0)
	{
		new_list = NULL;
	}
	else {
		/* there is a task */
		old_list = deque_queue->jobq;
		new_list = starpu_job_list_new();

		unsigned new_list_size = 0;

		starpu_job_itor_t i;
		starpu_job_t next_job;
		/* note that this starts at the _head_ of the list, so we put
 		 * elements at the back of the new list */
		for(i = starpu_job_list_begin(old_list);
			i != starpu_job_list_end(old_list);
			i  = next_job)
		{
			next_job = starpu_job_list_next(i);

			/* In case there are multiples implementations of the
 			 * codelet for a single device, We dont really care
			 * about the implementation used, so let's try the 
			 * first one. */
			if (starpu_worker_may_execute_task(workerid, i->task, 0))
			{
				/* this elements can be moved into the new list */
				new_list_size++;
				
				starpu_job_list_erase(old_list, i);
				starpu_job_list_push_back(new_list, i);
			}
		}

		if (new_list_size == 0)
		{
			/* the new list is empty ... */
			starpu_job_list_delete(new_list);
			new_list = NULL;
		}
		else
		{
			deque_queue->njobs -= new_list_size;
		}
	}
	
	_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);

	return new_list;
}
