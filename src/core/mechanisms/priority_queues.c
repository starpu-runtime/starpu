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

#include <starpu.h>
#include <common/config.h>
#include <core/mechanisms/priority_queues.h>

/*
 * Centralized queue with priorities 
 */


/* keep track of the total number of jobs to be scheduled to avoid infinite 
 * polling when there are really few jobs in the overall queue */
static pthread_cond_t *sched_cond;
static pthread_mutex_t *sched_mutex;

void _starpu_init_priority_queues_mechanisms(void)
{
	struct starpu_sched_policy_s *sched = _starpu_get_sched_policy();

	/* to access them more easily, we keep their address in local variables */
	sched_cond = &sched->sched_activity_cond;
	sched_mutex = &sched->sched_activity_mutex;
}

void _starpu_deinit_priority_queues_mechanisms(void)
{
}

struct starpu_jobq_s *_starpu_create_priority_jobq(void)
{
	struct starpu_jobq_s *q;

	q = malloc(sizeof(struct starpu_jobq_s));

	struct starpu_priority_jobq_s *central_queue;
	
	central_queue = malloc(sizeof(struct starpu_priority_jobq_s));
	q->queue = central_queue;

	pthread_mutex_init(&q->activity_mutex, NULL);
	pthread_cond_init(&q->activity_cond, NULL);

	central_queue->total_njobs = 0;

	unsigned prio;
	for (prio = 0; prio < NPRIO_LEVELS; prio++)
	{
		central_queue->jobq[prio] = starpu_job_list_new();
		central_queue->njobs[prio] = 0;
	}

	return q;
}

void _starpu_destroy_priority_jobq(struct starpu_jobq_s *jobq)
{
	struct starpu_priority_jobq_s *central_queue;

	central_queue = jobq->queue;

	unsigned prio;
	for (prio = 0; prio < NPRIO_LEVELS; prio++)
		starpu_job_list_delete(central_queue->jobq[prio]);

	free(central_queue);

	free(jobq);
}

int _starpu_priority_push_task(struct starpu_jobq_s *q, starpu_job_t j)
{
	STARPU_ASSERT(q);
	struct starpu_priority_jobq_s *queue = q->queue;

	/* if anyone is blocked on the entire machine, wake it up */
	pthread_mutex_lock(sched_mutex);
	pthread_cond_signal(sched_cond);
	pthread_mutex_unlock(sched_mutex);

	/* wake people waiting locally */
	pthread_mutex_lock(&q->activity_mutex);

	TRACE_JOB_PUSH(j, 1);
	
	unsigned priolevel = j->task->priority - STARPU_MIN_PRIO;

	starpu_job_list_push_front(queue->jobq[priolevel], j);
	queue->njobs[priolevel]++;
	queue->total_njobs++;

	pthread_cond_signal(&q->activity_cond);
	pthread_mutex_unlock(&q->activity_mutex);

	return 0;
}

starpu_job_t _starpu_priority_pop_task(struct starpu_jobq_s *q)
{
	starpu_job_t j = NULL;

	STARPU_ASSERT(q);
	struct starpu_priority_jobq_s *queue = q->queue;

	/* block until some event happens */
	pthread_mutex_lock(&q->activity_mutex);

	if ((queue->total_njobs == 0) && _starpu_machine_is_running())
	{
#ifdef STARPU_NON_BLOCKING_DRIVERS
		_starpu_datawizard_progress(q->memory_node, 1);
#else
		pthread_cond_wait(&q->activity_cond, &q->activity_mutex);
#endif
	}

	if (queue->total_njobs > 0)
	{
		unsigned priolevel = NPRIO_LEVELS - 1;
		do {
			if (queue->njobs[priolevel] > 0) {
				/* there is some task that we can grab */
				j = starpu_job_list_pop_back(queue->jobq[priolevel]);
				queue->njobs[priolevel]--;
				queue->total_njobs--;
				TRACE_JOB_POP(j, 0);
			}
		} while (!j && priolevel-- > 0);
	}

	pthread_mutex_unlock(&q->activity_mutex);

	return j;
}
