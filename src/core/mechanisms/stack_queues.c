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

#include <pthread.h>
#include <core/mechanisms/stack_queues.h>
#include <errno.h>

/* keep track of the total number of jobs to be scheduled to avoid infinite 
 * polling when there are really few jobs in the overall queue */
static unsigned total_number_of_jobs;

static pthread_cond_t *sched_cond;
static pthread_mutex_t *sched_mutex;

void init_stack_queues_mechanisms(void)
{
	total_number_of_jobs = 0;

	struct sched_policy_s *sched = get_sched_policy();

	/* to access them more easily, we keep their address in local variables */
	sched_cond = &sched->sched_activity_cond;
	sched_mutex = &sched->sched_activity_mutex;
}

struct jobq_s *create_stack(void)
{
	struct jobq_s *jobq;
	jobq = malloc(sizeof(struct jobq_s));

	struct stack_jobq_s *stack;
	stack = malloc(sizeof(struct stack_jobq_s));

	pthread_mutex_init(&jobq->activity_mutex, NULL);
	pthread_cond_init(&jobq->activity_cond, NULL);

	/* note that not all mechanisms (eg. the semaphore) have to be used */
	stack->jobq = job_list_new();
	stack->njobs = 0;
	stack->nprocessed = 0;

	stack->exp_start = timing_now();
	stack->exp_len = 0.0;
	stack->exp_end = stack->exp_start;

	jobq->queue = stack;

	return jobq;
}

unsigned get_total_njobs_stacks(void)
{
	return total_number_of_jobs;
}

unsigned get_stack_njobs(struct jobq_s *q)
{
	STARPU_ASSERT(q);

	struct stack_jobq_s *stack_queue = q->queue;

	return stack_queue->njobs;
}

unsigned get_stack_nprocessed(struct jobq_s *q)
{
	STARPU_ASSERT(q);

	struct stack_jobq_s *stack_queue = q->queue;

	return stack_queue->nprocessed;
}

void stack_push_prio_task(struct jobq_s *q, job_t task)
{
#ifndef NO_PRIO
	STARPU_ASSERT(q);
	struct stack_jobq_s *stack_queue = q->queue;

	/* if anyone is blocked on the entire machine, wake it up */
	pthread_mutex_lock(sched_mutex);
	total_number_of_jobs++;
	pthread_cond_signal(sched_cond);
	pthread_mutex_unlock(sched_mutex);

	/* wake people waiting locally */
	pthread_mutex_lock(&q->activity_mutex);

	TRACE_JOB_PUSH(task, 0);
	job_list_push_back(stack_queue->jobq, task);
	deque_queue->njobs++;
	deque_queue->nprocessed++;

	pthread_cond_signal(&q->activity_cond);
	pthread_mutex_unlock(&q->activity_mutex);
#else
	stack_push_task(q, task);
#endif
}

void stack_push_task(struct jobq_s *q, job_t task)
{
	STARPU_ASSERT(q);
	struct stack_jobq_s *stack_queue = q->queue;

	/* if anyone is blocked on the entire machine, wake it up */
	pthread_mutex_lock(sched_mutex);
	total_number_of_jobs++;
	pthread_cond_signal(sched_cond);
	pthread_mutex_unlock(sched_mutex);

	/* wake people waiting locally */
	pthread_mutex_lock(&q->activity_mutex);

	TRACE_JOB_PUSH(task, 0);
	job_list_push_front(stack_queue->jobq, task);
	deque_queue->njobs++;
	deque_queue->nprocessed++;

	pthread_cond_signal(&q->activity_cond);
	pthread_mutex_unlock(&q->activity_mutex);
}

job_t stack_pop_task(struct jobq_s *q)
{
	job_t j = NULL;

	STARPU_ASSERT(q);
	struct stack_jobq_s *stack_queue = q->queue;

	if (stack_queue->njobs == 0)
		return NULL;

	if (stack_queue->njobs > 0) 
	{
		/* there is a task */
		j = job_list_pop_back(stack_queue->jobq);
	
		STARPU_ASSERT(j);
		stack_queue->njobs--;
		
		TRACE_JOB_POP(j, 0);

		/* we are sure that we got it now, so at worst, some people thought 
		 * there remained some work and will soon discover it is not true */
		pthread_mutex_lock(sched_mutex);
		total_number_of_jobs--;
		pthread_mutex_unlock(sched_mutex);
	}
	
	return j;

}
