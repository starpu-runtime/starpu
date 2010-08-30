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

#include <core/policies/eager_central_priority_policy.h>

/* the former is the actual queue, the latter some container */
static struct starpu_jobq_s *jobq;

/* keep track of the total number of jobs to be scheduled to avoid infinite 
 * polling when there are really few jobs in the overall queue */
static pthread_cond_t global_sched_cond;
static pthread_mutex_t global_sched_mutex;

static void initialize_eager_center_priority_policy(struct starpu_machine_config_s *config, 
			__attribute__ ((unused))	struct starpu_sched_policy_s *_policy) 
{
	/* only a single queue (even though there are several internaly) */
	jobq = _starpu_create_priority_jobq();

	PTHREAD_MUTEX_INIT(&global_sched_mutex, NULL);
	PTHREAD_COND_INIT(&global_sched_cond, NULL);

	int workerid;
	for (workerid = 0; workerid < STARPU_NMAXWORKERS; workerid++)
		starpu_worker_set_sched_condition(workerid, &global_sched_cond, &global_sched_mutex);
}

static void deinitialize_eager_center_priority_policy(struct starpu_machine_config_s *config, 
		   __attribute__ ((unused)) struct starpu_sched_policy_s *_policy) 
{
	/* TODO check that there is no task left in the queue */

	/* deallocate the job queue */
	_starpu_destroy_priority_jobq(jobq);
}

static int _starpu_priority_push_task(starpu_job_t j)
{
	struct starpu_priority_jobq_s *queue = jobq->queue;

	/* wake people waiting for a task */
	PTHREAD_MUTEX_LOCK(&global_sched_mutex);

	STARPU_TRACE_JOB_PUSH(j, 1);
	
	unsigned priolevel = j->task->priority - STARPU_MIN_PRIO;

	starpu_job_list_push_front(queue->jobq[priolevel], j);
	queue->njobs[priolevel]++;
	queue->total_njobs++;

	PTHREAD_COND_SIGNAL(&global_sched_cond);
	PTHREAD_MUTEX_UNLOCK(&global_sched_mutex);

	return 0;
}

static starpu_job_t _starpu_priority_pop_task(void)
{
	starpu_job_t j = NULL;

	struct starpu_priority_jobq_s *queue = jobq->queue;

	/* block until some event happens */
	PTHREAD_MUTEX_LOCK(&global_sched_mutex);

	if ((queue->total_njobs == 0) && _starpu_machine_is_running())
	{
#ifdef STARPU_NON_BLOCKING_DRIVERS
		_starpu_datawizard_progress(q->memory_node, 1);
#else
		PTHREAD_COND_WAIT(&global_sched_cond, &global_sched_mutex);
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
				STARPU_TRACE_JOB_POP(j, 0);
			}
		} while (!j && priolevel-- > 0);
	}

	PTHREAD_MUTEX_UNLOCK(&global_sched_mutex);

	return j;
}

struct starpu_sched_policy_s _starpu_sched_prio_policy = {
	.init_sched = initialize_eager_center_priority_policy,
	.deinit_sched = deinitialize_eager_center_priority_policy,
	/* we always use priorities in that policy */
	.push_task = _starpu_priority_push_task,
	.push_prio_task = _starpu_priority_push_task,
	.pop_task = _starpu_priority_pop_task,
	.policy_name = "prio",
	.policy_description = "eager (with priorities)"
};
