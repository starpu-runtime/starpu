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

#include <core/jobs.h>
#include <core/workers.h>
#include <core/dependencies/data-concurrency.h>
#include <common/config.h>

size_t job_get_data_size(job_t j)
{
	size_t size = 0;

	struct starpu_task *task = j->task;

	unsigned nbuffers = task->cl->nbuffers;

	unsigned buffer;
	for (buffer = 0; buffer < nbuffers; buffer++)
	{
		data_state *state = task->buffers[buffer].state;
		size += state->ops->get_size(state);
	}

	return size;
}

/* create an internal job_t structure to encapsulate the task */
job_t __attribute__((malloc)) job_create(struct starpu_task *task)
{
	job_t job;

	job = job_new();

	job->task = task;

	job->predicted = 0.0;
	job->footprint_is_computed = 0;
	job->terminated = 0;

	if (task->synchronous)
	{
#if defined(__APPLE__) && defined(__MACH__)
		pthread_mutex_init(&job->sync_mutex, NULL);
		pthread_cond_init(&job->sync_cond, NULL);
#else
		if (sem_init(&job->sync_sem, 0, 0))
			perror("sem_init");
#endif
	}

	if (task->use_tag)
		tag_declare(task->tag_id, job);

	return job;
}

struct starpu_task * __attribute__((malloc)) starpu_task_create(void)
{
	struct starpu_task *task;

	task = calloc(1, sizeof(struct starpu_task));
	STARPU_ASSERT(task);

	task->priority = DEFAULT_PRIO;

	/* by default, we let StarPU free the task structure */
	task->cleanup = 1;

	return task;
}

void handle_job_termination(job_t j)
{
	struct starpu_task *task = j->task;

	if (STARPU_UNLIKELY(j->terminated))
		fprintf(stderr, "OOPS ... job %p was already terminated !!\n", j);

	j->terminated = 1;

	/* in case there are dependencies, wake up the proper tasks */
	notify_dependencies(j);

	/* the callback is executed after the dependencies so that we may remove the tag 
 	 * of the task itself */
	if (task->callback_func)
	{
		TRACE_START_CALLBACK(j);
		task->callback_func(task->callback_arg);
		TRACE_END_CALLBACK(j);
	}

	if (task->synchronous)
	{
#if defined(__APPLE__) && defined(__MACH__)
		pthread_mutex_lock(&j->sync_mutex);
		pthread_cond_signal(&j->sync_cond);
		pthread_mutex_unlock(&j->sync_mutex);
#else
		if (sem_post(&j->sync_sem))
			perror("sem_post");
#endif

		/* as this is a synchronous task, we do not delete the job 
		   structure which contains the j->sync_sem: we only liberate
		   it once the semaphore is destroyed */
	}
	else
	{
		if (j->task->cleanup)
			free(j->task);

		job_delete(j);
	}

}

static void block_sync_task(job_t j)
{
#if defined(__APPLE__) && defined(__MACH__)
	pthread_mutex_lock(&j->sync_mutex);
	if (!j->terminated)
		pthread_cond_wait(&j->sync_cond, &j->sync_mutex);
	pthread_mutex_unlock(&j->sync_mutex);
#else
	sem_wait(&j->sync_sem);
	sem_destroy(&j->sync_sem);
#endif

	/* as this is a synchronous task, the liberation of the job
	   structure was deferred */
	if (j->task->cleanup)
		free(j->task);

	job_delete(j);
}


/* This function is called when a new task is submitted to StarPU 
 * it returns 1 if the task deps are not fulfilled, 0 otherwise */
static unsigned not_all_task_deps_are_fulfilled(job_t j)
{
	if (!j->task->use_tag)
	{
		/* this task does not use tags, so we can go on */
		return 0;
	}

	struct tag_s *tag = j->tag;

	take_mutex(&tag->lock);

	if (tag->ndeps != tag->ndeps_completed)
	{
		tag->state = BLOCKED;
		ret = 1;
	}
	else {
		/* existing deps (if any) are fulfilled */
		tag->state = READY;
		ret = 0;
	}

	release_mutex(&tag->lock);
	return ret;
}

static unsigned enforce_deps_and_schedule(job_t j)
{
	unsigned ret;

	/* enfore task dependencies */
	if (not_all_task_deps_are_fulfilled(j))
		return 0;

#ifdef NO_DATA_RW_LOCK
	/* enforce data dependencies */
	if (submit_job_enforce_data_deps(j))
		return 0;
#endif

	ret = push_task(j);

	return ret;
}

/* application should submit new tasks to StarPU through this function */
int starpu_submit_task(struct starpu_task *task)
{
	int ret;
	unsigned is_sync = task->synchronous;

	STARPU_ASSERT(task);

	if (!worker_exists(task->cl->where))
		return -ENODEV;

	/* internally, StarPU manipulates a job_t which is a wrapper around a
 	* task structure */
	job_t j = job_create(task);

	ret = enforce_deps_and_schedule(j);

	if (is_sync)
		block_sync_task(j);

	return ret;
}

/* This function is supplied for convenience only, it is equivalent to setting
 * the proper flag and submitting the task with submit_task.
 * Note that this call is blocking, and will not make StarPU progress,
 * so it must only be called from the programmer thread, not by StarPU.
 * NB: This also means that it cannot be submitted within a callback ! */
int submit_sync_task(struct starpu_task *task)
{
	task->synchronous = 1;

	return starpu_submit_task(task);
}
