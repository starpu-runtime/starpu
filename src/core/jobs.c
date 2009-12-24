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
#include <core/task.h>
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
		starpu_data_handle handle = task->buffers[buffer].handle;
		size += handle->ops->get_size(handle);
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

	pthread_mutex_init(&job->sync_mutex, NULL);
	pthread_cond_init(&job->sync_cond, NULL);

	if (task->use_tag)
		tag_declare(task->tag_id, job);

	return job;
}

void starpu_wait_job(job_t j)
{
	STARPU_ASSERT(j->task);
	STARPU_ASSERT(!j->task->detach);

	pthread_mutex_lock(&j->sync_mutex);
	if (!j->terminated)
		pthread_cond_wait(&j->sync_cond, &j->sync_mutex);
	pthread_mutex_unlock(&j->sync_mutex);

	job_delete(j);
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
		/* so that we can check whether we are doing blocking calls
		 * within the callback */
		set_local_worker_status(STATUS_CALLBACK);

		TRACE_START_CALLBACK(j);
		task->callback_func(task->callback_arg);
		TRACE_END_CALLBACK(j);

		set_local_worker_status(STATUS_UNKNOWN);
	}

	if (!task->detach)
	{
		/* we do not desallocate the job structure if some is going to
		 * wait after the task */
		pthread_mutex_lock(&j->sync_mutex);
		pthread_cond_broadcast(&j->sync_cond);
		pthread_mutex_unlock(&j->sync_mutex);
	}
	else {
		/* no one is going to synchronize with that task so we release
 		 * the data structures now */
		if (task->detach)
			job_delete(j);

		if (task->destroy)
			free(task);
	}

	decrement_nsubmitted_tasks();
}

/* This function is called when a new task is submitted to StarPU 
 * it returns 1 if the task deps are not fulfilled, 0 otherwise */
static unsigned not_all_task_deps_are_fulfilled(job_t j)
{
	unsigned ret;

	if (!j->task->use_tag)
	{
		/* this task does not use tags, so we can go on */
		return 0;
	}

	struct tag_s *tag = j->tag;

	starpu_spin_lock(&tag->lock);

	if (tag->ndeps != tag->ndeps_completed)
	{
		tag->state = BLOCKED;
		ret = 1;
	}
	else {
		/* existing deps (if any) are fulfilled */
		tag->state = READY;
		/* already prepare for next run */
		tag->ndeps_completed = 0;
		ret = 0;
	}

	starpu_spin_unlock(&tag->lock);
	return ret;
}

unsigned enforce_deps_and_schedule(job_t j)
{
	unsigned ret;

	/* enfore task dependencies */
	if (not_all_task_deps_are_fulfilled(j))
		return 0;

	/* enforce data dependencies */
	if (submit_job_enforce_data_deps(j))
		return 0;

	ret = push_task(j);

	return ret;
}

struct job_s *pop_local_task(struct worker_s *worker)
{
	struct job_s *j = NULL;

	pthread_mutex_lock(&worker->local_jobs_mutex);

	if (!job_list_empty(worker->local_jobs))
		j = job_list_pop_back(worker->local_jobs);

	pthread_mutex_unlock(&worker->local_jobs_mutex);

	return j;
}

int push_local_task(struct worker_s *worker, struct job_s *j)
{
	/* TODO check that the worker is able to execute the task ! */

	pthread_mutex_lock(&worker->local_jobs_mutex);

	job_list_push_front(worker->local_jobs, j);

	pthread_mutex_unlock(&worker->local_jobs_mutex);

	/* XXX that's a bit excessive ... */
	wake_all_blocked_workers_on_node(worker->memory_node);

	return 0;
}
