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
		data_state *state = task->buffers[buffer].handle;
		size += state->ops->get_size(state);
	}

	return size;
}

/* create an internal job_t structure to encapsulate the task */
static job_t __attribute__((malloc)) job_create(struct starpu_task *task)
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
	task->use_tag = 0;
	task->synchronous = 0;

	task->execute_on_a_specific_worker = 0;

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

static unsigned enforce_deps_and_schedule(job_t j)
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

void starpu_display_codelet_stats(struct starpu_codelet_t *cl)
{
	unsigned worker;
	unsigned nworkers = starpu_get_worker_count();

	if (cl->model && cl->model->symbol)
		fprintf(stderr, "Statistics for codelet %s\n", cl->model->symbol);

	unsigned long total = 0;
	
	for (worker = 0; worker < nworkers; worker++)
		total += cl->per_worker_stats[worker];

	for (worker = 0; worker < nworkers; worker++)
	{
		char name[32];
		starpu_get_worker_name(worker, name, 32);

		fprintf(stderr, "\t%s -> %ld / %ld (%2.2f \%%)\n", name, cl->per_worker_stats[worker], total, (100.0f*cl->per_worker_stats[worker])/total);
	}
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
