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

#include <starpu.h>
#include <core/jobs.h>
#include <core/task.h>
#include <core/workers.h>
#include <core/dependencies/data_concurrency.h>
#include <common/config.h>
#include <common/utils.h>
#include <profiling/profiling.h>
#include <profiling/bound.h>

size_t _starpu_job_get_data_size(starpu_job_t j)
{
	size_t size = 0;

	struct starpu_task *task = j->task;

	unsigned nbuffers = task->cl->nbuffers;

	unsigned buffer;
	for (buffer = 0; buffer < nbuffers; buffer++)
	{
		starpu_data_handle handle = task->buffers[buffer].handle;
		size += _starpu_data_get_size(handle);
	}

	return size;
}

/* we need to identify each task to generate the DAG. */
static unsigned job_cnt = 0;

void _starpu_exclude_task_from_dag(struct starpu_task *task)
{
	starpu_job_t j = _starpu_get_job_associated_to_task(task);

	j->exclude_from_dag = 1;
}

/* create an internal starpu_job_t structure to encapsulate the task */
starpu_job_t __attribute__((malloc)) _starpu_job_create(struct starpu_task *task)
{
	starpu_job_t job;
        _STARPU_LOG_IN();

	job = starpu_job_new();

	job->task = task;

	job->footprint_is_computed = 0;
	job->submitted = 0;
	job->terminated = 0;

#ifndef STARPU_USE_FXT
	if (_starpu_bound_recording)
#endif
		job->job_id = STARPU_ATOMIC_ADD(&job_cnt, 1);
#ifdef STARPU_USE_FXT
	/* display all tasks by default */
        job->model_name = NULL;
#endif
	job->exclude_from_dag = 0;

	job->reduction_task = 0;

	_starpu_cg_list_init(&job->job_successors);

	PTHREAD_MUTEX_INIT(&job->sync_mutex, NULL);
	PTHREAD_COND_INIT(&job->sync_cond, NULL);

	job->bound_task = NULL;

	if (task->use_tag)
		_starpu_tag_declare(task->tag_id, job);

        _STARPU_LOG_OUT();
	return job;
}

void _starpu_job_destroy(starpu_job_t j)
{
	PTHREAD_COND_DESTROY(&j->sync_cond);
	PTHREAD_MUTEX_DESTROY(&j->sync_mutex);

	_starpu_cg_list_deinit(&j->job_successors);

	starpu_job_delete(j);
}

void _starpu_wait_job(starpu_job_t j)
{
	STARPU_ASSERT(j->task);
	STARPU_ASSERT(!j->task->detach);
        _STARPU_LOG_IN();

	PTHREAD_MUTEX_LOCK(&j->sync_mutex);

	/* We wait for the flag to have a value of 2 which means that both the
	 * codelet's implementation and its callback have been executed. That
	 * way, _starpu_wait_job won't return until the entire task was really
	 * executed (so that we cannot destroy the task while it is still being
	 * manipulated by the driver). */
	while (j->terminated != 2)
		PTHREAD_COND_WAIT(&j->sync_cond, &j->sync_mutex);

	PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
        _STARPU_LOG_OUT();
}

void _starpu_handle_job_termination(starpu_job_t j, unsigned job_is_already_locked)
{
	struct starpu_task *task = j->task;

	if (!job_is_already_locked)
		PTHREAD_MUTEX_LOCK(&j->sync_mutex);

	task->status = STARPU_TASK_FINISHED;

	/* in case there are dependencies, wake up the proper tasks */
	j->submitted = 0;
	_starpu_notify_dependencies(j);

	/* We must have set the j->terminated flag early, so that it is
	 * possible to express task dependencies within the callback
	 * function. A value of 1 means that the codelet was executed but that
	 * the callback is not done yet. */
	j->terminated = 1;

	if (!job_is_already_locked)
		PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

	/* the callback is executed after the dependencies so that we may remove the tag 
 	 * of the task itself */
	if (task->callback_func)
	{
		int profiling = starpu_profiling_status_get();
		if (profiling && task->profiling_info)
			starpu_clock_gettime(&task->profiling_info->callback_start_time);

		/* so that we can check whether we are doing blocking calls
		 * within the callback */
		_starpu_set_local_worker_status(STATUS_CALLBACK);
		
		
		/* Perhaps we have nested callbacks (eg. with chains of empty
		 * tasks). So we store the current task and we will restore it
		 * later. */
		struct starpu_task *current_task = starpu_get_current_task();

		_starpu_set_current_task(task);

		STARPU_TRACE_START_CALLBACK(j);
		task->callback_func(task->callback_arg);
		STARPU_TRACE_END_CALLBACK(j);
		
		_starpu_set_current_task(current_task);

		_starpu_set_local_worker_status(STATUS_UNKNOWN);

		if (profiling && task->profiling_info)
			starpu_clock_gettime(&task->profiling_info->callback_end_time);
	}

	_starpu_sched_post_exec_hook(task);

	STARPU_TRACE_TASK_DONE(j);

	/* NB: we do not save those values before the callback, in case the
	 * application changes some parameters eventually (eg. a task may not
	 * be generated if the application is terminated). */
	int destroy = task->destroy;
	int detach = task->detach;
	int regenerate = task->regenerate;

	if (!detach)
	{
		/* we do not desallocate the job structure if some is going to
		 * wait after the task */
		if (!job_is_already_locked)
			PTHREAD_MUTEX_LOCK(&j->sync_mutex);
		/* A value of 2 is put to specify that not only the codelet but
		 * also the callback were executed. */
		j->terminated = 2;
		PTHREAD_COND_BROADCAST(&j->sync_cond);

		if (!job_is_already_locked)
			PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
	}
	else {
		/* no one is going to synchronize with that task so we release
		 * the data structures now. In case the job was already locked
		 * by the caller, it is its responsability to destroy the task.
		 * */
		if (!job_is_already_locked && destroy)
			starpu_task_destroy(task);
	}

	if (regenerate)
	{
		STARPU_ASSERT(detach && !destroy && !task->synchronous);

		/* We reuse the same job structure */
		int ret = _starpu_submit_job(j, 1);
		STARPU_ASSERT(!ret);
	}	
	else {
		_starpu_decrement_nsubmitted_tasks();
	}
}

/* This function is called when a new task is submitted to StarPU 
 * it returns 1 if the tag deps are not fulfilled, 0 otherwise */
static unsigned _starpu_not_all_tag_deps_are_fulfilled(starpu_job_t j)
{
	unsigned ret;

	if (!j->task->use_tag)
	{
		/* this task does not use tags, so we can go on */
		return 0;
	}

	struct starpu_tag_s *tag = j->tag;

	struct starpu_cg_list_s *tag_successors = &tag->tag_successors;

	_starpu_spin_lock(&tag->lock);

	if (tag_successors->ndeps != tag_successors->ndeps_completed)
	{
		tag->state = STARPU_BLOCKED;
                j->task->status = STARPU_TASK_BLOCKED_ON_TAG;
		ret = 1;
	}
	else {
		/* existing deps (if any) are fulfilled */
		tag->state = STARPU_READY;
		/* already prepare for next run */
		tag_successors->ndeps_completed = 0;
		ret = 0;
	}

	_starpu_spin_unlock(&tag->lock);
	return ret;
}

static unsigned _starpu_not_all_task_deps_are_fulfilled(starpu_job_t j, unsigned job_is_already_locked)
{
	unsigned ret;

	struct starpu_cg_list_s *job_successors = &j->job_successors;

	if (!job_is_already_locked)
		PTHREAD_MUTEX_LOCK(&j->sync_mutex);	

	if (!j->submitted || (job_successors->ndeps != job_successors->ndeps_completed))
	{
                j->task->status = STARPU_TASK_BLOCKED_ON_TASK;
		ret = 1;
	}
	else {
		/* existing deps (if any) are fulfilled */
		/* already prepare for next run */
		job_successors->ndeps_completed = 0;
		ret = 0;
	}

	if (!job_is_already_locked)
		PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

	return ret;
}



/*
 *	In order, we enforce tag, task and data dependencies. The task is
 *	passed to the scheduler only once all these constraints are fulfilled.
 */
unsigned _starpu_enforce_deps_and_schedule(starpu_job_t j, unsigned job_is_already_locked)
{
	unsigned ret;
        _STARPU_LOG_IN();

	/* enfore tag dependencies */
	if (_starpu_not_all_tag_deps_are_fulfilled(j)) {
                _STARPU_LOG_OUT_TAG("not_all_tag_deps_are_fulfilled");
		return 0;
        }

	/* enfore task dependencies */
	if (_starpu_not_all_task_deps_are_fulfilled(j, job_is_already_locked)) {
                _STARPU_LOG_OUT_TAG("not_all_task_deps_are_fulfilled");
		return 0;
        }

	/* enforce data dependencies */
	if (_starpu_submit_job_enforce_data_deps(j)) {
                _STARPU_LOG_OUT_TAG("enforce_data_deps");
		return 0;
        }

	ret = _starpu_push_task(j, job_is_already_locked);

        _STARPU_LOG_OUT();
	return ret;
}

/* Tag deps are already fulfilled */
unsigned _starpu_enforce_deps_starting_from_task(starpu_job_t j, unsigned job_is_already_locked)
{
	unsigned ret;

	/* enfore task dependencies */
	if (_starpu_not_all_task_deps_are_fulfilled(j, 0))
		return 0;

	/* enforce data dependencies */
	if (_starpu_submit_job_enforce_data_deps(j))
		return 0;

	ret = _starpu_push_task(j, job_is_already_locked);

	return ret;
}

struct starpu_job_s *_starpu_pop_local_task(struct starpu_worker_s *worker)
{
	struct starpu_job_s *j = NULL;

	PTHREAD_MUTEX_LOCK(&worker->local_jobs_mutex);

	if (!starpu_job_list_empty(worker->local_jobs))
		j = starpu_job_list_pop_back(worker->local_jobs);

	PTHREAD_MUTEX_UNLOCK(&worker->local_jobs_mutex);

	return j;
}

int _starpu_push_local_task(struct starpu_worker_s *worker, struct starpu_job_s *j)
{
	/* Check that the worker is able to execute the task ! */
	STARPU_ASSERT(j->task && j->task->cl);
	if (STARPU_UNLIKELY(!(worker->worker_mask & j->task->cl->where)))
		return -ENODEV;

	PTHREAD_MUTEX_LOCK(&worker->local_jobs_mutex);

	starpu_job_list_push_front(worker->local_jobs, j);

	PTHREAD_MUTEX_UNLOCK(&worker->local_jobs_mutex);

#ifndef STARPU_NON_BLOCKING_DRIVERS
	/* XXX that's a bit excessive ... */
	_starpu_wake_all_blocked_workers_on_node(worker->memory_node);
#endif

	return 0;
}

const char *_starpu_get_model_name(starpu_job_t j)
{
	if (!j)
		return NULL;

	struct starpu_task *task = j->task;
        if (task && task->cl
            && task->cl->model
            && task->cl->model->symbol)
                return task->cl->model->symbol;
#ifdef STARPU_USE_FXT
        else {
                return j->model_name;
        }
#endif
        return NULL;
}
