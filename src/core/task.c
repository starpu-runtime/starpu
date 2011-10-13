/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2010  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
 * Copyright (C) 2011  INRIA
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

#include <starpu.h>
#include <starpu_profiling.h>
#include <starpu_task_bundle.h>
#include <core/workers.h>
#include <core/sched_ctx.h>
#include <core/jobs.h>
#include <core/task.h>
#include <common/config.h>
#include <common/utils.h>
#include <profiling/profiling.h>
#include <profiling/bound.h>

/* XXX this should be reinitialized when StarPU is shutdown (or we should make
 * sure that no task remains !) */
/* TODO we could make this hierarchical to avoid contention ? */
static pthread_cond_t submitted_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t submitted_mutex = PTHREAD_MUTEX_INITIALIZER;
static long int nsubmitted = 0;

static void _starpu_increment_nsubmitted_tasks(void);

/* This key stores the task currently handled by the thread, note that we
 * cannot use the worker structure to store that information because it is
 * possible that we have a task with a NULL codelet, which means its callback
 * could be executed by a user thread as well. */
static pthread_key_t current_task_key;

void starpu_task_init(struct starpu_task *task)
{
	STARPU_ASSERT(task);

	task->cl = NULL;
	task->cl_arg = NULL;
	task->cl_arg_size = 0;

	task->callback_func = NULL;
	task->callback_arg = NULL;

	task->priority = STARPU_DEFAULT_PRIO;
	task->use_tag = 0;
	task->synchronous = 0;

	task->execute_on_a_specific_worker = 0;

	task->bundle = NULL;

	task->detach = 1;

	/* by default, we do not let StarPU free the task structure since
	 * starpu_task_init is likely to be used only for statically allocated
	 * tasks */
	task->destroy = 0;

	task->regenerate = 0;

	task->status = STARPU_TASK_INVALID;

	task->profiling_info = NULL;

	task->predicted = -1.0;

	task->starpu_private = NULL;

	task->sched_ctx = _starpu_get_initial_sched_ctx()->id;
	
	task->control_task = 0;
}

/* Free all the ressources allocated for a task, without deallocating the task
 * structure itself (this is required for statically allocated tasks). */
void starpu_task_deinit(struct starpu_task *task)
{
	STARPU_ASSERT(task);

	/* If a buffer was allocated to store the profiling info, we free it. */
	if (task->profiling_info)
	{
		free(task->profiling_info);
		task->profiling_info = NULL;
	}

	/* If case the task is (still) part of a bundle */
	struct starpu_task_bundle *bundle = task->bundle;
	if (bundle)
	{
		PTHREAD_MUTEX_LOCK(&bundle->mutex);
		int ret = starpu_task_bundle_remove(bundle, task);

		/* Perhaps the bundle was destroyed when removing the last
		 * entry */
		if (ret != 1)
			PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
	}

	starpu_job_t j = (struct starpu_job_s *)task->starpu_private;

	if (j)
		_starpu_job_destroy(j);
}

struct starpu_task * __attribute__((malloc)) starpu_task_create(void)
{
	struct starpu_task *task;

	task = (struct starpu_task *) calloc(1, sizeof(struct starpu_task));
	STARPU_ASSERT(task);

	starpu_task_init(task);

	/* Dynamically allocated tasks are destroyed by default */
	task->destroy = 1;

	return task;
}

/* Free the ressource allocated during starpu_task_create. This function can be
 * called automatically after the execution of a task by setting the "destroy"
 * flag of the starpu_task structure (default behaviour). Calling this function
 * on a statically allocated task results in an undefined behaviour. */
void starpu_task_destroy(struct starpu_task *task)
{
	STARPU_ASSERT(task);

   /* If starpu_task_destroy is called in a callback, we just set the destroy
      flag. The task will be destroyed after the callback returns */
   if (task == starpu_get_current_task()
       && _starpu_get_local_worker_status() == STATUS_CALLBACK) {

      task->destroy = 1;

   } else {

      starpu_task_deinit(task);

      /* TODO handle the case of task with detach = 1 and destroy = 1 */
      /* TODO handle the case of non terminated tasks -> return -EINVAL */
	
      free(task);
   }
}

int starpu_task_wait(struct starpu_task *task)
{
        _STARPU_LOG_IN();
	STARPU_ASSERT(task);

	if (task->detach || task->synchronous) {
		_STARPU_DEBUG("Task is detached or asynchronous. Waiting returns immediately\n");
		_STARPU_LOG_OUT_TAG("einval");
		return -EINVAL;
	}

	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls())) {
		_STARPU_LOG_OUT_TAG("edeadlk");
		return -EDEADLK;
	}

	starpu_job_t j = (struct starpu_job_s *)task->starpu_private;

	_starpu_wait_job(j);

	/* as this is a synchronous task, the liberation of the job
	   structure was deferred */
	if (task->destroy)
		free(task);

        _STARPU_LOG_OUT();
	return 0;
}

starpu_job_t _starpu_get_job_associated_to_task(struct starpu_task *task)
{
	STARPU_ASSERT(task);

	if (!task->starpu_private)
	{
		starpu_job_t j = _starpu_job_create(task);
		task->starpu_private = j;
	}

	return (struct starpu_job_s *)task->starpu_private;
}

/* NB in case we have a regenerable task, it is possible that the job was
 * already counted. */
int _starpu_submit_job(starpu_job_t j, unsigned do_not_increment_nsubmitted)
{
        _STARPU_LOG_IN();
	/* notify bound computation of a new task */
	_starpu_bound_record(j);

	j->terminated = 0;

	if (!do_not_increment_nsubmitted){
		_starpu_increment_nsubmitted_tasks();
		_starpu_increment_nsubmitted_tasks_of_sched_ctx(j->task->sched_ctx);
	}

	PTHREAD_MUTEX_LOCK(&j->sync_mutex);
	
	j->submitted = 1;
       
	int ret = _starpu_enforce_deps_and_schedule(j, 1);

	PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

        _STARPU_LOG_OUT();
        return ret;
}

/* application should submit new tasks to StarPU through this function */
int starpu_task_submit(struct starpu_task *task)
{
	unsigned nsched_ctxs = _starpu_get_nsched_ctxs();

	task->sched_ctx = (nsched_ctxs == 1 || task->control_task) ? 
		0 : starpu_get_sched_ctx();
	int ret;
	unsigned is_sync = task->synchronous;
        _STARPU_LOG_IN();

	if (is_sync)
	{
		/* Perhaps it is not possible to submit a synchronous
		 * (blocking) task */
                if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls())) {
                        _STARPU_LOG_OUT_TAG("EDEADLK");
			return -EDEADLK;
                }

		task->detach = 0;
	}

	STARPU_ASSERT(task);

	if (task->cl)
	{
		uint32_t where = task->cl->where;
		unsigned i;
		if (!_starpu_worker_exists(where)) {
                        _STARPU_LOG_OUT_TAG("ENODEV");
			return -ENODEV;
                }
		assert(task->cl->nbuffers <= STARPU_NMAXBUFS);
		for (i = 0; i < task->cl->nbuffers; i++) {
			/* Make sure handles are not partitioned */
			assert(task->buffers[i].handle->nchildren == 0);
		}

		/* In case we require that a task should be explicitely
		 * executed on a specific worker, we make sure that the worker
		 * is able to execute this task.  */
		if (task->execute_on_a_specific_worker && !starpu_combined_worker_may_execute_task(task->workerid, task, 0)) {
                        _STARPU_LOG_OUT_TAG("ENODEV");
			return -ENODEV;
                }

		_starpu_detect_implicit_data_deps(task);

		if (task->cl->model)
			_starpu_load_perfmodel(task->cl->model);

		if (task->cl->power_model)
			_starpu_load_perfmodel(task->cl->power_model);
	}

	/* If profiling is activated, we allocate a structure to store the
	 * appropriate info. */
	struct starpu_task_profiling_info *info;
	int profiling = starpu_profiling_status_get();
	info = _starpu_allocate_profiling_info_if_needed(task);
	task->profiling_info = info;

	/* The task is considered as block until we are sure there remains not
	 * dependency. */
	task->status = STARPU_TASK_BLOCKED;


	if (profiling)
		starpu_clock_gettime(&info->submit_time);

	/* internally, StarPU manipulates a starpu_job_t which is a wrapper around a
	* task structure, it is possible that this job structure was already
	* allocated, for instance to enforce task depenencies. */
	starpu_job_t j = _starpu_get_job_associated_to_task(task);

	ret = _starpu_submit_job(j, 0);

	if (is_sync)
		_starpu_wait_job(j);

        _STARPU_LOG_OUT();
	return ret;
}

int _starpu_task_submit_internal(struct starpu_task *task)
{
	task->control_task = 1;
	return starpu_task_submit(task);
}

void starpu_display_codelet_stats(struct starpu_codelet_t *cl)
{
	unsigned worker;
	unsigned nworkers = starpu_worker_get_count();

	if (cl->model && cl->model->symbol)
		fprintf(stderr, "Statistics for codelet %s\n", cl->model->symbol);

	unsigned long total = 0;
	
	for (worker = 0; worker < nworkers; worker++)
		total += cl->per_worker_stats[worker];

	for (worker = 0; worker < nworkers; worker++)
	{
		char name[32];
		starpu_worker_get_name(worker, name, 32);

		fprintf(stderr, "\t%s -> %lu / %lu (%2.2f %%)\n", name, cl->per_worker_stats[worker], total, (100.0f*cl->per_worker_stats[worker])/total);
	}
}

/*
 * We wait for all the tasks that have already been submitted. Note that a
 * regenerable is not considered finished until it was explicitely set as
 * non-regenerale anymore (eg. from a callback).
 */
int starpu_task_wait_for_all(void)
{
	unsigned nsched_ctxs = _starpu_get_nsched_ctxs();
	unsigned sched_ctx = nsched_ctxs == 1 ? 0 : starpu_get_sched_ctx();
	starpu_wait_for_all_tasks_of_sched_ctx(sched_ctx);

	/* if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls())) */
	/* 	return -EDEADLK; */

	/* PTHREAD_MUTEX_LOCK(&submitted_mutex); */

	/* STARPU_TRACE_TASK_WAIT_FOR_ALL; */

	/* while (nsubmitted > 0) */
	/* 	PTHREAD_COND_WAIT(&submitted_cond, &submitted_mutex); */
	
	/* PTHREAD_MUTEX_UNLOCK(&submitted_mutex); */
	return 0;
}

void _starpu_decrement_nsubmitted_tasks(void)
{
	PTHREAD_MUTEX_LOCK(&submitted_mutex);

	if (--nsubmitted == 0)
		PTHREAD_COND_BROADCAST(&submitted_cond);

	STARPU_TRACE_UPDATE_TASK_CNT(nsubmitted);

	PTHREAD_MUTEX_UNLOCK(&submitted_mutex);

}

static void _starpu_increment_nsubmitted_tasks(void)
{
	PTHREAD_MUTEX_LOCK(&submitted_mutex);

	nsubmitted++;

	STARPU_TRACE_UPDATE_TASK_CNT(nsubmitted);

	PTHREAD_MUTEX_UNLOCK(&submitted_mutex);
}

void _starpu_initialize_current_task_key(void)
{
	pthread_key_create(&current_task_key, NULL);
}

/* Return the task currently executed by the worker, or NULL if this is called
 * either from a thread that is not a task or simply because there is no task
 * being executed at the moment. */
struct starpu_task *starpu_get_current_task(void)
{
	return (struct starpu_task *) pthread_getspecific(current_task_key);
}

void _starpu_set_current_task(struct starpu_task *task)
{
	pthread_setspecific(current_task_key, task);
}
