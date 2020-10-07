/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Thibaut Lambert
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
#include <core/jobs.h>
#include <core/task.h>
#include <core/workers.h>
#include <core/dependencies/data_concurrency.h>
#include <common/config.h>
#include <common/utils.h>
#include <common/graph.h>
#include <profiling/profiling.h>
#include <profiling/bound.h>
#include <core/debug.h>
#include <limits.h>

static int max_memory_use;
static unsigned long njobs, maxnjobs;

#ifdef STARPU_DEBUG
/* List of all jobs, for debugging */
static struct _starpu_job_multilist_all_submitted all_jobs_list;
static starpu_pthread_mutex_t all_jobs_list_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
#endif

void _starpu_job_init(void)
{
	max_memory_use = starpu_get_env_number_default("STARPU_MAX_MEMORY_USE", 0);
#ifdef STARPU_DEBUG
	_starpu_job_multilist_head_init_all_submitted(&all_jobs_list);
#endif
}

void _starpu_job_fini(void)
{
	if (max_memory_use)
	{
		_STARPU_DISP("Memory used for %lu tasks: %lu MiB\n", maxnjobs, (unsigned long) (maxnjobs * (sizeof(struct starpu_task) + sizeof(struct _starpu_job))) >> 20);
		STARPU_ASSERT_MSG(njobs == 0, "Some tasks have not been cleaned, did you forget to call starpu_task_destroy or starpu_task_clean?");
	}
}

void _starpu_exclude_task_from_dag(struct starpu_task *task)
{
	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	j->exclude_from_dag = 1;
	_STARPU_TRACE_TASK_EXCLUDE_FROM_DAG(j);
}

/* create an internal struct _starpu_job structure to encapsulate the task */
struct _starpu_job* STARPU_ATTRIBUTE_MALLOC _starpu_job_create(struct starpu_task *task)
{
	struct _starpu_job *job;
        _STARPU_LOG_IN();

	_STARPU_MALLOC(job, sizeof(*job));

	/* As most of the fields must be initialized at NULL, let's put 0
	 * everywhere */
	memset(job, 0, sizeof(*job));

	if (task->dyn_handles)
	{
		_STARPU_MALLOC(job->dyn_ordered_buffers, STARPU_TASK_GET_NBUFFERS(task) * sizeof(job->dyn_ordered_buffers[0]));
		_STARPU_CALLOC(job->dyn_dep_slots, STARPU_TASK_GET_NBUFFERS(task), sizeof(job->dyn_dep_slots[0]));
	}

	job->task = task;

#if !defined(STARPU_USE_FXT) && !defined(STARPU_DEBUG)
	if (_starpu_bound_recording || _starpu_task_break_on_push != -1 || _starpu_task_break_on_sched != -1 || _starpu_task_break_on_pop != -1 || _starpu_task_break_on_exec != -1 || STARPU_AYU_EVENT)
#endif
	{
		job->job_id = _starpu_fxt_get_job_id();
		STARPU_AYU_ADDTASK(job->job_id, task);
		STARPU_ASSERT(job->job_id != ULONG_MAX);
	}
	if (max_memory_use)
	{
		unsigned long jobs = STARPU_ATOMIC_ADDL(&njobs, 1);
		if (jobs > maxnjobs)
			maxnjobs = jobs;
	}

	_starpu_cg_list_init(&job->job_successors);

	STARPU_PTHREAD_MUTEX_INIT(&job->sync_mutex, NULL);
	STARPU_PTHREAD_COND_INIT(&job->sync_cond, NULL);

	/* By default we have sequential tasks */
	job->task_size = 1;

	if (task->use_tag)
		_starpu_tag_declare(task->tag_id, job);

	if (_starpu_graph_record)
		_starpu_graph_add_job(job);

        _STARPU_LOG_OUT();
	return job;
}

void _starpu_job_destroy(struct _starpu_job *j)
{
	/* Wait for any code that was still working on the job (and was
	 * probably our waker) */
	STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
	STARPU_PTHREAD_COND_DESTROY(&j->sync_cond);
	STARPU_PTHREAD_MUTEX_DESTROY(&j->sync_mutex);

	if (j->task_size > 1)
	{
		STARPU_PTHREAD_BARRIER_DESTROY(&j->before_work_barrier);
		STARPU_PTHREAD_BARRIER_DESTROY(&j->after_work_barrier);
		STARPU_ASSERT(j->after_work_busy_barrier == 0);
	}

	_starpu_cg_list_deinit(&j->job_successors);
	if (j->dyn_ordered_buffers)
	{
		free(j->dyn_ordered_buffers);
		j->dyn_ordered_buffers = NULL;
	}
	if (j->dyn_dep_slots)
	{
		free(j->dyn_dep_slots);
		j->dyn_dep_slots = NULL;
	}

	if (_starpu_graph_record && j->graph_node)
		_starpu_graph_drop_job(j);

	if (max_memory_use)
		(void) STARPU_ATOMIC_ADDL(&njobs, -1);

	free(j);
}

int _starpu_job_finished(struct _starpu_job *j)
{
	int ret;
	STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
	ret = j->terminated == 2;
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
	return ret;
}

void _starpu_wait_job(struct _starpu_job *j)
{
	STARPU_ASSERT(j->task);
	STARPU_ASSERT(!j->task->detach);
        _STARPU_LOG_IN();

	STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);

	/* We wait for the flag to have a value of 2 which means that both the
	 * codelet's implementation and its callback have been executed. That
	 * way, _starpu_wait_job won't return until the entire task was really
	 * executed (so that we cannot destroy the task while it is still being
	 * manipulated by the driver). */

	while (j->terminated != 2)
	{
		STARPU_PTHREAD_COND_WAIT(&j->sync_cond, &j->sync_mutex);
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
        _STARPU_LOG_OUT();
}

#ifdef STARPU_OPENMP
int _starpu_test_job_termination(struct _starpu_job *j)
{
	STARPU_ASSERT(j->task);
	STARPU_ASSERT(!j->task->detach);
	/* Disable Helgrind race complaint, since we really just want to poll j->terminated */
	if (STARPU_RUNNING_ON_VALGRIND)
	{
		int v = STARPU_PTHREAD_MUTEX_TRYLOCK(&j->sync_mutex);
		if (v != EBUSY)
		{
			STARPU_ASSERT(v == 0);
			int ret = (j->terminated == 2);
			STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
			return ret;
		}
		else
		{
			return 0;
		}
	}
	else
	{
		STARPU_SYNCHRONIZE();
		return j->terminated == 2;
	}
}
void _starpu_job_prepare_for_continuation_ext(struct _starpu_job *j, unsigned continuation_resubmit,
		void (*continuation_callback_on_sleep)(void *arg), void *continuation_callback_on_sleep_arg)
{
	STARPU_ASSERT(!j->continuation);
	/* continuation are not supported for parallel tasks for now */
	STARPU_ASSERT(j->task_size == 1);
	j->continuation = 1;
	j->continuation_resubmit = continuation_resubmit;
	j->continuation_callback_on_sleep = continuation_callback_on_sleep;
	j->continuation_callback_on_sleep_arg = continuation_callback_on_sleep_arg;
	j->job_successors.ndeps = 0;
}
/* Prepare a currently running job for accepting a new set of
 * dependencies in anticipation of becoming a continuation. */
void _starpu_job_prepare_for_continuation(struct _starpu_job *j)
{
	_starpu_job_prepare_for_continuation_ext(j, 1, NULL, NULL);
}
void _starpu_job_set_omp_cleanup_callback(struct _starpu_job *j,
		void (*omp_cleanup_callback)(void *arg), void *omp_cleanup_callback_arg)
{
	j->omp_cleanup_callback = omp_cleanup_callback;
	j->omp_cleanup_callback_arg = omp_cleanup_callback_arg;
}
#endif

void _starpu_handle_job_submission(struct _starpu_job *j)
{
	/* Need to atomically set submitted to 1 and check dependencies, since
	 * this is concucrent with _starpu_notify_cg */
	j->terminated = 0;

	if (!j->submitted)
		j->submitted = 1;
	else
		j->submitted = 2;

#ifdef STARPU_DEBUG
	STARPU_PTHREAD_MUTEX_LOCK(&all_jobs_list_mutex);
	_starpu_job_multilist_push_back_all_submitted(&all_jobs_list, j);
	STARPU_PTHREAD_MUTEX_UNLOCK(&all_jobs_list_mutex);
#endif
}

void starpu_task_end_dep_release(struct starpu_task *t)
{
	struct _starpu_job *j = _starpu_get_job_associated_to_task(t);
	_starpu_handle_job_termination(j);
}

void starpu_task_end_dep_add(struct starpu_task *t, int nb_deps)
{
	struct _starpu_job *j = _starpu_get_job_associated_to_task(t);
	STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
	t->nb_termination_call_required += nb_deps;
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
}

void _starpu_handle_job_termination(struct _starpu_job *j)
{
	if (j->task->nb_termination_call_required != 0)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
		int nb = j->task->nb_termination_call_required;
		j->task->nb_termination_call_required -= 1;
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		if (nb != 0) return;
	}

	struct starpu_task *task = j->task;
	struct starpu_task *end_rdep = NULL;
	unsigned sched_ctx = task->sched_ctx;
	double flops = task->flops;
	const unsigned continuation =
#ifdef STARPU_OPENMP
		j->continuation
#else
		0
#endif
		;
#ifdef STARPU_DEBUG
	STARPU_PTHREAD_MUTEX_LOCK(&all_jobs_list_mutex);
	_starpu_job_multilist_erase_all_submitted(&all_jobs_list, j);
	STARPU_PTHREAD_MUTEX_UNLOCK(&all_jobs_list_mutex);
#endif

	STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
	STARPU_ASSERT(task->status == STARPU_TASK_RUNNING);
#ifdef STARPU_OPENMP
	if (continuation)
	{
		task->status = STARPU_TASK_STOPPED;
	}
	else
#endif
	{
		task->status = STARPU_TASK_FINISHED;

		/* We must have set the j->terminated flag early, so that it is
		 * possible to express task dependencies within the callback
		 * function. A value of 1 means that the codelet was executed but that
		 * the callback is not done yet. */
		j->terminated = 1;
		end_rdep = j->end_rdep;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);


#ifdef STARPU_USE_SC_HYPERVISOR
	size_t data_size = 0;
#endif //STARPU_USE_SC_HYPERVISOR

	/* We release handle reference count */
	if (task->cl && !continuation)
	{
		unsigned i;
		unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
#ifdef STARPU_USE_SC_HYPERVISOR
		for(i = 0; i < nbuffers; i++)
		{
			starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, i);
			if (handle != NULL)
				data_size += _starpu_data_get_size(handle);
		}
#endif //STARPU_USE_SC_HYPERVISOR

		for (i = 0; i < nbuffers; i++)
		{
			starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, i);
			_starpu_spin_lock(&handle->header_lock);
			handle->busy_count--;
			if (!_starpu_data_check_not_busy(handle))
				_starpu_spin_unlock(&handle->header_lock);
		}
	}
	/* Check nowhere before releasing the sequential consistency (which may
	 * unregister the handle and free its switch_cl, and thus task->cl here.  */
	unsigned nowhere = !task->cl || task->cl->where == STARPU_NOWHERE || task->where == STARPU_NOWHERE;
	/* If this is a continuation, we do not release task dependencies now.
	 * Task dependencies will be released only when the continued task
	 * fully completes */
	if (!continuation)
	{
		/* Tell other tasks that we don't exist any more, thus no need for
		 * implicit dependencies any more.  */
		_starpu_release_task_enforce_sequential_consistency(j);
	}

	/* If the job was executed on a combined worker there is no need for the
	 * scheduler to process it : the task structure doesn't contain any valuable
	 * data as it's not linked to an actual worker */
	/* control task should not execute post_exec_hook */
	if(j->task_size == 1 && !nowhere && !j->internal
#ifdef STARPU_OPENMP
	/* If this is a continuation, we do not execute the post_exec_hook. The
	 * post_exec_hook will be run only when the continued task fully
	 * completes.
	 *
	 * Note: If needed, a specific hook could be added to handle stopped
	 * tasks */
	&& !continuation
#endif
			)
	{
		_starpu_sched_post_exec_hook(task);
#ifdef STARPU_USE_SC_HYPERVISOR
		int workerid = starpu_worker_get_id();
		_starpu_sched_ctx_post_exec_task_cb(workerid, task, data_size, j->footprint);
#endif //STARPU_USE_SC_HYPERVISOR

	}

	/* Remove ourself from the graph before notifying dependencies */
	if (_starpu_graph_record)
		_starpu_graph_drop_job(j);

	/* Task does not have a cl, but has explicit data dependencies, we need
	 * to tell them that we will not exist any more before notifying the
	 * tasks waiting for us
	 *
	 * For continuations, implicit dependency handles are only released
	 * when the task fully completes */
	if (j->implicit_dep_handle && !continuation)
	{
		starpu_data_handle_t handle = j->implicit_dep_handle;
		_starpu_release_data_enforce_sequential_consistency(j->task, &j->implicit_dep_slot, handle);
		/* Release reference taken while setting implicit_dep_handle */
		_starpu_spin_lock(&handle->header_lock);
		handle->busy_count--;
		if (!_starpu_data_check_not_busy(handle))
			_starpu_spin_unlock(&handle->header_lock);
	}

	_STARPU_TRACE_TASK_NAME(j);

	/* If this is a continuation, we do not notify task/tag dependencies
	 * now. Task/tag dependencies will be notified only when the continued
	 * task fully completes */
	if (!continuation)
	{
		/* in case there are dependencies, wake up the proper tasks */
		if (end_rdep)
			starpu_task_end_dep_release(end_rdep);
		_starpu_notify_dependencies(j);
	}

	/* If this is a continuation, we do not execute the callback
	 * now. The callback will be executed only when the continued
	 * task fully completes */
	if (!continuation)
	{
		/* the callback is executed after the dependencies so that we may remove the tag
		 * of the task itself */
		if (task->callback_func)
		{
			int profiling = starpu_profiling_status_get();
			if (profiling && task->profiling_info)
				_starpu_clock_gettime(&task->profiling_info->callback_start_time);

			/* so that we can check whether we are doing blocking calls
			 * within the callback */
			_starpu_set_local_worker_status(STATUS_CALLBACK);


			/* Perhaps we have nested callbacks (eg. with chains of empty
			 * tasks). So we store the current task and we will restore it
			 * later. */
			struct starpu_task *current_task = starpu_task_get_current();

			_starpu_set_current_task(task);

			_STARPU_TRACE_START_CALLBACK(j);
			task->callback_func(task->callback_arg);
			_STARPU_TRACE_END_CALLBACK(j);

			_starpu_set_current_task(current_task);

			_starpu_set_local_worker_status(STATUS_UNKNOWN);

			if (profiling && task->profiling_info)
				_starpu_clock_gettime(&task->profiling_info->callback_end_time);
		}
	}

	/* Note: For now, we keep the TASK_DONE trace event for continuation,
	 * however we could add a specific event for stopped tasks if needed.
	 */
	_STARPU_TRACE_TASK_DONE(j);

	/* NB: we do not save those values before the callback, in case the
	 * application changes some parameters eventually (eg. a task may not
	 * be generated if the application is terminated). */
	unsigned destroy = task->destroy;
	unsigned detach = task->detach;
	unsigned regenerate = task->regenerate;
	unsigned synchronous = task->synchronous;

	/* we do not desallocate the job structure if some is going to
	 * wait after the task */
	STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
	if (!continuation)
	{
#ifdef STARPU_OPENMP
		if (j->omp_cleanup_callback)
		{
			j->omp_cleanup_callback(j->omp_cleanup_callback_arg);
			j->omp_cleanup_callback = NULL;
			j->omp_cleanup_callback_arg = NULL;
		}
#endif
		/* A value of 2 is put to specify that not only the codelet but
		 * also the callback were executed. */
		j->terminated = 2;
	}
	STARPU_PTHREAD_COND_BROADCAST(&j->sync_cond);
	STARPU_AYU_REMOVETASK(j->job_id);
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

	if (detach && !continuation)
	{
		/* no one is going to synchronize with that task so we release
		 * the data structures now. In case the job was already locked
		 * by the caller, it is its responsability to destroy the task.
		 * */
		if (destroy)
			_starpu_task_destroy(task);
	}

	/* A continuation is not much different from a regenerated task. */
	if (regenerate || continuation)
	{
		STARPU_ASSERT_MSG((detach && !destroy && !synchronous)
				|| continuation
				, "Regenerated task must be detached (was %u), and not have detroy=1 (was %u) or synchronous=1 (was %u)", detach, destroy, synchronous);
		STARPU_AYU_ADDTASK(j->job_id, j->exclude_from_dag?NULL:task);

		{
#ifdef STARPU_OPENMP
			unsigned continuation_resubmit = j->continuation_resubmit;
			void (*continuation_callback_on_sleep)(void *arg) = j->continuation_callback_on_sleep;
			void *continuation_callback_on_sleep_arg = j->continuation_callback_on_sleep_arg;
			j->continuation_resubmit = 1;
			j->continuation_callback_on_sleep = NULL;
			j->continuation_callback_on_sleep_arg = NULL;
			if (!continuation || continuation_resubmit)
#endif
			{
				/* We reuse the same job structure */
				task->status = STARPU_TASK_BLOCKED;
				int ret = _starpu_submit_job(j);
				STARPU_ASSERT(!ret);
			}
#ifdef STARPU_OPENMP
			if (continuation && continuation_callback_on_sleep != NULL)
			{
				continuation_callback_on_sleep(continuation_callback_on_sleep_arg);
			}
#endif
		}
	}

	_starpu_decrement_nready_tasks_of_sched_ctx(sched_ctx, flops);
	_starpu_decrement_nsubmitted_tasks_of_sched_ctx(sched_ctx);
	struct _starpu_worker *worker;
	worker = _starpu_get_local_worker_key();
	if (worker)
	{
		STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);

		if(worker->removed_from_ctx[sched_ctx] == 1 && worker->shares_tasks_lists[sched_ctx] == 1)
		{
			_starpu_worker_gets_out_of_ctx(sched_ctx, worker);
			worker->removed_from_ctx[sched_ctx] = 0;
		}
		STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
	}
}

/* This function is called when a new task is submitted to StarPU
 * it returns 1 if the tag deps are not fulfilled, 0 otherwise */
static unsigned _starpu_not_all_tag_deps_are_fulfilled(struct _starpu_job *j)
{
	unsigned ret;

	if (!j->task->use_tag)
	{
		/* this task does not use tags, so we can go on */
		return 0;
	}

	struct _starpu_tag *tag = j->tag;

	struct _starpu_cg_list *tag_successors = &tag->tag_successors;

	_starpu_spin_lock(&tag->lock);
	STARPU_ASSERT_MSG(tag->is_assigned == 1 || !tag_successors->ndeps, "a tag can be assigned only one task to wake (%llu had %u assigned tasks, and %u successors)", (unsigned long long) tag->id, tag->is_assigned, tag_successors->ndeps);

	if (tag_successors->ndeps != tag_successors->ndeps_completed)
	{
		tag->state = STARPU_BLOCKED;
                j->task->status = STARPU_TASK_BLOCKED_ON_TAG;
		ret = 1;
	}
	else
	{
		/* existing deps (if any) are fulfilled */
		/* If the same tag is being signaled by several tasks, do not
		 * clear a DONE state. If it's the same job submitted several
		 * times with the same tag, we have to do it */
		if (j->submitted == 2 || tag->state != STARPU_DONE)
			tag->state = STARPU_READY;
		/* already prepare for next run */
		tag_successors->ndeps_completed = 0;
		ret = 0;
	}

	_starpu_spin_unlock(&tag->lock);
	return ret;
}

static unsigned _starpu_not_all_task_deps_are_fulfilled(struct _starpu_job *j)
{
	unsigned ret;

	struct _starpu_cg_list *job_successors = &j->job_successors;

	if (!j->submitted || (job_successors->ndeps != job_successors->ndeps_completed))
	{
		STARPU_ASSERT(j->task->status == STARPU_TASK_BLOCKED || j->task->status == STARPU_TASK_BLOCKED_ON_TAG);
                j->task->status = STARPU_TASK_BLOCKED_ON_TASK;
		ret = 1;
	}
	else
	{
		/* existing deps (if any) are fulfilled */
		/* already prepare for next run */
		job_successors->ndeps_completed = 0;
		ret = 0;
	}

	return ret;
}

/*
 *	In order, we enforce tag, task and data dependencies. The task is
 *	passed to the scheduler only once all these constraints are fulfilled.
 *
 *	The job mutex has to be taken for atomicity with task submission, and
 *	is released here.
 */
unsigned _starpu_enforce_deps_and_schedule(struct _starpu_job *j)
{
	unsigned ret;
        _STARPU_LOG_IN();

	/* enfore tag dependencies */
	if (_starpu_not_all_tag_deps_are_fulfilled(j))
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		_STARPU_LOG_OUT_TAG("not_all_tag_deps_are_fulfilled");
		return 0;
	}

	/* enfore task dependencies */
	if (_starpu_not_all_task_deps_are_fulfilled(j))
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		_STARPU_LOG_OUT_TAG("not_all_task_deps_are_fulfilled");
		return 0;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

	/* enforce data dependencies */
	if (_starpu_submit_job_enforce_data_deps(j))
	{
		_STARPU_LOG_OUT_TAG("enforce_data_deps");
		return 0;
	}

	ret = _starpu_push_task(j);

	_STARPU_LOG_OUT();
	return ret;
}

/* Tag deps are already fulfilled */
unsigned _starpu_enforce_deps_starting_from_task(struct _starpu_job *j)
{
	unsigned ret;

	/* enfore task dependencies */
	if (_starpu_not_all_task_deps_are_fulfilled(j))
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		return 0;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

	/* enforce data dependencies */
	if (_starpu_submit_job_enforce_data_deps(j))
		return 0;

	ret = _starpu_push_task(j);

	return ret;
}

#ifdef STARPU_OPENMP
/* When waking up a continuation, we only enforce new task dependencies */
unsigned _starpu_reenforce_task_deps_and_schedule(struct _starpu_job *j)
{
	unsigned ret;
        _STARPU_LOG_IN();
	STARPU_ASSERT(j->discontinuous);

	/* enfore task dependencies */
	if (_starpu_not_all_task_deps_are_fulfilled(j))
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		_STARPU_LOG_OUT_TAG("not_all_task_deps_are_fulfilled");
		return 0;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
	ret = _starpu_push_task(j);

	_STARPU_LOG_OUT();
	return ret;
}
#endif

/* This is called when a tag or task dependency is to be released.  */
void _starpu_enforce_deps_notify_job_ready_soon(struct _starpu_job *j, _starpu_notify_job_start_data *data, int tag)
{
	if (!j->submitted)
		/* It's not even submitted actually */
		return;
	struct _starpu_cg_list *job_successors = &j->job_successors;
        /* tag is 1 when we got woken up by a tag dependency about to be
         * released, and thus we have to check the exact numbner of
         * dependencies.  Otherwise it's a task dependency which is about to be
         * released.  */
	if (job_successors->ndeps != job_successors->ndeps_completed + 1 - tag)
		/* There are still other dependencies */
		return;

	_starpu_enforce_data_deps_notify_job_ready_soon(j, data);
}

/* Ordered tasks are simply recorded as they arrive in the local_ordered_tasks
 * ring buffer, indexed by order, and pulled from its head. */
/* TODO: replace with perhaps a heap */

/* This function must be called with worker->sched_mutex taken */
struct starpu_task *_starpu_pop_local_task(struct _starpu_worker *worker)
{
	struct starpu_task *task = NULL;

	if (worker->local_ordered_tasks_size)
	{
		task = worker->local_ordered_tasks[worker->current_ordered_task];
		if (task)
		{
			worker->local_ordered_tasks[worker->current_ordered_task] = NULL;
			STARPU_ASSERT(task->workerorder == worker->current_ordered_task_order);
			/* Next ordered task is there, return it */
			worker->current_ordered_task = (worker->current_ordered_task + 1) % worker->local_ordered_tasks_size;
			worker->current_ordered_task_order++;
			_starpu_pop_task_end(task);
			return task;
		}
	}

	if (!starpu_task_list_empty(&worker->local_tasks))
		task = starpu_task_list_pop_front(&worker->local_tasks);

	_starpu_pop_task_end(task);
	return task;
}

int _starpu_push_local_task(struct _starpu_worker *worker, struct starpu_task *task, int prio)
{
	/* Check that the worker is able to execute the task ! */
	STARPU_ASSERT(task && task->cl);
	if (STARPU_UNLIKELY(!(worker->worker_mask & task->where)))
		return -ENODEV;

	starpu_worker_lock(worker->workerid);

	if (task->execute_on_a_specific_worker && task->workerorder)
	{
		STARPU_ASSERT_MSG(task->workerorder >= worker->current_ordered_task_order, "worker order values must not have duplicates (%u pushed to worker %d, but %u already passed)", task->workerorder, worker->workerid, worker->current_ordered_task_order);
		/* Put it in the ordered task ring */
		unsigned needed = task->workerorder - worker->current_ordered_task_order + 1;
		if (worker->local_ordered_tasks_size < needed)
		{
			/* Increase the size */
			unsigned alloc = worker->local_ordered_tasks_size;
			struct starpu_task **new;

			if (!alloc)
				alloc = 1;
			while (alloc < needed)
				alloc *= 2;
			_STARPU_MALLOC(new, alloc * sizeof(*new));

			if (worker->local_ordered_tasks_size)
			{
				/* Put existing tasks at the beginning of the new ring */
				unsigned copied = worker->local_ordered_tasks_size - worker->current_ordered_task;
				memcpy(new, &worker->local_ordered_tasks[worker->current_ordered_task], copied * sizeof(*new));
				memcpy(new + copied, worker->local_ordered_tasks, (worker->local_ordered_tasks_size - copied) * sizeof(*new));
			}
			memset(new + worker->local_ordered_tasks_size, 0, (alloc - worker->local_ordered_tasks_size) * sizeof(*new));
			free(worker->local_ordered_tasks);
			worker->local_ordered_tasks = new;
			worker->local_ordered_tasks_size = alloc;
			worker->current_ordered_task = 0;
		}
		worker->local_ordered_tasks[(worker->current_ordered_task + task->workerorder - worker->current_ordered_task_order) % worker->local_ordered_tasks_size] = task;
	}
	else
	{
#ifdef STARPU_DEVEL
#warning FIXME use a prio_list
#endif
		if (prio)
			starpu_task_list_push_front(&worker->local_tasks, task);
		else
			starpu_task_list_push_back(&worker->local_tasks, task);
	}

	starpu_wake_worker_locked(worker->workerid);
	starpu_push_task_end(task);
	starpu_worker_unlock(worker->workerid);

	return 0;
}
