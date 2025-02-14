/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013-2013  Thibaut Lambert
 * Copyright (C) 2011-2011  Télécom Sud Paris
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
#include <datawizard/memory_nodes.h>
#include <profiling/profiling.h>
#include <profiling/bound.h>
#include <core/debug.h>
#include <limits.h>
#include <core/workers.h>

static int max_memory_use;
static int task_progress;
static unsigned long njobs_finished;
static unsigned long njobs, maxnjobs;

#ifdef STARPU_DEBUG
/* List of all jobs, for debugging */
static struct _starpu_job_multilist_all_submitted all_jobs_list;
static starpu_pthread_mutex_t all_jobs_list_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
#endif

void _starpu_job_crash();

void _starpu_job_init(void)
{
	max_memory_use = starpu_getenv_number_default("STARPU_MAX_MEMORY_USE", 0);
	task_progress = starpu_getenv_number_default("STARPU_TASK_PROGRESS", 0);
#ifdef STARPU_DEBUG
	_starpu_job_multilist_head_init_all_submitted(&all_jobs_list);
#endif
	_starpu_crash_add_hook(&_starpu_job_crash);
}

void _starpu_job_memory_use(int check)
{
	if (max_memory_use)
	{
		_STARPU_DISP("Memory used for %lu tasks: %lu MiB\n", maxnjobs, (unsigned long) (maxnjobs * (sizeof(struct starpu_task) + sizeof(struct _starpu_job))) >> 20);
		if (check)
			STARPU_ASSERT_MSG(njobs == 0, "Some tasks have not been cleaned, did you forget to call starpu_task_destroy or starpu_task_clean?");
	}
}

void _starpu_job_crash()
{
	_starpu_job_memory_use(0);
}

void _starpu_job_fini(void)
{
	_starpu_job_memory_use(1);
}

void _starpu_exclude_task_from_dag(struct starpu_task *task)
{
	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	j->exclude_from_dag = 1;
	_starpu_trace_task_exclude_from_dag(j);
}

/* create an internal struct _starpu_job structure to encapsulate the task */
struct _starpu_job* STARPU_ATTRIBUTE_MALLOC _starpu_job_create(struct starpu_task *task)
{
	struct _starpu_job *job;
	_STARPU_LOG_IN();

	/* As most of the fields must be initialized at NULL, let's put 0
	 * everywhere */
	_STARPU_CALLOC(job, 1, sizeof(*job));

	if (task->dyn_handles)
	{
		_STARPU_MALLOC(job->dyn_ordered_buffers, STARPU_TASK_GET_NBUFFERS(task) * sizeof(job->dyn_ordered_buffers[0]));
		_STARPU_CALLOC(job->dyn_dep_slots, STARPU_TASK_GET_NBUFFERS(task), sizeof(job->dyn_dep_slots[0]));
	}

	job->task = task;

	if (
#if defined(STARPU_DEBUG) || defined(STARPU_PROF_TASKSTUBS)
	    1
#elif defined(STARPU_USE_FXT)
	    fut_active
#else
	    _starpu_bound_recording || _starpu_task_break_on_push != -1 || _starpu_task_break_on_sched != -1 || _starpu_task_break_on_pop != -1 || _starpu_task_break_on_exec != -1 || STARPU_AYU_EVENT
#endif
	   )
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

	_starpu_cg_list_init0(&job->job_successors);

	STARPU_PTHREAD_MUTEX_INIT0(&job->sync_mutex, NULL);
	STARPU_PTHREAD_COND_INIT0(&job->sync_cond, NULL);

	/* By default we have sequential tasks */
	job->task_size = 1;

	job->workerid = -1;

	if (task->use_tag)
		_starpu_tag_declare(task->tag_id, job);

	if (_starpu_graph_record)
		_starpu_graph_add_job(job);

	_STARPU_LOG_OUT();
	return job;
}

struct _starpu_job* _starpu_get_job_associated_to_task_slow(struct starpu_task *task, struct _starpu_job *job)
{
	if (job == _STARPU_JOB_UNSET)
	{
		job = STARPU_VAL_COMPARE_AND_SWAP_PTR(&task->starpu_private, _STARPU_JOB_UNSET, _STARPU_JOB_SETTING);
		if (job != _STARPU_JOB_UNSET && job != _STARPU_JOB_SETTING)
		{
			/* Actually available in the meanwhile */
			STARPU_RMB();
			return job;
		}

		if (job == _STARPU_JOB_UNSET)
		{
			/* Ok, we have to do it */
			job = _starpu_job_create(task);
			STARPU_WMB();
			task->starpu_private = job;
			return job;
		}
	}

	/* Saw _STARPU_JOB_SETTING, somebody is doing it, wait for it.
	 * This is rare enough that busy-reading is fine enough. */
	while ((job = *(struct _starpu_job *volatile*) &task->starpu_private) == _STARPU_JOB_SETTING)
	{
		STARPU_UYIELD();
		STARPU_SYNCHRONIZE();
	}

	STARPU_RMB();
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
	j->job_successors.ndeps_completed = 0;
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

#ifdef STARPU_USE_FXT
	struct starpu_task *current = starpu_task_get_current();
	if (current)
	{
		struct _starpu_job *jcurrent = _starpu_get_job_associated_to_task(current);
		_starpu_trace_task_end_dep(jcurrent, j);
	}
#endif

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

	if (task_progress)
	{
		unsigned long jobs = STARPU_ATOMIC_ADDL(&njobs_finished, 1);

		fprintf(stderr,"\r%lu tasks finished (last %lu %p on %d)...", jobs, j->job_id, j->task, starpu_worker_get_id());
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

	if (!continuation)
	{
		void (*epilogue_callback)(void *) = task->epilogue_callback_func;
		/* the epilogue callback is executed before the dependencies release*/
		if (epilogue_callback)
		{
			enum _starpu_worker_status old_status = _starpu_get_local_worker_status();

			/* so that we can check whether we are doing blocking calls
			 * within the callback */
			if (!(old_status & STATUS_CALLBACK))
				_starpu_add_local_worker_status(STATUS_INDEX_CALLBACK, NULL);

			/* Perhaps we have nested callbacks (eg. with chains of empty
			 * tasks). So we store the current task and we will restore it
			 * later. */
			struct starpu_task *current_task = starpu_task_get_current();

			_starpu_set_current_task(task);

			_starpu_trace_start_callback(j);
			epilogue_callback(task->epilogue_callback_arg);
			_starpu_trace_end_callback(j);

			_starpu_set_current_task(current_task);

			if (!(old_status & STATUS_CALLBACK))
				_starpu_clear_local_worker_status(STATUS_INDEX_CALLBACK, NULL);
		}
	}

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

		/* already prepare for next run */
		struct _starpu_cg_list *job_successors = &j->job_successors;
		job_successors->ndeps_completed = 0;

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
	if (task->cl && !continuation
#ifdef STARPU_RECURSIVE_TASKS
	    && !j->is_recursive_task
#endif
	    )
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

	/* Get callback pointer for codelet before notifying dependencies, in
	   case dependencies free the codelet (see starpu_data_unregister for
	   instance) */
	void (*callback)(void *) = task->callback_func;
	if (!callback && task->cl)
		callback = task->cl->callback_func;

	/* If this is a continuation, we do not release task dependencies now.
	 * Task dependencies will be released only when the continued task
	 * fully completes */
	if (!continuation)
	{
		/* Tell other tasks that we don't exist any more, thus no need for
		 * implicit dependencies any more.  */
		_starpu_release_task_enforce_sequential_consistency(j);
	}

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

	if (!continuation)
	{
		/* If this is a continuation, we do not notify task/tag dependencies
		 * now. Task/tag dependencies will be notified only when the continued
		 * task fully completes */
		/* in case there are dependencies, wake up the proper tasks */
		if (end_rdep)
			starpu_task_end_dep_release(end_rdep);
		_starpu_notify_dependencies(j);

		/* If this is a continuation, we do not execute the callback
		 * now. The callback will be executed only when the continued
		 * task fully completes */
		/* the callback is executed after the dependencies so that we may remove the tag
		 * of the task itself */
		if (callback)
		{
			struct timespec *time = NULL;
			int profiling = starpu_profiling_status_get();
			if (profiling && task->profiling_info)
			{
				time = &task->profiling_info->callback_start_time;
				_starpu_clock_gettime(time);
			}
			enum _starpu_worker_status old_status = _starpu_get_local_worker_status();

			/* so that we can check whether we are doing blocking calls
			 * within the callback */
			if (!(old_status & STATUS_CALLBACK))
				_starpu_add_local_worker_status(STATUS_INDEX_CALLBACK, time);

			/* Perhaps we have nested callbacks (eg. with chains of empty
			 * tasks). So we store the current task and we will restore it
			 * later. */
			struct starpu_task *current_task = starpu_task_get_current();

			_starpu_set_current_task(task);

			_starpu_trace_start_callback(j);
			callback(task->callback_arg);
			_starpu_trace_end_callback(j);

			_starpu_set_current_task(current_task);

			if (profiling && task->profiling_info)
			{
				time = &task->profiling_info->callback_end_time;
				_starpu_clock_gettime(time);
			}

			if (!(old_status & STATUS_CALLBACK))
				_starpu_clear_local_worker_status(STATUS_INDEX_CALLBACK, time);
		}
	}

	/* Note: For now, we keep the TASK_DONE trace event for continuation,
	 * however we could add a specific event for stopped tasks if needed.
	 */
	_starpu_trace_task_done(j);

	STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);

	/* NB: we do not save those values before the callback, in case the
	 * application changes some parameters eventually (eg. a task may not
	 * be generated if the application is terminated). */
	unsigned destroy = task->destroy;
	unsigned detach = task->detach;
	unsigned regenerate = task->regenerate;
	unsigned synchronous = task->synchronous;

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
	task->prefetched = 0;
	STARPU_PTHREAD_COND_BROADCAST(&j->sync_cond);
	STARPU_AYU_REMOVETASK(j->job_id);
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

	/* we do not deallocate the job structure if some is going to
	 * wait after the task */
	if (detach && !continuation)
	{
		/* no one is going to synchronize with that task so we release
		 * the data structures now. In case the job was already locked
		 * by the caller, it is its responsibility to destroy the task.
		 * */
		if (destroy)
			_starpu_task_destroy(task);
	}

	/* A continuation is not much different from a regenerated task. */
	if (regenerate || continuation)
	{
		STARPU_ASSERT_MSG((detach && !destroy && !synchronous)
				|| continuation
				, "Regenerated task must be detached (was %u), and not have destroy=1 (was %u) or synchronous=1 (was %u)", detach, destroy, synchronous);
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
				int ret = _starpu_submit_job(j, 0);
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
		ret = 0;
	}

	return ret;
}

#ifdef STARPU_RECURSIVE_TASKS
int _starpu_recursive_task_unpartition_data_if_needed(struct _starpu_job *j)
{
	//_STARPU_DEBUG("[%s(%p)]\n", starpu_task_get_name(j->task), j->task);
	int unpartition_needed = 0;
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(j->task);
	unsigned nhandle = 0;
	unsigned i;
	struct starpu_task *control_task = NULL;

	for (i = 0; i < nbuffers; i++)
	{
		starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(j->task, i);
		enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(j->task, i);

		STARPU_PTHREAD_MUTEX_LOCK(&handle->unpartition_mutex);

		/**
		 * Version A
		 *
		 * We create a control task with the required data
		 * dependencies that will be automatically/magically
		 * handled by _starpu_data_partition_access_submit()
		 * called in _starpu_task_submit_head().
		 */
		if (handle->nplans > 0)
		{
			if (unpartition_needed == 0)
			{
				control_task = starpu_task_create();
				control_task->name = "ucontrol";
				_starpu_task_declare_deps_array(j->task, 1, &control_task, 0);

				unpartition_needed = 1;
			}

			//STARPU_TASK_SET_HANDLE(control_task, handle, nhandle);
			control_task->handles[nhandle] = handle;
			//STARPU_TASK_SET_MODE(control_task, mode, nhandle);
			control_task->modes[nhandle] = mode;
			nhandle ++;
		}
		/**
		 * Version B
		 *
		 * We find a way to call directly
		 * _starpu_data_partition_access_submit() here, and we
		 * (re-)plug the current task onto the last task
		 * generated by
		 * _starpu_data_partition_access_submit().
		 */
		else
		{
			//_starpu_data_partition_access_submit(handle, (mode & STARPU_W) != 0);
			// + replug on the current task
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&handle->unpartition_mutex);
	}

	// No data has been partitioned, let's keep going
	if (unpartition_needed == 0)
	{
		return 0;
	}

	// Add the dependency on the unpartition tasks
	STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
	j->task->status = STARPU_TASK_BLOCKED_ON_TASK;
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

	STARPU_ASSERT(control_task);
	int ret = starpu_task_submit(control_task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit(control_task)");

	return 1;
}

static int _starpu_turn_task_into_recursive_task(struct _starpu_job *j)
{
	if (j->already_turned_into_recursive_task)
	{
		/*
		 * We have first checked all dependencies of the recursive task,
		 * and secondly checked in a second stage the additional
		 * partition/unpartition dependencies
		 */
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		return 0;
	}
	j->already_turned_into_recursive_task = 1;
	//_STARPU_DEBUG("[%s(%p)]\n", starpu_task_get_name(j->task), j->task);

	if (j->is_recursive_task == 1)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		return 0;
	}
	else if (j->task->cl == NULL)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		return 0;
	}
	else
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		return _starpu_recursive_task_unpartition_data_if_needed(j);
	}
}

void _starpu_recursive_task_execute(struct _starpu_job *j)
{
	_starpu_trace_recursive_task(j);
	_starpu_trace_task_name_line_color(j);
	_starpu_trace_start_codelet_body(j, 0, NULL, 0, 0);
	STARPU_ASSERT_MSG(j->task->recursive_task_gen_dag_func!=NULL || (j->task->cl && j->task->cl->recursive_task_gen_dag_func!=NULL),
			  "task->recursive_task_gen_dag_func MUST be defined\n");

#ifdef STARPU_VERBOSE
	struct timespec tp;
	clock_gettime(CLOCK_MONOTONIC, &tp);
	unsigned long long timestamp = 1000000000ULL*tp.tv_sec + tp.tv_nsec;
	_STARPU_DEBUG("{%llu} [%s(%p)] Running recursive task\n", timestamp, starpu_task_get_name(j->task), j->task);
#endif
	if (j->task->recursive_task_gen_dag_func)
		j->task->recursive_task_gen_dag_func(j->task, j->task->recursive_task_gen_dag_func_arg);
	else
		j->task->cl->recursive_task_gen_dag_func(j->task, j->task->recursive_task_gen_dag_func_arg);
	j->task->where = STARPU_NOWHERE;
	_starpu_trace_end_codelet_body(j, 0, NULL, 0, 0);
}
#endif

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

	/* enforce tag dependencies */
	if (_starpu_not_all_tag_deps_are_fulfilled(j))
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		_STARPU_LOG_OUT_TAG("not_all_tag_deps_are_fulfilled");
		return 0;
	}

	/* enforce task dependencies */
	if (_starpu_not_all_task_deps_are_fulfilled(j))
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		_STARPU_LOG_OUT_TAG("not_all_task_deps_are_fulfilled");
		return 0;
	}

#ifdef STARPU_RECURSIVE_TASKS
	/* Wait for all dependencies at the correct level to be
	 * fulfilled before adding missing partition/unpartition
	 *
	 * If partition/unpartition are submitted we will enter the if
	 * case and come back later when these new dependencies are
	 * fulfilled
	 */
	if (_starpu_turn_task_into_recursive_task(j))
	{
		_STARPU_LOG_OUT_TAG("recursive_task");
		return 0;
	}
#else
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
#endif

#ifdef STARPU_RECURSIVE_TASKS
	if (j->is_recursive_task == 1)
	{
		_starpu_recursive_task_execute(j);
	}
	else
#endif
	{
		/* respect data concurrent access */
		if (_starpu_concurrent_data_access(j))
		{
			_STARPU_LOG_OUT_TAG("concurrent_data_access");
			return 0;
		}
	}

#ifdef STARPU_RECURSIVE_TASKS
	if (j->task->recursive_task_parent != 0)
		_starpu_trace_recursive_task_deps(j->task->recursive_task_parent, j);
#endif

	ret = _starpu_push_task(j);

	_STARPU_LOG_OUT();
	return ret;
}

/* Tag deps are already fulfilled */
unsigned _starpu_enforce_deps_starting_from_task(struct _starpu_job *j)
{
	unsigned ret;

	/* enforce task dependencies */
	if (_starpu_not_all_task_deps_are_fulfilled(j))
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		return 0;
	}
#ifdef STARPU_RECURSIVE_TASKS
	if (_starpu_turn_task_into_recursive_task(j))
	{
		_STARPU_LOG_OUT_TAG("recursive_task");
		return 0;
	}
#else
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
#endif

#ifdef STARPU_RECURSIVE_TASKS
	if (j->is_recursive_task == 1)
	{
		_starpu_recursive_task_execute(j);
	}
	else
#endif
	{
		/* respect data concurrent access */
		if (_starpu_concurrent_data_access(j))
			return 0;
	}

#ifdef STARPU_RECURSIVE_TASKS
	if (j->task->recursive_task_parent != 0)
		_starpu_trace_recursive_task_deps(j->task->recursive_task_parent, j);
#endif

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

	/* enforce task dependencies */
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

unsigned _starpu_take_deps_and_schedule(struct _starpu_job *j)
{
	unsigned ret;
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

	/* Take references */
	_starpu_submit_job_take_data_deps(j);

#ifdef STARPU_RECURSIVE_TASKS
	if (j->task->recursive_task_parent != 0)
		_starpu_trace_recursive_task_deps(j->task->recursive_task_parent, j);
#endif

	/* And immediately push task */
	ret = _starpu_push_task(j);

	return ret;
}

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

	if (!starpu_task_prio_list_empty(&worker->local_tasks))
		task = starpu_task_prio_list_pop_front_highest(&worker->local_tasks);

	_starpu_pop_task_end(task);
	return task;
}

int _starpu_push_local_task(struct _starpu_worker *worker, struct starpu_task *task)
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
		starpu_task_prio_list_push_back(&worker->local_tasks, task);
	}

	starpu_wake_worker_locked(worker->workerid);
	starpu_push_task_end(task);
	starpu_worker_unlock(worker->workerid);

	return 0;
}
