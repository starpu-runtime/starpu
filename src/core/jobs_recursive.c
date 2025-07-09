/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <common/config.h>
#include <common/utils.h>
#include <common/graph.h>
#include <core/dependencies/data_concurrency.h>
#include <core/debug.h>
#include <core/jobs.h>
#include <core/jobs_recursive.h>
#include <core/sched_policy.h>
#include <core/task.h>
#include <core/workers.h>
#include <datawizard/memory_nodes.h>
#include <profiling/splitter_bound.h>
#include <profiling/profiling.h>
#include <profiling/splitter_bound.h>

#ifdef STARPU_RECURSIVE_TASKS
#include <sched_policies/splitter.h>
#include <core/perfmodel/recursive_perfmodel.h>
// 1 to use the is_recursive_task_func of programmer, 0 to use StarPU internal splitter
static __volatile__ _Atomic unsigned long total_nb_workers = 0;
static unsigned long nb_tasks_ready = 0;

int (*splitter_policy) (struct _starpu_job*) = NULL;
unsigned call_splitter_on_scheduler = 0;
unsigned liberate_deps_on_first_exec = 0;

// some splitter push themselves tasks, so we do not need to push it, it has to set this variable to 0
static inline int splitter_func_look_programmer(struct _starpu_job *j)
{
	return (j->task->recursive_task_func && j->task->recursive_task_func(j->task, j->task->recursive_task_func_arg, _STARPU_TASK_GET_INTERFACES(j->task))) + (j->task->cl && j->task->cl->recursive_task_func && j->task->cl->recursive_task_func(j->task, j->task->recursive_task_func_arg, _STARPU_TASK_GET_INTERFACES(j->task)));
}

static inline int splitter_func_look_charge(void)
{
	return nb_tasks_ready < 2*total_nb_workers;
}

static inline int splitter_func_rectime_is_better(double rectime, double non_rectime)
{
	return rectime < 0.98 * non_rectime;
}

static inline int splitter_func_rectime_is_not_totally_uggly(double rectime, double non_rectime)
{
	return 2*rectime/total_nb_workers < non_rectime;
}

static inline int splitter_func_rectime_is_ok_compareto_nonrec_time(double rectime, double non_rectime)
{
	return rectime / (4 - (nb_tasks_ready/total_nb_workers)) < non_rectime;
}

static inline int splitter_func_look_rectime(struct _starpu_job *j, double rectime)
{
	unsigned long nb_tasks = nb_tasks_ready;
	double nonrec_time = starpu_task_expected_length_average(j->task, j->task->sched_ctx);
	if (splitter_func_rectime_is_better(rectime, nonrec_time))
		return 1; // this case, we win to divide the work compare to make it sequentially
	if (nb_tasks < total_nb_workers)
	{
		if (splitter_func_rectime_is_not_totally_uggly(rectime, nonrec_time))
			return 1;
	}
	else if (nb_tasks < 4*total_nb_workers)
	{
		if (splitter_func_rectime_is_ok_compareto_nonrec_time(rectime, nonrec_time))
			return 1;
	}
	return 0;
}

static inline int splitter_func_look_data(struct _starpu_job *j)
{
	// if we are here, we can look at the states of the handles used by j
	// returns 1 if there is a handle in W already partitioned

	int nbuffers = STARPU_TASK_GET_NBUFFERS(j->task);
	int i;
	for (i=0; i < nbuffers; i++)
	{
		if (STARPU_TASK_GET_MODE(j->task, i) & STARPU_W)
		{
			starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(j->task, i);
			if (handle->partitioned)
				return 1;
		}
	}
	return 0;
}

static int programmer_policy(struct _starpu_job *j)
{
	return splitter_func_look_programmer(j);
}

static int basic_splitter(struct _starpu_job *j)
{
	if (splitter_func_look_charge() && splitter_func_look_programmer(j))
	{
		return 1;
	}
	(void) STARPU_ATOMIC_ADD(&nb_tasks_ready, 1);
	return 0;
}

static int splitter_policy_improve_time(struct _starpu_job *j)
{
	if (splitter_func_look_programmer(j))
	{
		// this case, potentially we can have a recursive task
#if 0
		double rectime = _starpu_recursive_perfmodel_get_rectime(j->task);
		if (rectime >= 0.)
		{
			if (splitter_func_look_rectime(j, rectime))
				return 1;
		}
		else if (splitter_func_look_charge())
		{
			return 1;
		}
#endif
	}
	(void) STARPU_ATOMIC_ADD(&nb_tasks_ready, 1);
	return 0;
}

#define NMAX_HANDLES 100

static int splitter_policy_charge_with_gpus(struct _starpu_job *j)
{
	int back = 0;
	if (splitter_func_look_programmer(j))
	{
		return _splitter_choose_three_dimensions(j->task, 0, j->task->sched_ctx);
	}
	return back;
}

struct starpu_task *_starpu_recursive_task_unpartition_data_if_needed(struct _starpu_job *j);

void _starpu_rec_task_init(void)
{
	total_nb_workers = (unsigned long) starpu_worker_get_count();
	_starpu_recursive_perfmodel_init();
}

void _starpu_rec_task_deinit(void)
{
	_starpu_recursive_perfmodel_dump_created_subgraphs();
}

void _starpu_job_splitter_liberate_parent(struct _starpu_job *j)
{
	if (liberate_deps_on_first_exec && j->recursive.parent_task != NULL)
	{
		struct _starpu_job * pjob = _starpu_get_job_associated_to_task(j->recursive.parent_task);
		int need_term = 0;
		STARPU_PTHREAD_MUTEX_LOCK(&pjob->sync_mutex);
		need_term = !pjob->recursive.deps_free && pjob->recursive.already_end;
		if (need_term)
			pjob->recursive.deps_free ++;
		STARPU_PTHREAD_MUTEX_UNLOCK(&pjob->sync_mutex);
		if (need_term)
		{
			_STARPU_RECURSIVE_TASKS_DEBUG("Parent job %p is terminated.\n", pjob);
			_starpu_handle_job_termination(pjob);
		}
		STARPU_PTHREAD_MUTEX_LOCK(&pjob->sync_mutex);
		pjob->recursive.total_nchildren_end ++;
		STARPU_PTHREAD_MUTEX_UNLOCK(&pjob->sync_mutex);
	}
}

void _starpu_job_splitter_destroy(struct _starpu_job *j)
{
	if (liberate_deps_on_first_exec && j->recursive.parent_task != NULL)
	{
		struct _starpu_job * pjob = _starpu_get_job_associated_to_task(j->recursive.parent_task);
		int need_destroy = 0;
		STARPU_PTHREAD_MUTEX_LOCK(&pjob->sync_mutex);
		pjob->recursive.total_nchildren_destroy ++;
		if (pjob->recursive.already_end && pjob->recursive.total_nchildren_destroy == pjob->recursive.total_nchildren+1 && pjob->recursive.deps_free)
			need_destroy = 1;
		STARPU_PTHREAD_MUTEX_UNLOCK(&pjob->sync_mutex);
		if (need_destroy)
		{
			_STARPU_RECURSIVE_TASKS_DEBUG("Parent job %p is destroyed.\n", pjob);
			_starpu_recursive_job_destroy(pjob);
		}
	}
}

static inline void __starpu_job_is_ended(struct _starpu_job *j)
{
	_starpu_update_task_level_end(j->task);
#ifdef STARPU_HAVE_GLPK_H
	if (j->task && j->task->cl && j->task->where != STARPU_NOWHERE && j->task->cl->where != STARPU_NOWHERE)
	{
		_starpu_splitter_bound_delete(j);
	}
#endif
}

static void _starpu_job_one_real_subtask_is_ended(struct _starpu_job *j, enum starpu_worker_archtype task_executor)
{
	STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
	j->recursive.total_nsubtasks_cpu_end ++;
//	if (j->recursive.level == 0)
//		fprintf(stderr, "For task %p (%s), one task more is end : %d / %d ; nsubtot = %d/%d\n", j->task, j->task->name, j->recursive.total_nsubtasks_cpu_end, j->recursive.total_nsubtasks_cpu, j->recursive.total_nchildren_end, j->recursive.total_nchildren);
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
	if (j->recursive.parent_task)
		_starpu_job_one_real_subtask_is_ended(_starpu_get_job_associated_to_task(j->recursive.parent_task), task_executor);
}

void _starpu_job_splitter_termination(struct _starpu_job *j)
{
	STARPU_ASSERT_MSG(j->recursive.already_turned_into_recursive_task, "All tasks have to pass into the unpartition engine!\n");
	if (j->recursive.already_turned_into_recursive_task)
	{
		(void) STARPU_ATOMIC_ADD(&nb_tasks_ready, -1);

		if (j->recursive.scheduling && j->recursive.scheduling[j->recursive.ind_task_in_scheme] == '0' && j->recursive.parent_task && j->recursive.split_scheme && j->task->cl && !j->recursive.is_recursive_task)
		{
//			fprintf(stderr, "Call parent is end for %p %s\n", j->task, j->task->name);
			_starpu_job_one_real_subtask_is_ended(_starpu_get_job_associated_to_task(j->recursive.parent_task), STARPU_CPU_WORKER);
		}

		__starpu_job_is_ended(j);
	}
}

void _starpu_job_splitter_policy_init()
{
	int starpu_recursive_task_splitter_policy = starpu_getenv_number_default("STARPU_RECURSIVE_TASK_SPLITTER_POLICY", 0);
	if (starpu_recursive_task_splitter_policy == 0)
		splitter_policy = &programmer_policy;
	else if (starpu_recursive_task_splitter_policy == 1)
		splitter_policy = &basic_splitter;
	else if (starpu_recursive_task_splitter_policy == 2)
		splitter_policy = &splitter_policy_improve_time;
	else if (starpu_recursive_task_splitter_policy == 3)
	{
		splitter_policy = &splitter_policy_charge_with_gpus;
		starpu_bound_start(0, 0); // initiate glpk for computing bound
#ifdef STARPU_HAVE_GLPK_H
		_starpu_splitter_bound_start(); // initiate glpk for computing bound
#endif
	}
	liberate_deps_on_first_exec = starpu_getenv_number_default("STARPU_RECURSIVE_TASK_FREE_DEPS_ON_FIRST_REAL_TASK", 1);
	call_splitter_on_scheduler = starpu_getenv_number_default("STARPU_RECURSIVE_TASK_SPLITTER_CALL_ON_SCHEDULER", 1);
}

void _starpu_recursive_job_destroy(struct _starpu_job *j)
{
//	fprintf(stderr, "Destroy task %p %s\n", j->task, j->task->name);
	j->task->destroy = j->recursive.is_recursive_task_destroy;
	_starpu_splitter_bound_delete_split(j);
	_starpu_update_task_level_end(j->task);
	_starpu_job_splitter_termination(j);
	_starpu_task_destroy(j->task);
}

void _starpu_recursive_task_control_callback(void *arg)
{
	starpu_data_handle_t handle = (starpu_data_handle_t)arg;
	STARPU_PTHREAD_MUTEX_LOCK(handle->partition_mutex);
	if (handle->ctrl_unpartition)
	{
		handle->ctrl_unpartition->destroy = 1;
		handle->ctrl_unpartition = NULL;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(handle->partition_mutex);
}

// Returns the control task of the unpartition if it has be submitted, NULL else
struct starpu_task *_starpu_recursive_task_unpartition_data_if_needed(struct _starpu_job *j)
{
	_STARPU_DEBUG("[%d] %s(%p)\n", starpu_worker_get_id(), starpu_task_get_name(j->task), j->task);
	int unpartition_needed = 0;
	int check_required = 0;
	int ret;
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(j->task);
	unsigned ncontrol = 0;
	struct starpu_task *control_tasks[nbuffers];
	int controlled = 0;
	struct starpu_task *control_task_plus = NULL;
	unsigned i;
/*	int look_order[nbuffers];
	starpu_data_handle_t handlei, handlej;
	for (i = 0; i < nbuffers; i++)
		look_order[i] = i;
	for (i = 0; i < nbuffers; i++)
	{
		unsigned k;
		handlei = STARPU_TASK_GET_HANDLE(j->task, look_order[i]);
 		for (k = i+1; k < nbuffers; k++)
		{
			handlej = STARPU_TASK_GET_HANDLE(j->task, look_order[k]);
			if (handlei > handlej)
			{
				handlei = handlej;
				int tmp = look_order[i];
				look_order[i] = look_order[k];
				look_order[k] = tmp;
			}
		}
	}*/
	for (i = 0; i < nbuffers; i++)
	{
		starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(j->task, i);
		enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(j->task, i);
		if (!j->recursive.is_recursive_task_unpartitioned && !(mode & STARPU_RECURSIVE_TASK_STRONG))
			continue;
		struct starpu_task *control_task = NULL;

		/* If this is a partition task we shouldn't try to unpartition.
		 * If this is the unpartition for the first handle,
		 * we need to check if the children need unpartitioning. */
		STARPU_PTHREAD_MUTEX_LOCK(handle->partition_mutex);
		if (!check_required && j->task->cl == handle->switch_cl)
		{
			if (j->task == handle->last_unpartition)
			{
				/* Unpartition */
				check_required = 1;
				STARPU_PTHREAD_MUTEX_UNLOCK(handle->partition_mutex);
				continue;
			}
			else
			{
				/* Partition */
				_STARPU_DEBUG("[%p] This is a partition, break\n", j->task);
				STARPU_PTHREAD_MUTEX_UNLOCK(handle->partition_mutex);
				break;
			}
		}

		_STARPU_DEBUG("[%p] checking if data %d/%d: %p needs unpartitioning\n", j->task, i+1, nbuffers, handle);

		/**
		 * We create a control task with the required data
		 * dependencies that will be automatically/magically
		 * handled by _starpu_data_partition_access_submit()
		 * called in _starpu_task_submit_head().
		 */
		if (handle->partitioned > 0)
		{
			/* We first create a control task specific to this job
			   to prevent an already submitted control task from
			   pushing this job in another thread */
			if (!controlled)
			{
				controlled = 1;
				control_task_plus = starpu_task_create();
				control_task_plus->name = "ucontrol+";
				starpu_task_declare_deps(j->task, 1, control_task_plus);
			}
                        /* We create a control task for each data */
			unpartition_needed = 1;
			if (!handle->ctrl_unpartition)
			{
				control_task = starpu_task_create();
				control_task->name = "ucontrol";
				control_task->epilogue_callback_func = _starpu_recursive_task_control_callback;
				control_task->epilogue_callback_arg = handle;
				control_task->destroy = 0;
				control_tasks[ncontrol] = control_task;
				ncontrol++;
				unsigned prev_destroy = j->task->destroy;
				j->task->destroy = 0;
				starpu_task_declare_deps(j->task, 1, control_task);
				j->task->destroy = prev_destroy;
			}
			else
			{
				_STARPU_DEBUG("Unpartition needed. Plug the task into a previous ucontrol [%s(%p)]\n", starpu_task_get_name(j->task), j->task);

				unsigned prev_destroy = j->task->destroy;
				j->task->destroy = 0;
				starpu_task_declare_deps(j->task, 1, handle->ctrl_unpartition);
				j->task->destroy = prev_destroy;
				STARPU_PTHREAD_MUTEX_UNLOCK(handle->partition_mutex);
				continue; /* No need to go through the other tests */

			}
			if (!handle->last_unpartition)
			{
				_STARPU_DEBUG("Unpartition needed? Calling access_submit() [%s(%p)]\n", starpu_task_get_name(j->task), j->task);
				_starpu_data_partition_access_submit(handle, (mode & STARPU_W) != 0,  (mode & STARPU_W) != 0 && (mode & STARPU_R) == 0, control_task);
			}
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(handle->partition_mutex);
	}

	/* No control task were inserted, let's keep going */
	if (unpartition_needed == 0)
	{
		return NULL;
	}

	STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
//	STARPU_ASSERT(j->task->status != STARPU_TASK_READY);
	j->task->status = STARPU_TASK_BLOCKED_ON_TASK;
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

//	The control_task_plus task will be submitted by the caller once all the modification on the task has been made
//	ret = starpu_task_submit(control_task_plus);
//	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit(control_task_plus)");
	/* We loop to insert every control tasks */
	for (i=0; i<ncontrol; i++)
	{
		STARPU_ASSERT(control_tasks[i]);
		ret = starpu_task_submit(control_tasks[i]);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit(control_task)");
	}

	return control_task_plus;
}

#ifdef STARPU_HAVE_GLPK_H
static long starpu_njobs_have_been_ready= 0;
#endif

static inline void __starpu_job_is_ready(struct _starpu_job *j)
{
	_starpu_update_task_level(j->task);
#ifdef STARPU_HAVE_GLPK_H
	if (splitter_policy == &splitter_policy_charge_with_gpus && j->task && j->task->cl && j->task->where != STARPU_NOWHERE && j->task->cl->where != STARPU_NOWHERE)
	{
		_starpu_splitter_bound_record(j);
		if (STARPU_ATOMIC_ADD(&starpu_njobs_have_been_ready, 1) % 20 == 0)
		{
			_starpu_splitter_bound_calculate();
		}
	}
#endif
}

// returns 1 if the caller do not need to push the task
// for example, if there is a partition / unpartition submitted; or if the splitter function manage itself the push of task
// the returned task has to be submitted by the caller, except if we are on splitter_at_schedule mode
struct starpu_task * _starpu_turn_task_into_recursive_task(struct _starpu_job *j)
{
	if (j->recursive.already_turned_into_recursive_task)
	{
		/*
		 * We have first checked all dependencies of the recursive task,
		 * and secondly checked in a second stage the additional
		 * partition/unpartition dependencies
		 */

		/* So if we are here, we are ready to execute the task : data is unpartitioned
		 * We will retry a call to is_recursive_task, to change potentially the number of ready task
		 */

		__starpu_job_is_ready(j);
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		return NULL;
	}
	// before calling "is_recursive", we need to unpartition data if these are required for decision
	if (!j->recursive.is_recursive_task_unpartitioned && (j->task->cl != NULL || j->recursive.need_part_unpart))
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		struct starpu_task *cTask = _starpu_recursive_task_unpartition_data_if_needed(j);
		j->recursive.is_recursive_task_unpartitioned = 1;
		if (cTask)
		{
			if (!call_splitter_on_scheduler)
			{
				int ret =  starpu_task_submit(cTask);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit(control_task_plus)");
			}
			return cTask;
		}
		STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
	}
	j->recursive.is_recursive_task = splitter_policy(j);
	_STARPU_DEBUG("calling is recursive_task for job %p(%s) : %u\n", j, j->task->name, j->recursive.is_recursive_task);
	j->recursive.already_turned_into_recursive_task = 1;
	//_STARPU_DEBUG("[%s(%p)]\n", starpu_task_get_name(j->task), j->task);
	// Here, we make a call to is_recursive_task to know the good value

	if (j->recursive.is_recursive_task)
	{
		_starpu_splitter_bound_record_split(j);
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		return NULL;
	}
	else if (j->task->cl == NULL && !j->recursive.need_part_unpart)
	{
		_STARPU_DEBUG("Return without need part unpart for job %p\n", j);
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		__starpu_job_is_ready(j);
		return NULL;
	}
	else
	{
		_STARPU_DEBUG("Return By need part unpart for job %p\n", j);
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		struct starpu_task *cTask = _starpu_recursive_task_unpartition_data_if_needed(j);
		if (!call_splitter_on_scheduler && cTask)
		{
			int ret =  starpu_task_submit(cTask);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit(control_task_plus)");
		}
		if (!cTask)
		{
			// task ready,update level
			__starpu_job_is_ready(j);
		}
		return cTask;
	}
}

int _starpu_turn_task_into_recursive_task_at_scheduler(struct starpu_task *task)
{
	if (call_splitter_on_scheduler)
	{
		struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
		_STARPU_DEBUG("Blocking job %p while deciding if it is as recursive_task\n", j);
		enum starpu_task_status last_status = j->task->status;
		struct _starpu_worker *worker = _starpu_get_local_worker_key();
		int scheduler_was_pending = 0;
		if (worker)
		{
				STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
				scheduler_was_pending = worker->state_sched_op_pending;
				if (scheduler_was_pending)
					_starpu_worker_leave_sched_op(worker);
				STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
		}
		j->task->status = STARPU_TASK_BLOCKED;

		STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
		struct starpu_task *cTask = _starpu_turn_task_into_recursive_task(j);
		// if ret == 0 then we are ready to be executed and so we have to take the data lock
		int ret = 0;
		if (cTask)
		{	// We have submit an unpartition, we are not ready, so we need to release the concurrent lock
			_STARPU_DEBUG("Release task and data because we wait for an unpartition\n");
			// before wesay that we are not in scheduling
			_starpu_concurrent_data_release(j);
			_starpu_decrement_nready_tasks_of_sched_ctx(task->sched_ctx, task->flops);
			int sub =  starpu_task_submit(cTask);
			STARPU_CHECK_RETURN_VALUE(sub, "starpu_task_submit(control_task_plus)");
			ret = 1;
		}
		if (worker && scheduler_was_pending)
		{
			STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
			_starpu_worker_enter_sched_op(worker);
			STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
		}
		if(!ret)
		{
			// We can run ! unblock the task
			j->task->status = last_status;
		}
		return ret;
	}
	return 0;
}

void _starpu_recursive_task_execute(struct _starpu_job *j)
{
	_STARPU_DEBUG("We generate a subgraphfor job %p(%s)t\n", j, j->task->name);
	_starpu_trace_recursive_task(j);
	_starpu_trace_task_name_line_color(j);
	_starpu_trace_start_codelet_body(j, 0, NULL, starpu_worker_get_id(), 0);
	STARPU_ASSERT_MSG(j->task->recursive_task_gen_dag_func!=NULL || (j->task->cl && j->task->cl->recursive_task_gen_dag_func!=NULL),
			  "task->recursive_task_gen_dag_func MUST be defined\n");

#ifdef STARPU_VERBOSE
	struct timespec tp;
	clock_gettime(CLOCK_MONOTONIC, &tp);
	unsigned long long timestamp = 1000000000ULL*tp.tv_sec + tp.tv_nsec;
	_STARPU_DEBUG("{%llu} [%s(%p)] Running recursive task\n", timestamp, starpu_task_get_name(j->task), j->task);
#endif
	struct starpu_task *recursive_task_generating = _starpu_recursive_task_which_generate_dag();
	starpu_pthread_setspecific(_starpu_pthread_is_on_recursive_task_key, (void*)(uintptr_t) j->task);
	struct _starpu_recursive_perfmodel_subgraph *created_subgraph = NULL;
	int use_recursive_perfmodel = j && j->task && j->task->cl && j->task->cl->model;
	if (use_recursive_perfmodel)
	{
		STARPU_PTHREAD_RWLOCK_RDLOCK(&j->task->cl->model->state->model_rwlock);
		if (_starpu_recursive_perfmodel_get_subgraph_from_task(j->task) == NULL)
		{
			created_subgraph = _starpu_recursive_perfmodel_create_subgraph_from_task(j->task);
			j->recursive.subgraph_created = created_subgraph;
		}
		STARPU_PTHREAD_RWLOCK_UNLOCK(&j->task->cl->model->state->model_rwlock);
	}
	if (j->task->recursive_task_gen_dag_func)
		j->task->recursive_task_gen_dag_func(j->task, j->task->recursive_task_gen_dag_func_arg, _STARPU_TASK_GET_INTERFACES(j->task));
	else
		j->task->cl->recursive_task_gen_dag_func(j->task, j->task->recursive_task_gen_dag_func_arg, _STARPU_TASK_GET_INTERFACES(j->task));

	if (created_subgraph)
	{
		created_subgraph->subgraph_initialisation_is_finished = 1;
	}
	starpu_pthread_setspecific(_starpu_pthread_is_on_recursive_task_key, recursive_task_generating);

//	j->task->where = STARPU_NOWHERE;
	_starpu_trace_end_codelet_body(j, 0, NULL, starpu_worker_get_id(), 0);
}

int _starpu_job_is_recursive(struct _starpu_job *j)
{
	return j->recursive.is_recursive_task;
}

int _starpu_job_generate_dag_if_needed(struct _starpu_job *j)
{
	if (j->recursive.is_recursive_task)
	{
		j->task->status = STARPU_TASK_RUNNING;
		if (liberate_deps_on_first_exec)
		{
			j->recursive.is_recursive_task_destroy = j->task->destroy;
			j->task->destroy = 0;
		}

		_starpu_recursive_task_execute(j);
		_STARPU_DEBUG("Recursive_Task %p. Terminate job(%s)\n", j, j->task->name);
		return 1;
	}
	return 0;
}

void _starpu_handle_recursive_task_termination(struct _starpu_job *j)
{
	STARPU_ASSERT_MSG(j->recursive.is_recursive_task, "handle_recursive_task_termination has to be called only with submission tasks\n");
	_starpu_sched_post_exec_hook(j->task);
	if (liberate_deps_on_first_exec)
	{
		int need_term = 0, need_destroy = 0;
		STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
		if (j->recursive.total_nchildren_end > 0 || j->recursive.total_nchildren == 0)
		{
			need_term = 1;
			j->recursive.deps_free = 1;
		}
		j->recursive.already_end = 1;
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		if (need_term)
		{
			_STARPU_DEBUG("Job recursive task %p terminate itself\n", j);
			//fprintf(stderr, "Job recursive task %p terminate itself\n", j);
			_starpu_handle_job_termination(j);
		}
		STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
		j->recursive.total_nchildren_end ++;
		j->recursive.total_nchildren_destroy ++;
		if (j->recursive.total_nchildren_destroy == j->recursive.total_nchildren+1)
		 {
			 need_destroy = 1;
		 }
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		if (need_destroy)
		{
			_STARPU_DEBUG("Destroy job recursive task %p by itself\n", j);
			//fprintf(stderr, "Destroy job recursive task %p by itself\n", j);
			_starpu_recursive_job_destroy(j);
		}
	}
	else
	{
		_starpu_handle_job_termination(j);
	}
}
long _starpu_get_cuda_executor_id(struct starpu_task *t)
{
	struct _starpu_job *j = _starpu_get_job_associated_to_task(t);
	struct _starpu_job *pjob = j->recursive.parent_task == NULL ? NULL : _starpu_get_job_associated_to_task(j->recursive.parent_task);
	if (pjob == NULL || !pjob->recursive.split_scheme)
		return j->recursive.cuda_worker_executor_id;
	return _starpu_get_cuda_executor_id(j->recursive.parent_task);
}
void _starpu_set_cuda_executor_id(struct starpu_task *t, long id)
{
	struct _starpu_job *j = _starpu_get_job_associated_to_task(t);
	if (j->recursive.split_scheme == NULL || j->recursive.recursive_mode == SPLITTER_LITTLE_GPU || j->recursive.scheduling[j->recursive.ind_task_in_scheme] == '0')
		return;
	struct _starpu_job *pjob = j->recursive.parent_task == NULL ? NULL : _starpu_get_job_associated_to_task(j->recursive.parent_task);
	if (pjob == NULL || !pjob->recursive.split_scheme)
		j->recursive.cuda_worker_executor_id = id;
	else
		_starpu_set_cuda_executor_id(j->recursive.parent_task, id);
}

#endif
