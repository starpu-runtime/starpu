/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Thibaut Lambert
 * Copyright (C) 2016       Uppsala University
 * Copyright (C) 2017       Erwan Leria
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
#include <core/workers.h>
#include <core/sched_ctx.h>
#include <core/jobs.h>
#include <core/task.h>
#include <core/task_bundle.h>
#include <core/dependencies/data_concurrency.h>
#include <common/config.h>
#include <common/utils.h>
#include <common/fxt.h>
#include <profiling/profiling.h>
#include <profiling/bound.h>
#include <math.h>
#include <string.h>
#include <core/debug.h>
#include <core/sched_ctx.h>
#include <time.h>
#include <signal.h>
#include <core/simgrid.h>
#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif

/* XXX this should be reinitialized when StarPU is shutdown (or we should make
 * sure that no task remains !) */
/* TODO we could make this hierarchical to avoid contention ? */
//static starpu_pthread_cond_t submitted_cond = STARPU_PTHREAD_COND_INITIALIZER;

/* This key stores the task currently handled by the thread, note that we
 * cannot use the worker structure to store that information because it is
 * possible that we have a task with a NULL codelet, which means its callback
 * could be executed by a user thread as well. */
static starpu_pthread_key_t current_task_key;
static int limit_min_submitted_tasks;
static int limit_max_submitted_tasks;
static int watchdog_crash;
static int watchdog_delay;

/*
 * Function to call when watchdog detects that no task has finished for more than STARPU_WATCHDOG_TIMEOUT seconds
 */
static void (*watchdog_hook)(void *) = NULL;
static void * watchdog_hook_arg = NULL;

#define _STARPU_TASK_MAGIC 42

/* Called once at starpu_init */
void _starpu_task_init(void)
{
	STARPU_PTHREAD_KEY_CREATE(&current_task_key, NULL);
	limit_min_submitted_tasks = starpu_get_env_number("STARPU_LIMIT_MIN_SUBMITTED_TASKS");
	limit_max_submitted_tasks = starpu_get_env_number("STARPU_LIMIT_MAX_SUBMITTED_TASKS");
	watchdog_crash = starpu_get_env_number("STARPU_WATCHDOG_CRASH");
	watchdog_delay = starpu_get_env_number_default("STARPU_WATCHDOG_DELAY", 0);
}

void _starpu_task_deinit(void)
{
	STARPU_PTHREAD_KEY_DELETE(current_task_key);
}

void starpu_task_init(struct starpu_task *task)
{
	/* TODO: memcpy from a template instead? benchmark it */

	STARPU_ASSERT(task);

	/* As most of the fields must be initialised at NULL, let's put 0
	 * everywhere */
	memset(task, 0, sizeof(struct starpu_task));

	task->sequential_consistency = 1;
	task->where = -1;

	/* Now we can initialise fields which recquire custom value */
	/* Note: remember to update STARPU_TASK_INITIALIZER as well */
#if STARPU_DEFAULT_PRIO != 0
	task->priority = STARPU_DEFAULT_PRIO;
#endif

	task->detach = 1;

#if STARPU_TASK_INIT != 0
	task->status = STARPU_TASK_INIT;
#endif

	task->predicted = NAN;
	task->predicted_transfer = NAN;
	task->predicted_start = NAN;

	task->magic = _STARPU_TASK_MAGIC;
	task->sched_ctx = STARPU_NMAX_SCHED_CTXS;

	task->flops = 0.0;
}

/* Free all the ressources allocated for a task, without deallocating the task
 * structure itself (this is required for statically allocated tasks).
 * All values previously set by the user, like codelet and handles, remain
 * unchanged */
void starpu_task_clean(struct starpu_task *task)
{
	STARPU_ASSERT(task);
	task->magic = 0;

	/* If a buffer was allocated to store the profiling info, we free it. */
	if (task->profiling_info)
	{
		free(task->profiling_info);
		task->profiling_info = NULL;
	}

	/* If case the task is (still) part of a bundle */
	starpu_task_bundle_t bundle = task->bundle;
	if (bundle)
		starpu_task_bundle_remove(bundle, task);

	if (task->dyn_handles)
	{
		free(task->dyn_handles);
		task->dyn_handles = NULL;
		free(task->dyn_interfaces);
		task->dyn_interfaces = NULL;
	}

	if (task->dyn_modes)
	{
		free(task->dyn_modes);
		task->dyn_modes = NULL;
	}

	struct _starpu_job *j = (struct _starpu_job *)task->starpu_private;

	if (j)
	{
		_starpu_job_destroy(j);
		task->starpu_private = NULL;
	}
}

struct starpu_task * STARPU_ATTRIBUTE_MALLOC starpu_task_create(void)
{
	struct starpu_task *task;

	_STARPU_MALLOC(task, sizeof(struct starpu_task));
	starpu_task_init(task);

	/* Dynamically allocated tasks are destroyed by default */
	task->destroy = 1;

	return task;
}

/* Free the ressource allocated during starpu_task_create. This function can be
 * called automatically after the execution of a task by setting the "destroy"
 * flag of the starpu_task structure (default behaviour). Calling this function
 * on a statically allocated task results in an undefined behaviour. */
void _starpu_task_destroy(struct starpu_task *task)
{

	/* If starpu_task_destroy is called in a callback, we just set the destroy
	   flag. The task will be destroyed after the callback returns */
	if (task == starpu_task_get_current()
	    && _starpu_get_local_worker_status() == STATUS_CALLBACK)
	{
		task->destroy = 1;
	}
	else
	{
		starpu_task_clean(task);
		/* TODO handle the case of task with detach = 1 and destroy = 1 */
		/* TODO handle the case of non terminated tasks -> assertion failure, it's too dangerous to be doing something like this */

		/* Does user want StarPU release cl_arg ? */
		if (task->cl_arg_free)
			free(task->cl_arg);

		/* Does user want StarPU release callback_arg ? */
		if (task->callback_arg_free)
			free(task->callback_arg);

		/* Does user want StarPU release prologue_callback_arg ? */
		if (task->prologue_callback_arg_free)
			free(task->prologue_callback_arg);

		/* Does user want StarPU release prologue_pop_arg ? */
		if (task->prologue_callback_pop_arg_free)
			free(task->prologue_callback_pop_arg);

		free(task);
	}
}

void starpu_task_destroy(struct starpu_task *task)
{
	STARPU_ASSERT(task);
	STARPU_ASSERT_MSG(!task->destroy || !task->detach, "starpu_task_destroy must not be called for task with destroy = 1 and detach = 1");
	_starpu_task_destroy(task);
}

int starpu_task_finished(struct starpu_task *task)
{
	STARPU_ASSERT(task);
	STARPU_ASSERT_MSG(!task->detach, "starpu_task_finished can only be called on tasks with detach = 0");
	return _starpu_job_finished(_starpu_get_job_associated_to_task(task));
}

int starpu_task_wait(struct starpu_task *task)
{
        _STARPU_LOG_IN();
	STARPU_ASSERT(task);

	STARPU_ASSERT_MSG(!task->detach, "starpu_task_wait can only be called on tasks with detach = 0");

	if (task->detach || task->synchronous)
	{
		_STARPU_DEBUG("Task is detached or synchronous. Waiting returns immediately\n");
		_STARPU_LOG_OUT_TAG("einval");
		return -EINVAL;
	}

	STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "starpu_task_wait must not be called from a task or callback");

	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	_STARPU_TRACE_TASK_WAIT_START(j);

	starpu_do_schedule();
	_starpu_wait_job(j);

	/* as this is a synchronous task, the liberation of the job
	   structure was deferred */
	if (task->destroy)
		_starpu_task_destroy(task);

	_STARPU_TRACE_TASK_WAIT_END();
        _STARPU_LOG_OUT();
	return 0;
}

int starpu_task_wait_array(struct starpu_task **tasks, unsigned nb_tasks)
{
	unsigned i;

	for (i = 0; i < nb_tasks; i++)
	{
		int ret = starpu_task_wait(tasks[i]);
		if (ret)
			return ret;
	}
	return 0;
}

#ifdef STARPU_OPENMP
int _starpu_task_test_termination(struct starpu_task *task)
{
	STARPU_ASSERT(task);
	STARPU_ASSERT_MSG(!task->detach, "starpu_task_wait can only be called on tasks with detach = 0");

	if (task->detach || task->synchronous)
	{
		_STARPU_DEBUG("Task is detached or synchronous\n");
		_STARPU_LOG_OUT_TAG("einval");
		return -EINVAL;
	}

	struct _starpu_job *j = (struct _starpu_job *)task->starpu_private;

	int ret = _starpu_test_job_termination(j);

	if (ret)
	{
		if (task->destroy)
			_starpu_task_destroy(task);
	}

	return ret;
}
#endif

/* NB in case we have a regenerable task, it is possible that the job was
 * already counted. */
int _starpu_submit_job(struct _starpu_job *j)
{
	struct starpu_task *task = j->task;
	int ret;
#ifdef STARPU_OPENMP
	const unsigned continuation = j->continuation;
#else
	const unsigned continuation = 0;
#endif

	_STARPU_LOG_IN();
	/* notify bound computation of a new task */
	_starpu_bound_record(j);

	_starpu_increment_nsubmitted_tasks_of_sched_ctx(j->task->sched_ctx);
	_starpu_sched_task_submit(task);

#ifdef STARPU_USE_SC_HYPERVISOR
	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(j->task->sched_ctx);
	if(sched_ctx != NULL && j->task->sched_ctx != _starpu_get_initial_sched_ctx()->id && j->task->sched_ctx != STARPU_NMAX_SCHED_CTXS
	   && sched_ctx->perf_counters != NULL)
	{
		struct starpu_perfmodel_arch arch;
		_STARPU_MALLOC(arch.devices, sizeof(struct starpu_perfmodel_device));
		arch.ndevices = 1;
		arch.devices[0].type = STARPU_CPU_WORKER;
		arch.devices[0].devid = 0;
		arch.devices[0].ncores = 1;
		_starpu_compute_buffers_footprint(j->task->cl->model, &arch, 0, j);
		free(arch.devices);
		size_t data_size = 0;
		if (j->task->cl)
		{
			unsigned i, nbuffers = STARPU_TASK_GET_NBUFFERS(j->task);
			for(i = 0; i < nbuffers; i++)
			{
				starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, i);
				if (handle != NULL)
					data_size += _starpu_data_get_size(handle);
			}
		}

		_STARPU_TRACE_HYPERVISOR_BEGIN();
		sched_ctx->perf_counters->notify_submitted_job(j->task, j->footprint, data_size);
		_STARPU_TRACE_HYPERVISOR_END();
	}
#endif//STARPU_USE_SC_HYPERVISOR

	/* We retain handle reference count */
	if (task->cl && !continuation)
	{
		unsigned i;
		unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
		for (i=0; i<nbuffers; i++)
		{
			starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, i);
			_starpu_spin_lock(&handle->header_lock);
			handle->busy_count++;
			_starpu_spin_unlock(&handle->header_lock);
		}
	}

	STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);

	_starpu_handle_job_submission(j);

#ifdef STARPU_OPENMP
	if (continuation)
	{
		j->discontinuous = 1;
		j->continuation  = 0;
	}
#endif

#ifdef STARPU_OPENMP
	if (continuation)
	{
		ret = _starpu_reenforce_task_deps_and_schedule(j);
	}
	else
#endif
	{
		ret = _starpu_enforce_deps_and_schedule(j);
	}

	_STARPU_LOG_OUT();
	return ret;
}

/* Note: this is racy, so valgrind would complain. But since we'll always put
 * the same values, this is not a problem. */
void _starpu_codelet_check_deprecated_fields(struct starpu_codelet *cl)
{
	if (!cl)
		return;
	if (cl->checked)
	{
		STARPU_RMB();
		return;
	}

	uint32_t where = cl->where;
	int is_where_unset = where == 0;
	unsigned i, some_impl;

	/* Check deprecated and unset fields (where, <device>_func,
 	 * <device>_funcs) */

	/* CPU */
	if (cl->cpu_func && cl->cpu_func != STARPU_MULTIPLE_CPU_IMPLEMENTATIONS && cl->cpu_funcs[0])
	{
		_STARPU_DISP("[warning] [struct starpu_codelet] both cpu_func and cpu_funcs are set. Ignoring cpu_func.\n");
		cl->cpu_func = STARPU_MULTIPLE_CPU_IMPLEMENTATIONS;
	}
	if (cl->cpu_func && cl->cpu_func != STARPU_MULTIPLE_CPU_IMPLEMENTATIONS)
	{
		cl->cpu_funcs[0] = cl->cpu_func;
		cl->cpu_func = STARPU_MULTIPLE_CPU_IMPLEMENTATIONS;
	}
	some_impl = 0;
	for (i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
		if (cl->cpu_funcs[i])
		{
			some_impl = 1;
			break;
		}
	if (some_impl && cl->cpu_func == 0)
	{
		cl->cpu_func = STARPU_MULTIPLE_CPU_IMPLEMENTATIONS;
	}
	if (some_impl && is_where_unset)
	{
		where |= STARPU_CPU;
	}

	/* CUDA */
	if (cl->cuda_func && cl->cuda_func != STARPU_MULTIPLE_CUDA_IMPLEMENTATIONS && cl->cuda_funcs[0])
	{
		_STARPU_DISP("[warning] [struct starpu_codelet] both cuda_func and cuda_funcs are set. Ignoring cuda_func.\n");
		cl->cuda_func = STARPU_MULTIPLE_CUDA_IMPLEMENTATIONS;
	}
	if (cl->cuda_func && cl->cuda_func != STARPU_MULTIPLE_CUDA_IMPLEMENTATIONS)
	{
		cl->cuda_funcs[0] = cl->cuda_func;
		cl->cuda_func = STARPU_MULTIPLE_CUDA_IMPLEMENTATIONS;
	}
	some_impl = 0;
	for (i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
		if (cl->cuda_funcs[i])
		{
			some_impl = 1;
			break;
		}
	if (some_impl && cl->cuda_func == 0)
	{
		cl->cuda_func = STARPU_MULTIPLE_CUDA_IMPLEMENTATIONS;
	}
	if (some_impl && is_where_unset)
	{
		where |= STARPU_CUDA;
	}

	/* OpenCL */
	if (cl->opencl_func && cl->opencl_func != STARPU_MULTIPLE_OPENCL_IMPLEMENTATIONS && cl->opencl_funcs[0])
	{
		_STARPU_DISP("[warning] [struct starpu_codelet] both opencl_func and opencl_funcs are set. Ignoring opencl_func.\n");
		cl->opencl_func = STARPU_MULTIPLE_OPENCL_IMPLEMENTATIONS;
	}
	if (cl->opencl_func && cl->opencl_func != STARPU_MULTIPLE_OPENCL_IMPLEMENTATIONS)
	{
		cl->opencl_funcs[0] = cl->opencl_func;
		cl->opencl_func = STARPU_MULTIPLE_OPENCL_IMPLEMENTATIONS;
	}
	some_impl = 0;
	for (i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
		if (cl->opencl_funcs[i])
		{
			some_impl = 1;
			break;
		}
	if (some_impl && cl->opencl_func == 0)
	{
		cl->opencl_func = STARPU_MULTIPLE_OPENCL_IMPLEMENTATIONS;
	}
	if (some_impl && is_where_unset)
	{
		where |= STARPU_OPENCL;
	}

	some_impl = 0;
	for (i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
		if (cl->mic_funcs[i])
		{
			some_impl = 1;
			break;
		}
	if (some_impl && is_where_unset)
	{
		where |= STARPU_MIC;
	}

	some_impl = 0;
	for (i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
		if (cl->mpi_ms_funcs[i])
		{
			some_impl = 1;
			break;
		}
	if (some_impl && is_where_unset)
	{
		where |= STARPU_MPI_MS;
	}

	some_impl = 0;
	for (i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
		if (cl->cpu_funcs_name[i])
		{
			some_impl = 1;
			break;
		}
	if (some_impl && is_where_unset)
	{
		where |= STARPU_MIC|STARPU_MPI_MS;
	}
	cl->where = where;

	STARPU_WMB();
	cl->checked = 1;
}

void _starpu_task_check_deprecated_fields(struct starpu_task *task STARPU_ATTRIBUTE_UNUSED)
{
	/* None any more */
}

static int _starpu_task_submit_head(struct starpu_task *task)
{
	unsigned is_sync = task->synchronous;
	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	if (task->status == STARPU_TASK_STOPPED || task->status == STARPU_TASK_FINISHED)
		task->status = STARPU_TASK_INIT;
	else
		STARPU_ASSERT(task->status == STARPU_TASK_INIT);

	if (j->internal)
	{
		// Internal tasks are submitted to initial context
		task->sched_ctx = _starpu_get_initial_sched_ctx()->id;
	}
	else if (task->sched_ctx == STARPU_NMAX_SCHED_CTXS)
	{
		// If the task has not specified a context, we set the current context
		task->sched_ctx = _starpu_sched_ctx_get_current_context();
	}

	if (is_sync)
	{
		/* Perhaps it is not possible to submit a synchronous
		 * (blocking) task */
		STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "submitting a synchronous task must not be done from a task or a callback");
		task->detach = 0;
	}

	_starpu_task_check_deprecated_fields(task);
	_starpu_codelet_check_deprecated_fields(task->cl);
	if (task->where== -1 && task->cl)
		task->where = task->cl->where;

	if (task->cl)
	{
		unsigned i;
		unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
		_STARPU_TRACE_UPDATE_TASK_CNT(0);

		/* Check buffers */
		if (task->dyn_handles == NULL)
			STARPU_ASSERT_MSG(STARPU_TASK_GET_NBUFFERS(task) <= STARPU_NMAXBUFS,
					  "Codelet %p has too many buffers (%d vs max %d). Either use --enable-maxbuffers configure option to increase the max, or use dyn_handles instead of handles.",
					  task->cl, STARPU_TASK_GET_NBUFFERS(task), STARPU_NMAXBUFS);

		if (task->dyn_handles)
		{
			_STARPU_MALLOC(task->dyn_interfaces, nbuffers * sizeof(void *));
		}

		for (i = 0; i < nbuffers; i++)
		{
			starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, i);
			enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, i);
			int node = task->cl->specific_nodes ? STARPU_CODELET_GET_NODE(task->cl, i) : -1;
			/* Make sure handles are valid */
			STARPU_ASSERT_MSG(handle->magic == _STARPU_TASK_MAGIC, "data %p is invalid (was it already unregistered?)", handle);
			/* Make sure handles are not partitioned */
			STARPU_ASSERT_MSG(handle->nchildren == 0, "only unpartitioned data (or the pieces of a partitioned data) can be used in a task");
			/* Make sure the specified node exists */
			STARPU_ASSERT_MSG(node == STARPU_SPECIFIC_NODE_LOCAL || node == STARPU_SPECIFIC_NODE_CPU || node == STARPU_SPECIFIC_NODE_SLOW || (node >= 0 && node < (int) starpu_memory_nodes_get_count()), "The codelet-specified memory node does not exist");
			/* Provide the home interface for now if any,
			 * for can_execute hooks */
			if (handle->home_node != -1)
				_STARPU_TASK_SET_INTERFACE(task, starpu_data_get_interface_on_node(handle, handle->home_node), i);
			if (!(task->cl->flags & STARPU_CODELET_NOPLANS) &&
			    ((handle->nplans && !handle->nchildren) || handle->siblings)
			    && handle->partition_automatic_disabled == 0
			    )
				/* This handle is involved with asynchronous
				 * partitioning as a parent or a child, make
				 * sure the right plan is active, submit
				 * appropiate partitioning / unpartitioning if
				 * not */
				_starpu_data_partition_access_submit(handle, (mode & STARPU_W) != 0);
		}

		/* Check the type of worker(s) required by the task exist */
		if (!_starpu_worker_exists(task))
		{
			_STARPU_LOG_OUT_TAG("ENODEV");
			return -ENODEV;
		}

		/* In case we require that a task should be explicitely
		 * executed on a specific worker, we make sure that the worker
		 * is able to execute this task.  */
		if (task->execute_on_a_specific_worker && !starpu_combined_worker_can_execute_task(task->workerid, task, 0))
		{
			_STARPU_LOG_OUT_TAG("ENODEV");
			return -ENODEV;
		}

		if (task->cl->model)
			_starpu_init_and_load_perfmodel(task->cl->model);

		if (task->cl->energy_model)
			_starpu_init_and_load_perfmodel(task->cl->energy_model);
	}

	return 0;
}

/* application should submit new tasks to StarPU through this function */
int starpu_task_submit(struct starpu_task *task)
{
	_STARPU_LOG_IN();
	STARPU_ASSERT(task);
	STARPU_ASSERT_MSG(task->magic == _STARPU_TASK_MAGIC, "Tasks must be created with starpu_task_create, or initialized with starpu_task_init.");
	STARPU_ASSERT_MSG(starpu_is_initialized(), "starpu_init must be called (and return no error) before submitting tasks.");

	int ret;
	unsigned is_sync = task->synchronous;
	starpu_task_bundle_t bundle = task->bundle;
	/* internally, StarPU manipulates a struct _starpu_job * which is a wrapper around a
	* task structure, it is possible that this job structure was already
	* allocated. */
	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
	const unsigned continuation =
#ifdef STARPU_OPENMP
		j->continuation
#else
		0
#endif
		;

	if (!j->internal)
	{
		int nsubmitted_tasks = starpu_task_nsubmitted();
		if (limit_max_submitted_tasks >= 0 && limit_max_submitted_tasks < nsubmitted_tasks
			&& limit_min_submitted_tasks >= 0 && limit_min_submitted_tasks < nsubmitted_tasks)
		{
			starpu_do_schedule();
			_STARPU_TRACE_TASK_THROTTLE_START();
			starpu_task_wait_for_n_submitted(limit_min_submitted_tasks);
			_STARPU_TRACE_TASK_THROTTLE_END();
		}
	}

	_STARPU_TRACE_TASK_SUBMIT_START();

	ret = _starpu_task_submit_head(task);
	if (ret)
	{
		_STARPU_TRACE_TASK_SUBMIT_END();
		return ret;
	}

	if (!continuation)
	{
		STARPU_ASSERT_MSG(!j->submitted || j->terminated >= 1, "Tasks can not be submitted a second time before being terminated. Please use different task structures, or use the regenerate flag to let the task resubmit itself automatically.");
		_STARPU_TRACE_TASK_SUBMIT(j,
			_starpu_get_sched_ctx_struct(task->sched_ctx)->iterations[0],
			_starpu_get_sched_ctx_struct(task->sched_ctx)->iterations[1]);
	}

	/* If this is a continuation, we don't modify the implicit data dependencies detected earlier. */
	if (task->cl && !continuation)
	{
		_starpu_job_set_ordered_buffers(j);
		_starpu_detect_implicit_data_deps(task);
	}

	if (bundle)
	{
		/* We need to make sure that models for other tasks of the
		 * bundle are also loaded, so the scheduler can estimate the
		 * duration of the whole bundle */
		STARPU_PTHREAD_MUTEX_LOCK(&bundle->mutex);

		struct _starpu_task_bundle_entry *entry;
		entry = bundle->list;

		while (entry)
		{
			if (entry->task->cl->model)
				_starpu_init_and_load_perfmodel(entry->task->cl->model);

			if (entry->task->cl->energy_model)
				_starpu_init_and_load_perfmodel(entry->task->cl->energy_model);

			entry = entry->next;
		}

		STARPU_PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
	}

	/* If profiling is activated, we allocate a structure to store the
	 * appropriate info. */
	struct starpu_profiling_task_info *info;
	int profiling = starpu_profiling_status_get();
	info = _starpu_allocate_profiling_info_if_needed(task);
	task->profiling_info = info;

	/* The task is considered as block until we are sure there remains not
	 * dependency. */
	task->status = STARPU_TASK_BLOCKED;

	if (profiling)
		_starpu_clock_gettime(&info->submit_time);

	ret = _starpu_submit_job(j);
#ifdef STARPU_SIMGRID
	if (_starpu_simgrid_task_submit_cost())
		starpu_sleep(0.000001);
#endif

	if (is_sync)
	{
		_starpu_sched_do_schedule(task->sched_ctx);
		_starpu_wait_job(j);
		if (task->destroy)
		     _starpu_task_destroy(task);
	}

	_STARPU_TRACE_TASK_SUBMIT_END();
        _STARPU_LOG_OUT();
	return ret;
}

int _starpu_task_submit_internally(struct starpu_task *task)
{
	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
	j->internal = 1;
	return starpu_task_submit(task);
}

/* application should submit new tasks to StarPU through this function */
int starpu_task_submit_to_ctx(struct starpu_task *task, unsigned sched_ctx_id)
{
	task->sched_ctx = sched_ctx_id;
	return starpu_task_submit(task);
}

/* The StarPU core can submit tasks directly to the scheduler or a worker,
 * skipping dependencies completely (when it knows what it is doing).  */
int _starpu_task_submit_nodeps(struct starpu_task *task)
{
	int ret = _starpu_task_submit_head(task);
	STARPU_ASSERT(ret == 0);

	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	_starpu_increment_nsubmitted_tasks_of_sched_ctx(j->task->sched_ctx);
	_starpu_sched_task_submit(task);

	STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
	_starpu_handle_job_submission(j);
	_starpu_increment_nready_tasks_of_sched_ctx(j->task->sched_ctx, j->task->flops, j->task);
	if (task->cl)
		/* This would be done by data dependencies checking */
		_starpu_job_set_ordered_buffers(j);
	STARPU_ASSERT(task->status == STARPU_TASK_BLOCKED);
	task->status = STARPU_TASK_READY;
	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

	return _starpu_push_task(j);
}

/*
 * worker->sched_mutex must be locked when calling this function.
 */
int _starpu_task_submit_conversion_task(struct starpu_task *task,
					unsigned int workerid)
{
	int ret;
	STARPU_ASSERT(task->cl);
	STARPU_ASSERT(task->execute_on_a_specific_worker);

	ret = _starpu_task_submit_head(task);
	STARPU_ASSERT(ret == 0);

	/* We retain handle reference count that would have been acquired by data dependencies.  */
	unsigned i;
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
	for (i=0; i<nbuffers; i++)
	{
		starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, i);
		_starpu_spin_lock(&handle->header_lock);
		handle->busy_count++;
		_starpu_spin_unlock(&handle->header_lock);
	}

	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	_starpu_increment_nsubmitted_tasks_of_sched_ctx(j->task->sched_ctx);
	_starpu_sched_task_submit(task);

	STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
	_starpu_handle_job_submission(j);
	_starpu_increment_nready_tasks_of_sched_ctx(j->task->sched_ctx, j->task->flops, j->task);
	_starpu_job_set_ordered_buffers(j);

	STARPU_ASSERT(task->status == STARPU_TASK_INIT);
	task->status = STARPU_TASK_READY;
	_starpu_profiling_set_task_push_start_time(task);

	unsigned node = starpu_worker_get_memory_node(workerid);
	if (starpu_get_prefetch_flag())
		starpu_prefetch_task_input_on_node(task, node);

	struct _starpu_worker *worker;
	worker = _starpu_get_worker_struct(workerid);
	starpu_task_list_push_back(&worker->local_tasks, task);
	starpu_wake_worker_locked(worker->workerid);

	_starpu_profiling_set_task_push_end_time(task);

	STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
	return 0;
}

void starpu_codelet_init(struct starpu_codelet *cl)
{
	memset(cl, 0, sizeof(struct starpu_codelet));
}

#define _STARPU_CODELET_WORKER_NAME_LEN 32

void starpu_codelet_display_stats(struct starpu_codelet *cl)
{
	unsigned worker;
	unsigned nworkers = starpu_worker_get_count();

	if (cl->name)
		fprintf(stderr, "Statistics for codelet %s\n", cl->name);
	else if (cl->model && cl->model->symbol)
		fprintf(stderr, "Statistics for codelet %s\n", cl->model->symbol);

	unsigned long total = 0;

	for (worker = 0; worker < nworkers; worker++)
		total += cl->per_worker_stats[worker];

	for (worker = 0; worker < nworkers; worker++)
	{
		char name[_STARPU_CODELET_WORKER_NAME_LEN];
		starpu_worker_get_name(worker, name, _STARPU_CODELET_WORKER_NAME_LEN);

		fprintf(stderr, "\t%s -> %lu / %lu (%2.2f %%)\n", name, cl->per_worker_stats[worker], total, (100.0f*cl->per_worker_stats[worker])/total);
	}
}

/*
 * We wait for all the tasks that have already been submitted. Note that a
 * regenerable is not considered finished until it was explicitely set as
 * non-regenerale anymore (eg. from a callback).
 */
int _starpu_task_wait_for_all_and_return_nb_waited_tasks(void)
{
	unsigned nsched_ctxs = _starpu_get_nsched_ctxs();
	unsigned sched_ctx_id = nsched_ctxs == 1 ? 0 : starpu_sched_ctx_get_context();

	/* if there is no indication about which context to wait,
	   we wait for all tasks submitted to starpu */
	if (sched_ctx_id == STARPU_NMAX_SCHED_CTXS)
	{
		_STARPU_DEBUG("Waiting for all tasks\n");
		STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "starpu_task_wait_for_all must not be called from a task or callback");
		STARPU_AYU_BARRIER();
		struct _starpu_machine_config *config = _starpu_get_machine_config();
		if(config->topology.nsched_ctxs == 1)
		{
			_starpu_sched_do_schedule(0);
			return _starpu_task_wait_for_all_in_ctx_and_return_nb_waited_tasks(0);
		}
		else
		{
			int s;
			for(s = 0; s < STARPU_NMAX_SCHED_CTXS; s++)
			{
				if(config->sched_ctxs[s].do_schedule == 1)
				{
					_starpu_sched_do_schedule(config->sched_ctxs[s].id);
				}
			}
			for(s = 0; s < STARPU_NMAX_SCHED_CTXS; s++)
			{
				if(config->sched_ctxs[s].do_schedule == 1)
				{
					starpu_task_wait_for_all_in_ctx(config->sched_ctxs[s].id);
				}
			}
			return 0;
		}
	}
	else
	{
		_starpu_sched_do_schedule(sched_ctx_id);
		_STARPU_DEBUG("Waiting for tasks submitted to context %u\n", sched_ctx_id);
		return _starpu_task_wait_for_all_in_ctx_and_return_nb_waited_tasks(sched_ctx_id);
	}
}

int starpu_task_wait_for_all(void)
{
	_starpu_task_wait_for_all_and_return_nb_waited_tasks();
	return 0;
}

int _starpu_task_wait_for_all_in_ctx_and_return_nb_waited_tasks(unsigned sched_ctx)
{
	_STARPU_TRACE_TASK_WAIT_FOR_ALL_START();
	int ret = _starpu_wait_for_all_tasks_of_sched_ctx(sched_ctx);
	_STARPU_TRACE_TASK_WAIT_FOR_ALL_END();
	/* TODO: improve Temanejo into knowing about contexts ... */
	STARPU_AYU_BARRIER();
	return ret;
}

int starpu_task_wait_for_all_in_ctx(unsigned sched_ctx)
{
	_starpu_task_wait_for_all_in_ctx_and_return_nb_waited_tasks(sched_ctx);
	return 0;
}

/*
 * We wait until there's a certain number of the tasks that have already been
 * submitted left. Note that a regenerable is not considered finished until it
 * was explicitely set as non-regenerale anymore (eg. from a callback).
 */
int starpu_task_wait_for_n_submitted(unsigned n)
{
	unsigned nsched_ctxs = _starpu_get_nsched_ctxs();
	unsigned sched_ctx_id = nsched_ctxs == 1 ? 0 : starpu_sched_ctx_get_context();

	/* if there is no indication about which context to wait,
	   we wait for all tasks submitted to starpu */
	if (sched_ctx_id == STARPU_NMAX_SCHED_CTXS)
	{
		_STARPU_DEBUG("Waiting for all tasks\n");
		STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "starpu_task_wait_for_n_submitted must not be called from a task or callback");

		struct _starpu_machine_config *config = _starpu_get_machine_config();
		if(config->topology.nsched_ctxs == 1)
			_starpu_wait_for_n_submitted_tasks_of_sched_ctx(0, n);
		else
		{
			int s;
			for(s = 0; s < STARPU_NMAX_SCHED_CTXS; s++)
			{
				if(config->sched_ctxs[s].do_schedule == 1)
				{
					_starpu_wait_for_n_submitted_tasks_of_sched_ctx(config->sched_ctxs[s].id, n);
				}
			}
		}

		return 0;
	}
	else
	{
		_STARPU_DEBUG("Waiting for tasks submitted to context %u\n", sched_ctx_id);
		_starpu_wait_for_n_submitted_tasks_of_sched_ctx(sched_ctx_id, n);
	}
	return 0;
}

int starpu_task_wait_for_n_submitted_in_ctx(unsigned sched_ctx, unsigned n)
{
	_starpu_wait_for_n_submitted_tasks_of_sched_ctx(sched_ctx, n);

	return 0;
}
/*
 * We wait until there is no ready task any more (i.e. StarPU will not be able
 * to progress any more).
 */
int starpu_task_wait_for_no_ready(void)
{
	STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "starpu_task_wait_for_no_ready must not be called from a task or callback");

	struct _starpu_machine_config *config = _starpu_get_machine_config();
	if(config->topology.nsched_ctxs == 1)
	{
		_starpu_sched_do_schedule(0);
		_starpu_wait_for_no_ready_of_sched_ctx(0);
	}
	else
	{
		int s;
		for(s = 0; s < STARPU_NMAX_SCHED_CTXS; s++)
		{
			if(config->sched_ctxs[s].do_schedule == 1)
			{
				_starpu_sched_do_schedule(config->sched_ctxs[s].id);
			}
		}
		for(s = 0; s < STARPU_NMAX_SCHED_CTXS; s++)
		{
			if(config->sched_ctxs[s].do_schedule == 1)
			{
				_starpu_wait_for_no_ready_of_sched_ctx(config->sched_ctxs[s].id);
			}
		}
	}

	return 0;
}

void starpu_iteration_push(unsigned long iteration)
{
	struct _starpu_sched_ctx *ctx = _starpu_get_sched_ctx_struct(_starpu_sched_ctx_get_current_context());
	unsigned level = ctx->iteration_level++;
	if (level < sizeof(ctx->iterations)/sizeof(ctx->iterations[0]))
		ctx->iterations[level] = iteration;
}

void starpu_iteration_pop(void)
{
	struct _starpu_sched_ctx *ctx = _starpu_get_sched_ctx_struct(_starpu_sched_ctx_get_current_context());
	STARPU_ASSERT_MSG(ctx->iteration_level > 0, "calls to starpu_iteration_pop must match starpu_iteration_push calls");
	unsigned level = ctx->iteration_level--;
	if (level < sizeof(ctx->iterations)/sizeof(ctx->iterations[0]))
		ctx->iterations[level] = -1;
}

void starpu_do_schedule(void)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	if(config->topology.nsched_ctxs == 1)
		_starpu_sched_do_schedule(0);
	else
	{
		int s;
		for(s = 0; s < STARPU_NMAX_SCHED_CTXS; s++)
		{
			if(config->sched_ctxs[s].do_schedule == 1)
			{
				_starpu_sched_do_schedule(config->sched_ctxs[s].id);
			}
		}
	}
}

void
starpu_drivers_request_termination(void)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();

	STARPU_PTHREAD_MUTEX_LOCK(&config->submitted_mutex);
	int nsubmitted = starpu_task_nsubmitted();
	config->submitting = 0;
	if (nsubmitted == 0)
	{
		ANNOTATE_HAPPENS_AFTER(&config->running);
		config->running = 0;
		ANNOTATE_HAPPENS_BEFORE(&config->running);
		STARPU_WMB();
		int s;
		for(s = 0; s < STARPU_NMAX_SCHED_CTXS; s++)
		{
			if(config->sched_ctxs[s].do_schedule == 1)
			{
				_starpu_check_nsubmitted_tasks_of_sched_ctx(config->sched_ctxs[s].id);
			}
		}
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&config->submitted_mutex);
}

int starpu_task_nsubmitted(void)
{
	int nsubmitted = 0;
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	if(config->topology.nsched_ctxs == 1)
		nsubmitted = _starpu_get_nsubmitted_tasks_of_sched_ctx(0);
	else
	{
		int s;
		for(s = 0; s < STARPU_NMAX_SCHED_CTXS; s++)
		{
			if(config->sched_ctxs[s].do_schedule == 1)
			{
				nsubmitted += _starpu_get_nsubmitted_tasks_of_sched_ctx(config->sched_ctxs[s].id);
			}
		}
	}
	return nsubmitted;
}


int starpu_task_nready(void)
{
	int nready = 0;
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	if(config->topology.nsched_ctxs == 1)
		nready = starpu_sched_ctx_get_nready_tasks(0);
	else
	{
		int s;
		for(s = 0; s < STARPU_NMAX_SCHED_CTXS; s++)
		{
			if(config->sched_ctxs[s].do_schedule == 1)
			{
				nready += starpu_sched_ctx_get_nready_tasks(config->sched_ctxs[s].id);
			}
		}
	}

	return nready;
}

/* Return the task currently executed by the worker, or NULL if this is called
 * either from a thread that is not a task or simply because there is no task
 * being executed at the moment. */
struct starpu_task *starpu_task_get_current(void)
{
	return (struct starpu_task *) STARPU_PTHREAD_GETSPECIFIC(current_task_key);
}

void _starpu_set_current_task(struct starpu_task *task)
{
	STARPU_PTHREAD_SETSPECIFIC(current_task_key, task);
}

int starpu_task_get_current_data_node(unsigned i)
{
	struct starpu_task *task = starpu_task_get_current();
	if (!task)
		return -1;

	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
	struct _starpu_data_descr *descrs = _STARPU_JOB_GET_ORDERED_BUFFERS(j);
	unsigned orderedindex = descrs[i].orderedindex;
	return descrs[orderedindex].node;
}

#ifdef STARPU_OPENMP
/* Prepare the fields of the currentl task for accepting a new set of
 * dependencies in anticipation of becoming a continuation.
 *
 * When the task becomes 'continued', it will only be queued again when the new
 * set of dependencies is fulfilled. */
void _starpu_task_prepare_for_continuation(void)
{
	_starpu_job_prepare_for_continuation(_starpu_get_job_associated_to_task(starpu_task_get_current()));
}

void _starpu_task_prepare_for_continuation_ext(unsigned continuation_resubmit,
		void (*continuation_callback_on_sleep)(void *arg), void *continuation_callback_on_sleep_arg)
{
	_starpu_job_prepare_for_continuation_ext(_starpu_get_job_associated_to_task(starpu_task_get_current()),
		continuation_resubmit, continuation_callback_on_sleep, continuation_callback_on_sleep_arg);
}

void _starpu_task_set_omp_cleanup_callback(struct starpu_task *task, void (*omp_cleanup_callback)(void *arg), void *omp_cleanup_callback_arg)
{
	_starpu_job_set_omp_cleanup_callback(_starpu_get_job_associated_to_task(task),
		omp_cleanup_callback, omp_cleanup_callback_arg);
}
#endif

/*
 * Returns 0 if tasks does not use any multiformat handle, 1 otherwise.
 */
int
_starpu_task_uses_multiformat_handles(struct starpu_task *task)
{
	unsigned i;
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
	for (i = 0; i < nbuffers; i++)
	{
		if (_starpu_data_is_multiformat_handle(STARPU_TASK_GET_HANDLE(task, i)))
			return 1;
	}

	return 0;
}

/*
 * Checks whether the given handle needs to be converted in order to be used on
 * the node given as the second argument.
 */
int
_starpu_handle_needs_conversion_task(starpu_data_handle_t handle,
				     unsigned int node)
{
	return _starpu_handle_needs_conversion_task_for_arch(handle, starpu_node_get_kind(node));
}

int
_starpu_handle_needs_conversion_task_for_arch(starpu_data_handle_t handle,
				     enum starpu_node_kind node_kind)
{
	/*
	 * Here, we assume that CUDA devices and OpenCL devices use the
	 * same data structure. A conversion is only needed when moving
	 * data from a CPU to a GPU, or the other way around.
	 */
	switch (node_kind)
	{
		case STARPU_CPU_RAM:
			switch(starpu_node_get_kind(handle->mf_node))
			{
				case STARPU_CPU_RAM:
					return 0;
				case STARPU_CUDA_RAM:      /* Fall through */
				case STARPU_OPENCL_RAM:
				case STARPU_MIC_RAM:
                                case STARPU_MPI_MS_RAM:
					return 1;
				default:
					STARPU_ABORT();
			}
			break;
		case STARPU_CUDA_RAM:    /* Fall through */
		case STARPU_OPENCL_RAM:
		case STARPU_MIC_RAM:
		case STARPU_MPI_MS_RAM:
			switch(starpu_node_get_kind(handle->mf_node))
			{
				case STARPU_CPU_RAM:
					return 1;
				case STARPU_CUDA_RAM:
				case STARPU_OPENCL_RAM:
				case STARPU_MIC_RAM:
                                case STARPU_MPI_MS_RAM:
					return 0;
				default:
					STARPU_ABORT();
			}
			break;
		default:
			STARPU_ABORT();
	}
	/* that instruction should never be reached */
	return -EINVAL;
}

void starpu_task_set_implementation(struct starpu_task *task, unsigned impl)
{
	_starpu_get_job_associated_to_task(task)->nimpl = impl;
}

unsigned starpu_task_get_implementation(struct starpu_task *task)
{
	return _starpu_get_job_associated_to_task(task)->nimpl;
}

unsigned long starpu_task_get_job_id(struct starpu_task *task)
{
	return _starpu_get_job_associated_to_task(task)->job_id;
}

static starpu_pthread_t watchdog_thread;

static int sleep_some(float timeout)
{
	/* If we do a sleep(timeout), we might have to wait too long at the end of the computation. */
	/* To avoid that, we do several sleep() of 1s (and check after each if starpu is still running) */
	float t;
	for (t = timeout ; t > 1.; t--)
	{
		starpu_sleep(1.);
		if (!_starpu_machine_is_running())
			/* Application finished, don't bother finishing the sleep */
			return 0;
	}
	/* and one final sleep (of less than 1 s) with the rest (if needed) */
	if (t > 0.)
		starpu_sleep(t);
	return 1;
}

/* Check from times to times that StarPU does finish some tasks */
static void *watchdog_func(void *arg)
{
	char *timeout_env = arg;
	float timeout, delay;

#ifdef _MSC_VER
	timeout = ((float) _atoi64(timeout_env)) / 1000000;
#else
	timeout = ((float) atoll(timeout_env)) / 1000000;
#endif
	delay = ((float) watchdog_delay) / 1000000;
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	starpu_pthread_setname("watchdog");

	if (!sleep_some(delay))
		return NULL;

	STARPU_PTHREAD_MUTEX_LOCK(&config->submitted_mutex);
	while (_starpu_machine_is_running())
	{
		int last_nsubmitted = starpu_task_nsubmitted();
		config->watchdog_ok = 0;
		STARPU_PTHREAD_MUTEX_UNLOCK(&config->submitted_mutex);

		if (!sleep_some(timeout))
			return NULL;

		STARPU_PTHREAD_MUTEX_LOCK(&config->submitted_mutex);
		if (!config->watchdog_ok && last_nsubmitted
				&& last_nsubmitted == starpu_task_nsubmitted())
		{
			if (watchdog_hook == NULL)
				_STARPU_MSG("The StarPU watchdog detected that no task finished for %fs (can be configured through STARPU_WATCHDOG_TIMEOUT)\n",
									timeout);
			else
				watchdog_hook(watchdog_hook_arg);

			if (watchdog_crash)
			{
				_STARPU_MSG("Crashing the process\n");
				raise(SIGABRT);
			}
			else if (watchdog_hook == NULL)
				_STARPU_MSG("Set the STARPU_WATCHDOG_CRASH environment variable if you want to abort the process in such a case\n");
		}
		/* Only shout again after another period */
		config->watchdog_ok = 1;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&config->submitted_mutex);
	return NULL;
}

void starpu_task_watchdog_set_hook(void (*hook)(void *), void *hook_arg)
{
	watchdog_hook = hook;
	watchdog_hook_arg = hook_arg;
}

void _starpu_watchdog_init()
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	char *timeout_env = starpu_getenv("STARPU_WATCHDOG_TIMEOUT");

	STARPU_PTHREAD_MUTEX_INIT(&config->submitted_mutex, NULL);

	if (!timeout_env)
		return;

	STARPU_PTHREAD_CREATE(&watchdog_thread, NULL, watchdog_func, timeout_env);
}

void _starpu_watchdog_shutdown(void)
{
	char *timeout_env = starpu_getenv("STARPU_WATCHDOG_TIMEOUT");

	if (!timeout_env)
		return;

	STARPU_PTHREAD_JOIN(watchdog_thread, NULL);
}
