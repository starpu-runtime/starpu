/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <common/knobs.h>
#include <datawizard/memory_nodes.h>
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

/* global counters */
static int __g_total_submitted;
static int __g_peak_submitted;
static int __g_peak_ready;

/* global counter variables */
starpu_perf_counter_int64_t _starpu_task__g_total_submitted__value;
starpu_perf_counter_int64_t _starpu_task__g_peak_submitted__value;
starpu_perf_counter_int64_t _starpu_task__g_current_submitted__value;
starpu_perf_counter_int64_t _starpu_task__g_peak_ready__value;
starpu_perf_counter_int64_t _starpu_task__g_current_ready__value;

/* per-worker counters */
static int __w_total_executed;
static int __w_cumul_execution_time;

/* per-codelet counters */
static int __c_total_submitted;
static int __c_peak_submitted;
static int __c_peak_ready;
static int __c_total_executed;
static int __c_cumul_execution_time;

/* - */

/* per-scheduler knobs */
static int __s_max_priority_cap_knob;
static int __s_min_priority_cap_knob;

/* knob variables */
static int __s_max_priority_cap__value;
static int __s_min_priority_cap__value;

static struct starpu_perf_knob_group * __kg_starpu_task__per_scheduler;

/* - */

static void global_sample_updater(struct starpu_perf_counter_sample *sample, void *context)
{
	STARPU_ASSERT(context == NULL); /* no context for the global updater */
	(void)context;

	_starpu_perf_counter_sample_set_int64_value(sample, __g_total_submitted, _starpu_task__g_total_submitted__value);
	_starpu_perf_counter_sample_set_int64_value(sample, __g_peak_submitted, _starpu_task__g_peak_submitted__value);
	_starpu_perf_counter_sample_set_int64_value(sample, __g_peak_ready, _starpu_task__g_peak_ready__value);
}

static void per_worker_sample_updater(struct starpu_perf_counter_sample *sample, void *context)
{
	STARPU_ASSERT(context != NULL);
	struct _starpu_worker *worker = context;

	_starpu_perf_counter_sample_set_int64_value(sample, __w_total_executed, worker->__w_total_executed__value);
	_starpu_perf_counter_sample_set_double_value(sample, __w_cumul_execution_time, worker->__w_cumul_execution_time__value);
}

static void per_codelet_sample_updater(struct starpu_perf_counter_sample *sample, void *context)
{
	STARPU_ASSERT(sample->listener != NULL && sample->listener->set != NULL);
	struct starpu_perf_counter_set *set = sample->listener->set;
	STARPU_ASSERT(set->scope == starpu_perf_counter_scope_per_codelet);
	STARPU_ASSERT(context != NULL);
	struct starpu_codelet *cl = context;

	_starpu_perf_counter_sample_set_int64_value(sample, __c_total_submitted, cl->perf_counter_values->task.total_submitted);
	_starpu_perf_counter_sample_set_int64_value(sample, __c_peak_submitted, cl->perf_counter_values->task.peak_submitted);
	_starpu_perf_counter_sample_set_int64_value(sample, __c_peak_ready, cl->perf_counter_values->task.peak_ready);
	_starpu_perf_counter_sample_set_int64_value(sample, __c_total_executed, cl->perf_counter_values->task.total_executed);
	_starpu_perf_counter_sample_set_double_value(sample, __c_cumul_execution_time, cl->perf_counter_values->task.cumul_execution_time);
}

void _starpu__task_c__register_counters(void)
{
	{
		const enum starpu_perf_counter_scope scope = starpu_perf_counter_scope_global;
		__STARPU_PERF_COUNTER_REG("starpu.task", scope, g_total_submitted, int64, "number of tasks submitted globally (since StarPU initialization)");
		__STARPU_PERF_COUNTER_REG("starpu.task", scope, g_peak_submitted, int64, "maximum simultaneous number of tasks submitted and not yet ready, globally (since StarPU initialization)");
		__STARPU_PERF_COUNTER_REG("starpu.task", scope, g_peak_ready, int64, "maximum simultaneous number of tasks ready and not yet executing, globally (since StarPU initialization)");

		_starpu_perf_counter_register_updater(scope, global_sample_updater);
	}

	{
		const enum starpu_perf_counter_scope scope = starpu_perf_counter_scope_per_worker;
		__STARPU_PERF_COUNTER_REG("starpu.task", scope, w_total_executed, int64, "number of tasks executed on this worker (since StarPU initialization)");
		__STARPU_PERF_COUNTER_REG("starpu.task", scope, w_cumul_execution_time, double, "cumulated execution time of tasks executed on this worker (microseconds, since StarPU initialization)");

		_starpu_perf_counter_register_updater(scope, per_worker_sample_updater);
	}

	{
		const enum starpu_perf_counter_scope scope = starpu_perf_counter_scope_per_codelet;
		__STARPU_PERF_COUNTER_REG("starpu.task", scope, c_total_submitted, int64, "number of codelet's task instances submitted using this codelet (since enabled)");
		__STARPU_PERF_COUNTER_REG("starpu.task", scope, c_peak_submitted, int64, "maximum simultaneous number of codelet's task instances submitted and not yet ready (since enabled)");
		__STARPU_PERF_COUNTER_REG("starpu.task", scope, c_peak_ready, int64, "maximum simultaneous number of codelet's task instances ready and not yet executing (since enabled)");
		__STARPU_PERF_COUNTER_REG("starpu.task", scope, c_total_executed, int64, "number of codelet's task instances executed using this codelet (since enabled)");
		__STARPU_PERF_COUNTER_REG("starpu.task", scope, c_cumul_execution_time, double, "cumulated execution time of codelet's task instances (since enabled)");

		_starpu_perf_counter_register_updater(scope, per_codelet_sample_updater);
	}
}

/* - */

static void sched_knobs__set(const struct starpu_perf_knob * const knob, void *context, const struct starpu_perf_knob_value * const value)
{
	const char * const sched_policy_name = *(const char **)context;
	(void) sched_policy_name;
	if (knob->id == __s_max_priority_cap_knob)
	{
		STARPU_ASSERT(value->val_int32_t <= STARPU_MAX_PRIO);
		STARPU_ASSERT(value->val_int32_t >= STARPU_MIN_PRIO);
		STARPU_ASSERT(value->val_int32_t >= __s_min_priority_cap__value);
		__s_max_priority_cap__value = value->val_int32_t;
	}
	else if (knob->id == __s_min_priority_cap_knob)
	{
		STARPU_ASSERT(value->val_int32_t <= STARPU_MAX_PRIO);
		STARPU_ASSERT(value->val_int32_t >= STARPU_MIN_PRIO);
		STARPU_ASSERT(value->val_int32_t <= __s_max_priority_cap__value);
		__s_min_priority_cap__value = value->val_int32_t;
	}
	else
	{
		STARPU_ASSERT(0);
		abort();
	}
}

static void sched_knobs__get(const struct starpu_perf_knob * const knob, void *context,       struct starpu_perf_knob_value * const value)
{
	const char * const sched_policy_name = *(const char **)context;
	(void) sched_policy_name;
	if (knob->id == __s_max_priority_cap_knob)
	{
		value->val_int32_t = __s_max_priority_cap__value;
	}
	else if (knob->id == __s_min_priority_cap_knob)
	{
		value->val_int32_t = __s_min_priority_cap__value;
	}
	else
	{
		STARPU_ASSERT(0);
		abort();
	}
}

void _starpu__task_c__register_knobs(void)
{
#if 0
	{
		const enum starpu_perf_knob_scope scope = starpu_perf_knob_scope_global;
		__kg_starpu_global = _starpu_perf_knob_group_register(scope, global_knobs__set, global_knobs__get);
	}
#endif

#if 0
	{
		const enum starpu_perf_knob_scope scope = starpu_perf_knob_scope_per_worker;
		__kg_starpu_worker__per_worker = _starpu_perf_knob_group_register(scope, worker_knobs__set, worker_knobs__get);
	}
#endif

	{
		const enum starpu_perf_knob_scope scope = starpu_perf_knob_scope_per_scheduler;
		__kg_starpu_task__per_scheduler = _starpu_perf_knob_group_register(scope, sched_knobs__set, sched_knobs__get);

		/* TODO: priority capping knobs actually work globally for now, the sched policy name is ignored */
		__STARPU_PERF_KNOB_REG("starpu.task", __kg_starpu_task__per_scheduler, s_max_priority_cap_knob, int32, "force task priority to this value or below (priority value)");
		__s_max_priority_cap__value = STARPU_MAX_PRIO;

		__STARPU_PERF_KNOB_REG("starpu.task", __kg_starpu_task__per_scheduler, s_min_priority_cap_knob, int32, "force task priority to this value or above (priority value)");
		__s_min_priority_cap__value = STARPU_MIN_PRIO;
	}
}

void _starpu__task_c__unregister_knobs(void)
{
	_starpu_perf_knob_group_unregister(__kg_starpu_task__per_scheduler);
	__kg_starpu_task__per_scheduler = NULL;
}

/* - */

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
	limit_min_submitted_tasks = starpu_getenv_number("STARPU_LIMIT_MIN_SUBMITTED_TASKS");
	limit_max_submitted_tasks = starpu_getenv_number("STARPU_LIMIT_MAX_SUBMITTED_TASKS");
	watchdog_crash = starpu_getenv_number_default("STARPU_WATCHDOG_CRASH", 0);
	watchdog_delay = starpu_getenv_number_default("STARPU_WATCHDOG_DELAY", 0);
}

void _starpu_task_deinit(void)
{
	STARPU_PTHREAD_KEY_DELETE(current_task_key);
}

void starpu_set_limit_min_submitted_tasks(int limit_min)
{
	limit_min_submitted_tasks = limit_min;
}

void starpu_set_limit_max_submitted_tasks(int limit_max)
{
	limit_max_submitted_tasks = limit_max;
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


static struct starpu_codelet _starpu_data_sync_cl =
{
	.where = STARPU_NOWHERE,
	.nbuffers = STARPU_VARIABLE_NBUFFERS
};

struct starpu_task * STARPU_ATTRIBUTE_MALLOC starpu_task_create_sync(starpu_data_handle_t handle, enum starpu_data_access_mode mode)
{
	struct starpu_task *task = starpu_task_create();
	task->cl = &_starpu_data_sync_cl;
	STARPU_TASK_SET_HANDLE(task, handle, 0);
	STARPU_TASK_SET_MODE(task, mode, 0);
	task->nbuffers = 1;
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
	    && _starpu_get_local_worker_status() & STATUS_CALLBACK)
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

		/* Does user want StarPU release cl_ret ? */
		if (task->cl_ret_free)
			free(task->cl_ret);

		/* Does user want StarPU release callback_arg ? */
		if (task->callback_arg_free)
			free(task->callback_arg);

		/* Does user want StarPU release epilogue callback_arg ? */
		if (task->epilogue_callback_arg_free)
			free(task->epilogue_callback_arg);

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

void starpu_task_set_destroy(struct starpu_task *task)
{
	STARPU_ASSERT(task);
	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
	STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
	STARPU_ASSERT_MSG(!task->destroy, "starpu_task_set_destroy must not be called for task with destroy = 1");
	if (j->terminated == 2)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
		/* It's already over, _starpu_handle_job_termination will not
		 * destroy it, do it ourself */
		_starpu_task_destroy(task);
	}
	else
	{
		/* Let _starpu_handle_job_termination destroy it */
		task->destroy = 1;
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
	}
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

	_starpu_perf_counter_update_global_sample();
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
int _starpu_submit_job(struct _starpu_job *j, int nodeps)
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

	if (nodeps)
	{
		ret = _starpu_take_deps_and_schedule(j);
	}
	else
	{
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

#if defined(STARPU_USE_CPU) || defined(STARPU_SIMGRID)
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
#endif

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
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
#endif

#if defined(STARPU_USE_HIP)
	some_impl = 0;
	for (i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
		if (cl->hip_funcs[i])
		{
			some_impl = 1;
			break;
		}
	if (some_impl && is_where_unset)
	{
		where |= STARPU_HIP;
	}
#endif

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
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
#endif

#ifdef STARPU_USE_MAX_FPGA
	/* FPGA */
	some_impl = 0;
	for (i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
		if (cl->max_fpga_funcs[i])
		{
			some_impl = 1;
			break;
		}
	if (some_impl && is_where_unset)
	{
		where |= STARPU_MAX_FPGA;
	}
#endif

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	some_impl = 0;
	for (i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
		if (cl->cpu_funcs_name[i])
		{
			some_impl = 1;
			break;
		}
	if (some_impl && is_where_unset)
	{
		where |= STARPU_MPI_MS;
	}
#endif

#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
	some_impl = 0;
	for (i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
		if (cl->cpu_funcs_name[i])
		{
			some_impl = 1;
			break;
		}
	if (some_impl && is_where_unset)
	{
		where |= STARPU_TCPIP_MS;
	}
#endif

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

#ifdef STARPU_BUBBLE
	if ((j->task->bubble_func && j->task->bubble_func(j->task, j->task->bubble_func_arg)) || (j->task->cl && j->task->cl->bubble_func && j->task->cl->bubble_func(j->task, j->task->bubble_func_arg)))
		j->is_bubble = 1;
	else
		j->is_bubble = 0;
#endif

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

#ifdef STARPU_DEBUG
	if (task->workerids)
	{
		unsigned i;
		for (i = 0; i < task->workerids_len; i++)
			if (task->workerids[i] != 0)
				break;
		STARPU_ASSERT_MSG(i < task->workerids_len, "The workerids array can't contain only zeros, it would not be executable at all.");
	}
#endif

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

		if (STARPU_UNLIKELY(task->dyn_handles))
		{
			_STARPU_MALLOC(task->dyn_interfaces, nbuffers * sizeof(void *));
		}

		struct _starpu_data_descr *descrs = _STARPU_JOB_GET_ORDERED_BUFFERS(j);
		for (i = 0; i < nbuffers; i++)
		{
			starpu_data_handle_t handle = descrs[i].handle;
			enum starpu_data_access_mode mode = descrs[i].mode;

			int node = task->cl->specific_nodes ? STARPU_CODELET_GET_NODE(task->cl, i) : -1;
			/* Make sure handles are valid */
			STARPU_ASSERT_MSG(handle->magic == _STARPU_TASK_MAGIC, "data %p is invalid (was it already unregistered?)", handle);
			/* Make sure handles are not partitioned */
			STARPU_ASSERT_MSG(handle->nchildren == 0, "only unpartitioned data (or the pieces of a partitioned data) can be used in a task");
			/* Make sure the specified node exists */
			STARPU_ASSERT_MSG(node == STARPU_SPECIFIC_NODE_LOCAL || node == STARPU_SPECIFIC_NODE_CPU || node == STARPU_SPECIFIC_NODE_SLOW || node == STARPU_SPECIFIC_NODE_LOCAL_OR_CPU || node == STARPU_SPECIFIC_NODE_NONE || (node >= 0 && node < (int) starpu_memory_nodes_get_count()), "The codelet-specified memory node does not exist");
			/* Provide the home interface for now if any,
			 * for can_execute hooks */
			if (handle->home_node != -1)
				_STARPU_TASK_SET_INTERFACE(task, starpu_data_get_interface_on_node(handle, handle->home_node), i);

			if (!(task->cl->flags & STARPU_CODELET_NOPLANS) &&
			    ((handle->nplans && !handle->nchildren) || handle->siblings)
#ifdef STARPU_BUBBLE
			    && !j->is_bubble
			    /*
			     * => require to set the is_bubble a soon as possible and not in the turn_task_into_bubble.
			     */
#endif
			    && !(mode & STARPU_NOPLAN))
				/* This handle is involved with asynchronous
				 * partitioning as a parent or a child, make
				 * sure the right plan is active, submit
				 * appropiate partitioning / unpartitioning if
				 * not */
				_starpu_data_partition_access_submit(handle, (mode & STARPU_W) != 0);
		}

		/* Check the type of worker(s) required by the task exist */
		if (STARPU_UNLIKELY(!_starpu_worker_exists(task)))
		{
			_STARPU_LOG_OUT_TAG("ENODEV");
			return -ENODEV;
		}

		/* In case we require that a task should be explicitely
		 * executed on a specific worker, we make sure that the worker
		 * is able to execute this task.  */
		if (STARPU_UNLIKELY(task->execute_on_a_specific_worker && !starpu_combined_worker_can_execute_task(task->workerid, task, 0)))
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
int _starpu_task_submit(struct starpu_task *task, int nodeps)
{
	_STARPU_LOG_IN();
	STARPU_ASSERT(task);
	STARPU_ASSERT_MSG(task->magic == _STARPU_TASK_MAGIC, "Tasks must be created with starpu_task_create, or initialized with starpu_task_init.");
	STARPU_ASSERT_MSG(starpu_is_initialized(), "starpu_init must be called (and return no error) before submitting tasks.");

	int ret;
	{
		/* task knobs */
		if (task->priority > __s_max_priority_cap__value)
			task->priority = __s_max_priority_cap__value;
		if (task->priority < __s_min_priority_cap__value)
			task->priority = __s_min_priority_cap__value;
	}

	if (task->transaction != NULL)
	{
		/* If task is part of a transaction, add its handle to the task
		 * handle list with a STARPU_R access mode to allow concurrency among the epoch
		 * tasks while serializing it with epoch and transactions operations */
		STARPU_ASSERT(task->cl->nbuffers == STARPU_VARIABLE_NBUFFERS);
		STARPU_ASSERT(!_starpu_trs_epoch_list_empty(&task->transaction->epoch_list));
		task->trs_epoch = _starpu_trs_epoch_list_back(&task->transaction->epoch_list);
		int nbuffers = task->nbuffers;
		int allocated_nbuffers = (task->dyn_handles != NULL)?nbuffers:0;
		task->nbuffers++;
		starpu_task_insert_data_process_arg(task->cl, task, &allocated_nbuffers, &nbuffers, STARPU_R, task->transaction->handle);
	}

	unsigned is_sync = task->synchronous;
	starpu_task_bundle_t bundle = task->bundle;
	STARPU_ASSERT_MSG(!(nodeps && bundle), "not supported\n");
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
	if (!_starpu_perf_counter_paused() && !j->internal && !continuation)
	{
		(void) STARPU_PERF_COUNTER_ADD64(&_starpu_task__g_total_submitted__value, 1);
		int64_t value = STARPU_PERF_COUNTER_ADD64(&_starpu_task__g_current_submitted__value, 1);
		_starpu_perf_counter_update_max_int64(&_starpu_task__g_peak_submitted__value, value);
		_starpu_perf_counter_update_global_sample();

		if (task->cl && task->cl->perf_counter_values)
		{
			struct starpu_perf_counter_sample_cl_values * const pcv = task->cl->perf_counter_values;

			(void) STARPU_PERF_COUNTER_ADD64(&pcv->task.total_submitted, 1);
			value = STARPU_PERF_COUNTER_ADD64(&pcv->task.current_submitted, 1);
			_starpu_perf_counter_update_max_int64(&pcv->task.peak_submitted, value);
			_starpu_perf_counter_update_per_codelet_sample(task->cl);
		}
	}
	STARPU_ASSERT_MSG(!(nodeps && continuation), "not supported\n");

	if (!j->internal && limit_max_submitted_tasks >= 0 && limit_min_submitted_tasks >= 0)
	{
		int nsubmitted_tasks = starpu_task_nsubmitted();
		if (limit_max_submitted_tasks < nsubmitted_tasks
			&& limit_min_submitted_tasks < nsubmitted_tasks)
		{
			starpu_do_schedule();
			_STARPU_TRACE_TASK_THROTTLE_START();
			starpu_task_wait_for_n_submitted(limit_min_submitted_tasks);
			_STARPU_TRACE_TASK_THROTTLE_END();
		}
	}

	_STARPU_TRACE_TASK_SUBMIT_START();

	if (task->cl && !continuation)
	{
		_starpu_job_set_ordered_buffers(j);
	}

	ret = _starpu_task_submit_head(task);
	if (ret)
	{
		_STARPU_TRACE_TASK_SUBMIT_END();
		return ret;
	}

	if (!continuation)
	{
#ifndef STARPU_NO_ASSERT
		STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
		STARPU_ASSERT_MSG(!j->submitted || j->terminated >= 1, "Tasks can not be submitted a second time before being terminated. Please use different task structures, or use the regenerate flag to let the task resubmit itself automatically.");
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
#endif
		_STARPU_TRACE_TASK_SUBMIT(j,
			_starpu_get_sched_ctx_struct(task->sched_ctx)->iterations[0],
			_starpu_get_sched_ctx_struct(task->sched_ctx)->iterations[1]);
	}

	/* If this is a continuation, we don't modify the implicit data dependencies detected earlier. */
	if (task->cl && !continuation && !nodeps
#ifdef STARPU_BUBBLE
	    && !j->is_bubble
#endif
		)
	{
	    _starpu_detect_implicit_data_deps(task);
	}

	if (STARPU_UNLIKELY(bundle))
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
	struct starpu_profiling_task_info *info = task->profiling_info;
	int profiling = starpu_profiling_status_get();
	if (!info)
	{
		info = _starpu_allocate_profiling_info_if_needed(task);
		task->profiling_info = info;
	}

	/* The task is considered as block until we are sure there remains not
	 * dependency. */
	task->status = STARPU_TASK_BLOCKED;

	if (STARPU_UNLIKELY(profiling))
		_starpu_clock_gettime(&info->submit_time);

	ret = _starpu_submit_job(j, nodeps);
#ifdef STARPU_SIMGRID
	if (_starpu_simgrid_task_submit_cost())
		starpu_sleep(0.000001);
#endif

	if (is_sync)
	{
		if (starpu_is_paused())
		{
			static int warned;
			if (!warned)
			{
				warned = 1;
				_STARPU_DISP("[warning]: A task with synchronous=1 was submitted after calling starpu_pause(). We will thus hang until starpu_resume() gets called.\n");
			}
		}
		_starpu_sched_do_schedule(task->sched_ctx);
		_starpu_wait_job(j);
		if (task->destroy)
		     _starpu_task_destroy(task);
	}

	_STARPU_TRACE_TASK_SUBMIT_END();
	_STARPU_LOG_OUT();
	return ret;
}

#undef starpu_task_submit
int starpu_task_submit(struct starpu_task *task)
{
#ifdef STARPU_BUBBLE_VERBOSE
	struct timespec tp;
	clock_gettime(CLOCK_MONOTONIC, &tp);
	unsigned long long timestamp = 1000000000ULL*tp.tv_sec + tp.tv_nsec;
	_STARPU_DEBUG("{%llu} [%s(%p)] Submission | id %lu\n", timestamp, starpu_task_get_name(task), task, starpu_task_get_job_id(task));
#endif
	return _starpu_task_submit(task, 0);
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
int starpu_task_submit_nodeps(struct starpu_task *task)
{
	return _starpu_task_submit(task, 1);
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

	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	_starpu_job_set_ordered_buffers(j);

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
	starpu_task_prio_list_push_back(&worker->local_tasks, task);
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
 * We wait for all tasks that have been submitted to the scheduling context and its nested contexts
 */
void _starpu_do_schedule_in_nested_ctx(unsigned sched_ctx_id)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	unsigned s;
	for(s = 0; s < STARPU_NMAX_SCHED_CTXS; s++)
	{
		if(config->sched_ctxs[s].id != STARPU_NMAX_SCHED_CTXS && config->sched_ctxs[s].do_schedule == 1 && config->sched_ctxs[s].nesting_sched_ctx == sched_ctx_id && s != sched_ctx_id)
		{
			_starpu_do_schedule_in_nested_ctx(s);
		}
	}
	_starpu_sched_do_schedule(sched_ctx_id);
}

int _starpu_task_wait_for_all_in_nested_ctx_and_return_nb_waited_tasks(unsigned sched_ctx_id)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	unsigned nb_waited_tasks = 0;
	unsigned s;

	for(s = 0; s < STARPU_NMAX_SCHED_CTXS; s++)
	{
		if(config->sched_ctxs[s].id != STARPU_NMAX_SCHED_CTXS && config->sched_ctxs[s].nesting_sched_ctx == sched_ctx_id && s != sched_ctx_id)
		{
			_STARPU_DEBUG("Recursively waiting for tasks submitted to sub context %u of %u\n", s, sched_ctx_id);
			nb_waited_tasks += _starpu_task_wait_for_all_in_nested_ctx_and_return_nb_waited_tasks(s);
		}
	}

	nb_waited_tasks += _starpu_task_wait_for_all_in_ctx_and_return_nb_waited_tasks(sched_ctx_id);
	return nb_waited_tasks;
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
		//		_starpu_sched_do_schedule(sched_ctx_id);
		//		_STARPU_DEBUG("Waiting for tasks submitted to context %u\n", sched_ctx_id);
		//		return _starpu_task_wait_for_all_in_ctx_and_return_nb_waited_tasks(sched_ctx_id);
		_starpu_do_schedule_in_nested_ctx(sched_ctx_id);
		_STARPU_DEBUG("Waiting for tasks submitted to context %u\n", sched_ctx_id);
		return _starpu_task_wait_for_all_in_nested_ctx_and_return_nb_waited_tasks(sched_ctx_id);
	}
}

int starpu_task_wait_for_all(void)
{
	_starpu_task_wait_for_all_and_return_nb_waited_tasks();
	if (!_starpu_perf_counter_paused())
		_starpu_perf_counter_update_global_sample();
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
	if (!_starpu_perf_counter_paused())
		_starpu_perf_counter_update_global_sample();
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

	}
	else
	{
		_STARPU_DEBUG("Waiting for tasks submitted to context %u\n", sched_ctx_id);
		_starpu_wait_for_n_submitted_tasks_of_sched_ctx(sched_ctx_id, n);
	}
	if (!_starpu_perf_counter_paused())
		_starpu_perf_counter_update_global_sample();
	return 0;
}

int starpu_task_wait_for_n_submitted_in_ctx(unsigned sched_ctx, unsigned n)
{
	_starpu_wait_for_n_submitted_tasks_of_sched_ctx(sched_ctx, n);

	if (!_starpu_perf_counter_paused())
		_starpu_perf_counter_update_global_sample();
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

	if (!_starpu_perf_counter_paused())
		_starpu_perf_counter_update_global_sample();
	return 0;
}

void starpu_iteration_push(unsigned long iteration)
{
	unsigned id = _starpu_sched_ctx_get_current_context();
	STARPU_ASSERT(id <= STARPU_NMAX_SCHED_CTXS);
	struct _starpu_sched_ctx *ctx = _starpu_get_sched_ctx_struct(id);
	unsigned level = ctx->iteration_level++;
	if (level < sizeof(ctx->iterations)/sizeof(ctx->iterations[0]))
		ctx->iterations[level] = iteration;
}

void starpu_iteration_pop(void)
{
	unsigned id = _starpu_sched_ctx_get_current_context();
	STARPU_ASSERT(id <= STARPU_NMAX_SCHED_CTXS);
	struct _starpu_sched_ctx *ctx = _starpu_get_sched_ctx_struct(id);
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

struct starpu_task *starpu_worker_get_current_task(unsigned workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	if (worker->pipeline_length)
		return worker->current_tasks[worker->first_task];
	else
		return worker->current_task;
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
		case STARPU_MPI_MS_RAM:
		case STARPU_TCPIP_MS_RAM:
			switch(starpu_node_get_kind(handle->mf_node))
			{
				case STARPU_CPU_RAM:
				case STARPU_MPI_MS_RAM:
				case STARPU_TCPIP_MS_RAM:
					return 0;
				default:
					return 1;
			}
			break;
		default:
			switch(starpu_node_get_kind(handle->mf_node))
			{
				case STARPU_CPU_RAM:
				case STARPU_MPI_MS_RAM:
				case STARPU_TCPIP_MS_RAM:
					return 1;
				default:
					return 0;
			}
			break;
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

	_starpu_crash_call_hooks();
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

/* Transaction clean up callback called when the transaction trs_end
 * task completes. */
static void _starpu_transaction_callback(void *_p_trs)
{
	struct starpu_transaction *p_trs = _p_trs;

	_starpu_spin_destroy(&p_trs->lock);
	starpu_data_unregister_submit(p_trs->handle);
	starpu_free(p_trs);
}

/* Task function for the trs_begin and trs_begin_no_sync codelets. */
static void _starpu_transaction_begin(void *buffers[], void *cl_args)
{
	struct starpu_transaction *p_trs = cl_args;
	STARPU_ASSERT(p_trs->state == _starpu_trs_initialized);
	_starpu_spin_lock(&p_trs->lock);
	STARPU_ASSERT(!_starpu_trs_epoch_list_empty(&p_trs->epoch_list));
	struct _starpu_trs_epoch *p_epoch = _starpu_trs_epoch_list_front(&p_trs->epoch_list);
	STARPU_ASSERT(p_epoch->state == _starpu_trs_epoch_inactive);
	_starpu_spin_unlock(&p_trs->lock);

	int epoch_confirmed = 1;

	/* If the transaction has a user 'do_start_func', we call it to
	 * decide whether the new epoch is confirmed or cancelled. */
	if (p_trs->do_start_func != NULL)
	{
		void * sync_buf = p_epoch->do_sync ? buffers[1] : NULL;
		epoch_confirmed = p_trs->do_start_func(sync_buf, p_epoch->do_start_arg);
	}

	if (epoch_confirmed)
	{
		p_epoch->state = _starpu_trs_epoch_confirmed;
	}
	else
	{
		p_epoch->state = _starpu_trs_epoch_cancelled;
	}
	STARPU_WMB();
}

/* Task function for the trs_end codelet, in charge of cleaning the last epoch. */
static void _starpu_transaction_end(void *buffers[], void *cl_args)
{
	(void)buffers;
	struct starpu_transaction *p_trs = cl_args;
	_starpu_spin_lock(&p_trs->lock);
	STARPU_ASSERT(!_starpu_trs_epoch_list_empty(&p_trs->epoch_list));
	struct _starpu_trs_epoch *p_epoch = _starpu_trs_epoch_list_pop_front(&p_trs->epoch_list);
	STARPU_ASSERT(p_epoch->state == _starpu_trs_epoch_confirmed
			|| p_epoch->state == _starpu_trs_epoch_cancelled);
	_starpu_spin_unlock(&p_trs->lock);

	p_epoch->state = _starpu_trs_epoch_terminated;

	_starpu_trs_epoch_delete(p_epoch);
	p_epoch = NULL;

	/* TODO: transition to end */
	STARPU_ASSERT(_starpu_trs_epoch_list_empty(&p_trs->epoch_list));
}

/* Task function for the trs_next_epoch codelet, in charge of transitioning from a
 * an epoch to the next. */
static void _starpu_transaction_next_epoch(void *buffers[], void *cl_args)
{
	struct starpu_transaction *p_trs = cl_args;
	_starpu_spin_lock(&p_trs->lock);
	STARPU_ASSERT(!_starpu_trs_epoch_list_empty(&p_trs->epoch_list));
	struct _starpu_trs_epoch *p_previous_epoch = _starpu_trs_epoch_list_pop_front(&p_trs->epoch_list);
	STARPU_ASSERT((p_previous_epoch->state == _starpu_trs_epoch_confirmed)
			|| (p_previous_epoch->state == _starpu_trs_epoch_cancelled));
	STARPU_ASSERT(!_starpu_trs_epoch_list_empty(&p_trs->epoch_list));
	struct _starpu_trs_epoch *p_next_epoch = _starpu_trs_epoch_list_front(&p_trs->epoch_list);
	STARPU_ASSERT(p_next_epoch->state == _starpu_trs_epoch_inactive);
	_starpu_spin_unlock(&p_trs->lock);

	p_previous_epoch->state = _starpu_trs_epoch_terminated;
	_starpu_trs_epoch_delete(p_previous_epoch);

	/* TODO: transition to next epoch */

	int epoch_confirmed = 1;

	if (p_trs->do_start_func != NULL)
	{
		void * sync_buf = p_next_epoch->do_sync ? buffers[1] : NULL;
		epoch_confirmed = p_trs->do_start_func(sync_buf, p_next_epoch->do_start_arg);
	}

	if (epoch_confirmed)
	{
		p_next_epoch->state = _starpu_trs_epoch_confirmed;
	}
	else
	{
		p_next_epoch->state = _starpu_trs_epoch_cancelled;
	}
	STARPU_WMB();
}

/* Transaction begin codelet, without implicit sync on a previously
 * accessed data. */
struct starpu_codelet _starpu_codelet_trs_begin_no_sync =
{
	.cpu_funcs = {_starpu_transaction_begin},
	.modes = {STARPU_W},
	.nbuffers = 1,
	.model = &starpu_perfmodel_nop,
	.name = "starpu_transaction_begin_no_sync"
};

/* Transaction begin codelet, with an implicit sync on a previously
 * accessed data. */
struct starpu_codelet _starpu_codelet_trs_begin =
{
	.cpu_funcs = {_starpu_transaction_begin},
	.modes = {STARPU_W, STARPU_RW},
	.nbuffers = 2,
	.model = &starpu_perfmodel_nop,
	.name = "starpu_transaction_begin"
};

/* Transaction end codelet. */
struct starpu_codelet _starpu_codelet_trs_end =
{
	.cpu_funcs = {_starpu_transaction_end},
	.modes = {STARPU_RW},
	.nbuffers = 1,
	.model = &starpu_perfmodel_nop,
	.name = "starpu_transaction_end"
};

/* Epoch transition codelet. */
struct starpu_codelet _starpu_codelet_trs_next_epoch =
{
	.cpu_funcs = {_starpu_transaction_next_epoch},
	.modes = {STARPU_RW},
	.nbuffers = 1,
	.model = &starpu_perfmodel_nop,
	.name = "starpu_transaction_next_epoch"
};

/* Main entry point for creating and activating a transaction object.
 *
 * . do_start_func: a boolean function to decide whether each new epoch start should
 * be confirmed or not.
 * . do_start_sync_handle: a starpu data handle on which the transaction
 * start should depend on, or NULL if no sync is required. The handle is
 * passed to do_start_func()
 * . do_start_arg: an argument passed to do_start_func().*/
static struct starpu_transaction *_do_starpu_transaction_open(int(*do_start_func)(void *buffer, void *arg), starpu_data_handle_t do_start_sync_handle, void *do_start_arg)
{
	struct starpu_transaction *p_trs = NULL;
	int ret = starpu_malloc((void **)&p_trs, sizeof(*p_trs));
	STARPU_ASSERT(ret == 0);
	_starpu_spin_init(&p_trs->lock);
	_starpu_trs_epoch_list_init(&p_trs->epoch_list);

	p_trs->do_start_func = do_start_func;

	p_trs->dummy_data = 0;
	starpu_variable_data_register(&p_trs->handle, STARPU_MAIN_RAM, (uintptr_t)&p_trs->dummy_data, sizeof(p_trs->dummy_data));

	struct _starpu_trs_epoch *p_epoch = _starpu_trs_epoch_new();

	struct starpu_task *task = starpu_task_create();
	task->callback_func = NULL;
	task->cl_arg = p_trs;
	task->handles[0] = p_trs->handle;
	if (do_start_sync_handle != NULL)
	{
		p_epoch->do_sync = 1;
		task->cl = &_starpu_codelet_trs_begin;
		task->handles[1] = do_start_sync_handle;
	}
	else
	{
		p_epoch->do_sync = 0;
		task->cl = &_starpu_codelet_trs_begin_no_sync;
	}
	p_epoch->is_begin = 1;
	p_epoch->state = _starpu_trs_epoch_inactive;
	p_epoch->do_start_arg = do_start_arg;
	_starpu_trs_epoch_list_push_back(&p_trs->epoch_list, p_epoch);
	p_trs->state = _starpu_trs_initialized;

	ret = starpu_task_submit(task);
	if (ret == -ENODEV)
	{
		starpu_data_unregister(p_trs->handle);
		starpu_free(p_trs);
		return NULL;
	}
	STARPU_ASSERT(ret == 0);
	return p_trs;
}

struct starpu_transaction *starpu_transaction_open(int(*do_start_func)(void *buffer, void *arg), void *do_start_arg)
{
	return _do_starpu_transaction_open(do_start_func, NULL, do_start_arg);
}

void starpu_transaction_close(struct starpu_transaction *p_trs)
{
	STARPU_ASSERT(p_trs->state == _starpu_trs_initialized);
	struct starpu_task *task = starpu_task_create();
	task->cl = &_starpu_codelet_trs_end;
	task->callback_func = _starpu_transaction_callback;
	task->callback_arg = p_trs;
	task->handles[0] = p_trs->handle;
	task->cl_arg = p_trs;

	_starpu_spin_lock(&p_trs->lock);
	STARPU_ASSERT(!_starpu_trs_epoch_list_empty(&p_trs->epoch_list));
	struct _starpu_trs_epoch *p_epoch = _starpu_trs_epoch_list_back(&p_trs->epoch_list);
	_starpu_spin_unlock(&p_trs->lock);
	p_epoch->is_end = 1;

	int ret = starpu_task_submit(task);
	STARPU_ASSERT(ret == 0);
}

void starpu_transaction_next_epoch(struct starpu_transaction *p_trs, void *do_start_arg)
{
	STARPU_ASSERT(p_trs->state == _starpu_trs_initialized);
	struct _starpu_trs_epoch *p_epoch = _starpu_trs_epoch_new();
	struct starpu_task *task = starpu_task_create();
	task->cl = &_starpu_codelet_trs_next_epoch;
	task->handles[0] = p_trs->handle;
	task->cl_arg = p_trs;
	p_epoch->do_sync = 0;
	p_epoch->do_start_arg = do_start_arg;
	p_epoch->state = _starpu_trs_epoch_inactive;

	_starpu_spin_lock(&p_trs->lock);
	_starpu_trs_epoch_list_push_back(&p_trs->epoch_list, p_epoch);
	_starpu_spin_unlock(&p_trs->lock);
	int ret = starpu_task_submit(task);
	STARPU_ASSERT(ret == 0);
}

static void _starpu_ft_check_support(const struct starpu_task *task)
{
	unsigned nbuffers = STARPU_TASK_GET_NBUFFERS(task);
	unsigned i;

	for (i = 0; i < nbuffers; i++)
	{
		enum starpu_data_access_mode mode = STARPU_TASK_GET_MODE(task, i);
		STARPU_ASSERT_MSG (mode == STARPU_R || mode == STARPU_W,
				"starpu_task_failed is only supported for tasks with access modes STARPU_R and STARPU_W");
	}
}

struct starpu_task *starpu_task_ft_create_retry
(const struct starpu_task *meta_task, const struct starpu_task *template_task, void (*check_ft)(void *))
{
	/* Create a new task to actually perform the result */
	struct starpu_task *new_task = starpu_task_create();

	*new_task = *template_task;
	new_task->prologue_callback_func = NULL;
	/* XXX: cl_arg needs to be duplicated */
	STARPU_ASSERT_MSG(!meta_task->cl_arg_free || !meta_task->cl_arg, "not supported yet");
	STARPU_ASSERT_MSG(!meta_task->callback_func, "not supported");
	new_task->callback_func = check_ft;
	new_task->callback_arg = (void*) meta_task;
	new_task->callback_arg_free = 0;
	new_task->prologue_callback_arg_free = 0;
	STARPU_ASSERT_MSG(!new_task->prologue_callback_pop_arg_free, "not supported");
	new_task->use_tag = 0;
	new_task->synchronous = 0;
	new_task->destroy = 1;
	new_task->regenerate = 0;
	new_task->no_submitorder = 1;
	new_task->failed = 0;
	new_task->scheduled = 0;
	new_task->prefetched = 0;
	new_task->status = STARPU_TASK_INIT;
	new_task->profiling_info = NULL;
	new_task->prev = NULL;
	new_task->next = NULL;
	new_task->starpu_private = NULL;
	new_task->omp_task = NULL;

	return new_task;
}

static void _starpu_default_check_ft(void *arg)
{
	struct starpu_task *meta_task = arg;
	struct starpu_task *current_task = starpu_task_get_current();
	struct starpu_task *new_task;
	int ret;

	if (!current_task->failed)
	{
		starpu_task_ft_success(meta_task);
		return;
	}

	new_task = starpu_task_ft_create_retry
(meta_task, current_task, _starpu_default_check_ft);

	ret = starpu_task_submit_nodeps(new_task);
	STARPU_ASSERT(!ret);
}

void starpu_task_ft_prologue(void *arg)
{
	struct starpu_task *meta_task = starpu_task_get_current();
	struct starpu_task *new_task;
	void (*check_ft)(void*) = arg;
	int ret;

	if (!check_ft)
		check_ft = _starpu_default_check_ft;

	/* Create a task which will do the actual computation */
	new_task = starpu_task_ft_create_retry
(meta_task, meta_task, check_ft);

	ret = starpu_task_submit_nodeps(new_task);
	STARPU_ASSERT(!ret);

	/* Make the parent task wait for the result getting correct */
	starpu_task_end_dep_add(meta_task, 1);
	meta_task->where = STARPU_NOWHERE;
}

void starpu_task_ft_failed(struct starpu_task *task)
{
	_starpu_ft_check_support(task);

	task->failed = 1;
}

void starpu_task_ft_success(struct starpu_task *meta_task)
{
	starpu_task_end_dep_release(meta_task);
}

char *starpu_task_status_get_as_string(enum starpu_task_status status)
{
	switch(status)
	{
	case(STARPU_TASK_INIT) : return "STARPU_TASK_INIT";
	case(STARPU_TASK_BLOCKED): return "STARPU_TASK_BLOCKED";
	case(STARPU_TASK_READY): return "STARPU_TASK_READY";
	case(STARPU_TASK_RUNNING): return "STARPU_TASK_RUNNING";
	case(STARPU_TASK_FINISHED): return "STARPU_TASK_FINISHED";
	case(STARPU_TASK_BLOCKED_ON_TAG): return "STARPU_TASK_BLOCKED_ON_TAG";
	case(STARPU_TASK_BLOCKED_ON_TASK): return "STARPU_TASK_BLOCKED_ON_TASK";
	case(STARPU_TASK_BLOCKED_ON_DATA): return "STARPU_TASK_BLOCKED_ON_DATA";
	case(STARPU_TASK_STOPPED): return "STARPU_TASK_STOPPED";
	default: return "STARPU_TASK_unknown_status";
	}
}

void starpu_codelet_nop_func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
}

struct starpu_codelet starpu_codelet_nop =
{
	.cpu_funcs = {starpu_codelet_nop_func},
	.cuda_funcs = {starpu_codelet_nop_func},
	.hip_funcs = {starpu_codelet_nop_func},
	.opencl_funcs = {starpu_codelet_nop_func},
	.cpu_funcs_name = {"starpu_codelet_nop_func"},
	.model = NULL,
	.nbuffers = 0
};
