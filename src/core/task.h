/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __CORE_TASK_H__
#define __CORE_TASK_H__

/** @file */

#include <starpu.h>
#include <common/config.h>
#include <core/jobs.h>
#include <common/starpu_spinlock.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma GCC visibility push(hidden)

/** Internal version of starpu_task_destroy: don't check task->destroy flag */
void _starpu_task_destroy(struct starpu_task *task);

#ifdef STARPU_OPENMP
/** Test for the termination of the task.
 * Call starpu_task_destroy if required and the task is terminated. */
int _starpu_task_test_termination(struct starpu_task *task);
#endif

/** A pthread key is used to store the task currently executed on the thread.
 * _starpu_task_init initializes this pthread key and
 * _starpu_set_current_task updates its current value. */
void _starpu_task_init(void);
void _starpu_task_deinit(void);
void _starpu_set_current_task(struct starpu_task *task);

int _starpu_submit_job(struct _starpu_job *j, int nodeps);

void _starpu_task_declare_deps_array(struct starpu_task *task, unsigned ndeps, struct starpu_task *task_array[], int check);

#define _STARPU_JOB_UNSET ((struct _starpu_job *) NULL)
#define _STARPU_JOB_SETTING ((struct _starpu_job *) 1)

/** Returns the job structure (which is the internal data structure associated
 * to a task). */
struct _starpu_job *_starpu_get_job_associated_to_task_slow(struct starpu_task *task, struct _starpu_job *job);
static inline struct _starpu_job *_starpu_get_job_associated_to_task(struct starpu_task *task)
{
	STARPU_ASSERT(task);
	struct _starpu_job *job = *(struct _starpu_job * volatile *) &task->starpu_private;

	if (STARPU_LIKELY(job != _STARPU_JOB_UNSET && job != _STARPU_JOB_SETTING))
	{
		/* Already available */
		STARPU_RMB();
		return job;
	}

	return _starpu_get_job_associated_to_task_slow(task, job);
}

/** Submits starpu internal tasks to the initial context */
int _starpu_task_submit_internally(struct starpu_task *task);

int _starpu_handle_needs_conversion_task(starpu_data_handle_t handle,
					 unsigned int node);
int
_starpu_handle_needs_conversion_task_for_arch(starpu_data_handle_t handle,
				     enum starpu_node_kind node_kind);

#ifdef STARPU_OPENMP
/** Prepare the current task for accepting new dependencies before becoming a continuation. */
void _starpu_task_prepare_for_continuation_ext(unsigned continuation_resubmit,
		void (*continuation_callback_on_sleep)(void *arg), void *continuation_callback_on_sleep_arg);

void _starpu_task_prepare_for_continuation(void);

void _starpu_task_set_omp_cleanup_callback(struct starpu_task *task, void (*omp_cleanup_callback)(void *arg),
		void *omp_cleanup_callback_arg);
#endif

int _starpu_task_uses_multiformat_handles(struct starpu_task *task);

int _starpu_task_submit_conversion_task(struct starpu_task *task,
					unsigned int workerid);

void _starpu_task_check_deprecated_fields(struct starpu_task *task);
void _starpu_codelet_check_deprecated_fields(struct starpu_codelet *cl);

static inline starpu_cpu_func_t _starpu_task_get_cpu_nth_implementation(struct starpu_codelet *cl, unsigned nimpl)
{
	return cl->cpu_funcs[nimpl];
}

static inline starpu_cuda_func_t _starpu_task_get_cuda_nth_implementation(struct starpu_codelet *cl, unsigned nimpl)
{
	return cl->cuda_funcs[nimpl];
}

static inline starpu_hip_func_t _starpu_task_get_hip_nth_implementation(struct starpu_codelet *cl, unsigned nimpl)
{
	return cl->hip_funcs[nimpl];
}

static inline starpu_opencl_func_t _starpu_task_get_opencl_nth_implementation(struct starpu_codelet *cl, unsigned nimpl)
{
	return cl->opencl_funcs[nimpl];
}

static inline starpu_max_fpga_func_t _starpu_task_get_fpga_nth_implementation(struct starpu_codelet *cl, unsigned nimpl)
{
	return cl->max_fpga_funcs[nimpl];
}

static inline const char *_starpu_task_get_cpu_name_nth_implementation(struct starpu_codelet *cl, unsigned nimpl)
{
	return cl->cpu_funcs_name[nimpl];
}

#define _STARPU_TASK_SET_INTERFACE(task, interface, i) do { if (task->dyn_handles) task->dyn_interfaces[i] = interface; else task->interfaces[i] = interface;} while(0)
#define _STARPU_TASK_GET_INTERFACES(task) ((task->dyn_handles) ? task->dyn_interfaces : task->interfaces)

void _starpu_watchdog_init(void);
void _starpu_watchdog_shutdown(void);

int _starpu_task_wait_for_all_and_return_nb_waited_tasks(void);
int _starpu_task_wait_for_all_in_ctx_and_return_nb_waited_tasks(unsigned sched_ctx);

#pragma GCC visibility pop

#ifdef BUILDING_STARPU
LIST_CREATE_TYPE_NOSTRUCT(starpu_task, prev, next);
PRIO_LIST_CREATE_TYPE(starpu_task, priority);
#endif

/** transaction states */
enum _starpu_trs_state
{
	_starpu_trs_uninitialized	= 0,
	_starpu_trs_initialized		= 1,
};

/** transaction epoch states */
enum _starpu_trs_epoch_state
{
	_starpu_trs_epoch_uninitialized	= 0,

	/** epoch is initialized but its entry task has not yet been executed to decide whether to confirm of cancel its execution */
	_starpu_trs_epoch_inactive	= 1,

	/** epoch has been confirmed for execution, its tasks will be actually executed */
	_starpu_trs_epoch_confirmed	= 2,

	/** epoch has been cancelled, its task will be skipped */
	_starpu_trs_epoch_cancelled	= 3,

	/** the exit task of the epoch has been executed */
	_starpu_trs_epoch_terminated	= 4,
};

LIST_TYPE(_starpu_trs_epoch,
	enum _starpu_trs_epoch_state state;

	/** if 1, the epoch entry task will wait on some user-supplied handle
	 * TODO: only used for first epoch on transaction opening for now, add for next epoch */
	int do_sync;

	/** if 1, the epoch is the first of the transaction */
	int is_begin;

	/** if 1, the epoch will be the last, and the transaction will be closed after its execution */
	int is_end;

	/** inline argument supplied by the user and passed to the user function deciding whether to start
	 * or cancel the epoch execution */
	void *do_start_arg;
);

struct starpu_transaction
{
	/** epoch list lock */
	struct _starpu_spinlock lock;
	struct _starpu_trs_epoch_list epoch_list;

	/** handle of the transaction object */
	starpu_data_handle_t handle;

	/** dummy data area referenced by the handle */
	int dummy_data;

	/** user function to decide whether to start or cancel an epoch execution, buffer[0] will
	 * optionally refer to an user suppled handle's object */
	int (*do_start_func)(void *buffer, void* arg);
	enum _starpu_trs_state state;

	/** flags, unused for now */
	int flags;
};

#ifdef __cplusplus
}
#endif

#endif // __CORE_TASK_H__
