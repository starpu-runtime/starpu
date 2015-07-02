/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2015  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2013, 2015  CNRS
 * Copyright (C) 2011, 2014 INRIA
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

#include <starpu.h>
#include <common/config.h>
#include <core/jobs.h>

/* Internal version of starpu_task_destroy: don't check task->destroy flag */
void _starpu_task_destroy(struct starpu_task *task);

#ifdef STARPU_OPENMP
/* Test for the termination of the task.
 * Call starpu_task_destroy if required and the task is terminated. */
int _starpu_task_test_termination(struct starpu_task *task);
#endif

/* A pthread key is used to store the task currently executed on the thread.
 * _starpu_task_init initializes this pthread key and
 * _starpu_set_current_task updates its current value. */
void _starpu_task_init(void);
void _starpu_task_deinit(void);
void _starpu_set_current_task(struct starpu_task *task);

/* NB the second argument makes it possible to count regenerable tasks only
 * once. */
int _starpu_submit_job(struct _starpu_job *j);

int _starpu_task_submit_nodeps(struct starpu_task *task);

void _starpu_task_declare_deps_array(struct starpu_task *task, unsigned ndeps, struct starpu_task *task_array[], int check);

/* Returns the job structure (which is the internal data structure associated
 * to a task). */
struct _starpu_job *_starpu_get_job_associated_to_task(struct starpu_task *task);

/* Submits starpu internal tasks to the initial context */
int _starpu_task_submit_internally(struct starpu_task *task);

int _starpu_handle_needs_conversion_task(starpu_data_handle_t handle,
					 unsigned int node);
int
_starpu_handle_needs_conversion_task_for_arch(starpu_data_handle_t handle,
				     enum starpu_node_kind node_kind);

#ifdef STARPU_OPENMP
/* Prepare the current task for accepting new dependencies before becoming a continuation. */
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

static inline starpu_opencl_func_t _starpu_task_get_opencl_nth_implementation(struct starpu_codelet *cl, unsigned nimpl)
{
	return cl->opencl_funcs[nimpl];
}

static inline starpu_mic_func_t _starpu_task_get_mic_nth_implementation(struct starpu_codelet *cl, unsigned nimpl)
{
	return cl->mic_funcs[nimpl];
}

static inline starpu_scc_func_t _starpu_task_get_scc_nth_implementation(struct starpu_codelet *cl, unsigned nimpl)
{
	return cl->scc_funcs[nimpl];
}

static inline char *_starpu_task_get_cpu_name_nth_implementation(struct starpu_codelet *cl, unsigned nimpl)
{
	return cl->cpu_funcs_name[nimpl];
}

#define _STARPU_TASK_SET_INTERFACE(task, interface, i) do { if (task->dyn_handles) task->dyn_interfaces[i] = interface; else task->interfaces[i] = interface;} while(0)
#define _STARPU_TASK_GET_INTERFACES(task) ((task->dyn_handles) ? task->dyn_interfaces : task->interfaces)

void _starpu_watchdog_init(void);
void _starpu_watchdog_shutdown(void);

#endif // __CORE_TASK_H__
