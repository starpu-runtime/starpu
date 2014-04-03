/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014  Inria
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
#ifdef STARPU_OPENMP
#include <util/openmp_runtime_support.h>
#include <stdlib.h>
#include <ctype.h>
#include <strings.h>

#define _STARPU_STACKSIZE 2097152

static struct starpu_omp_global _global_state;
static starpu_pthread_key_t omp_thread_key;
static starpu_pthread_key_t omp_task_key;

struct starpu_omp_global *_starpu_omp_global_state = NULL;
double _starpu_omp_clock_ref = 0.0; /* clock reference for starpu_omp_get_wtick */


static struct starpu_omp_device *create_omp_device_struct(void)
{
	struct starpu_omp_device *dev = malloc(sizeof(*dev));
	if (dev == NULL)
		_STARPU_ERROR("memory allocation failed");

	/* TODO: initialize dev->icvs with proper values */ 
	memset(&dev->icvs, 0, sizeof(dev->icvs));

	return dev;
}

static struct starpu_omp_region *create_omp_region_struct(struct starpu_omp_region *parent_region, struct starpu_omp_device *owner_device, int nb_threads)
{
	struct starpu_omp_region *region = malloc(sizeof(*region));
	if (region == NULL)
		_STARPU_ERROR("memory allocation failed");

	region->parent_region = parent_region;
	region->owner_device = owner_device;
	region->threads_list = calloc(nb_threads, sizeof(*region->threads_list));
	if (region->threads_list == NULL)
		_STARPU_ERROR("memory allocation failed");

	region->nb_threads = nb_threads;
	return region;
}

static void set_master_thread(struct starpu_omp_region *region, struct starpu_omp_thread *master_thread)
{
	STARPU_ASSERT(region->nb_threads >= 1 && region->threads_list[0] == NULL);
	region->threads_list[0] = master_thread;
}

static struct starpu_omp_thread *create_omp_thread_struct(struct starpu_omp_region *owner_region)
{
	struct starpu_omp_thread *thread = malloc(sizeof(*thread));
	if (thread == NULL)
		_STARPU_ERROR("memory allocation failed");
	thread->owner_region = owner_region;
	memset(&thread->ctx, 0, sizeof(thread->ctx));
	return thread;
}

static void starpu_omp_task_entry(struct starpu_omp_task *task)
{
	task->f(task->starpu_buffers, task->starpu_cl_arg);
	task->state = starpu_omp_task_state_terminated;
	/* at the end of starpu task function, give hand back to the owner thread */
	setcontext(&task->owner_thread->ctx);
}

static void starpu_omp_task_preempt(void)
{
	struct starpu_omp_task *task = STARPU_PTHREAD_GETSPECIFIC(omp_task_key);
	struct starpu_omp_thread *thread = STARPU_PTHREAD_GETSPECIFIC(omp_thread_key);
	task->state = starpu_omp_task_state_preempted;

	/* we reached a blocked state, give hand back to the thread */
	swapcontext(&task->ctx, &thread->ctx);
}

static void starpu_omp_task_exec(void *buffers[], void *cl_arg)
{
	struct starpu_omp_task *task = starpu_task_get_current()->omp_task;
	STARPU_PTHREAD_SETSPECIFIC(omp_task_key, task);
	struct starpu_omp_thread *thread = STARPU_PTHREAD_GETSPECIFIC(omp_thread_key);
	if (task->state != starpu_omp_task_state_preempted)
	{
		task->starpu_buffers = buffers;
		task->starpu_cl_arg = cl_arg;
	}
	task->state = starpu_omp_task_state_clear;

	/* launch actual task on its own stack, or restore a previously preempted task */
	swapcontext(&thread->ctx, &task->ctx);

	STARPU_ASSERT(task->state == starpu_omp_task_state_preempted
			|| task->state == starpu_omp_task_state_terminated);
	STARPU_PTHREAD_SETSPECIFIC(omp_task_key, NULL);

	/* TODO: analyse the cause of the return and take appropriate steps */
}

static struct starpu_omp_task *create_omp_task_struct(struct starpu_omp_task *parent_task,
		struct starpu_omp_thread *owner_thread, struct starpu_omp_region *owner_region, int is_implicit)
{
	struct starpu_omp_task *task = malloc(sizeof(*task));
	if (task == NULL)
		_STARPU_ERROR("memory allocation failed");
	task->parent_task = parent_task;
	task->owner_thread = owner_thread;
	task->owner_region = owner_region;
	task->is_implicit = is_implicit;
	/* TODO: initialize task->data_env_icvs with proper values */ 
	memset(&task->data_env_icvs, 0, sizeof(task->data_env_icvs));
	if (is_implicit)
	{
	  /* TODO: initialize task->implicit_task_icvs with proper values */ 
		memset(&task->implicit_task_icvs, 0, sizeof(task->implicit_task_icvs));
	}
	task->starpu_task = NULL;
	task->starpu_buffers = NULL;
	task->starpu_cl_arg = NULL;
	task->f = NULL;
	task->state = starpu_omp_task_state_clear;

	if (parent_task == NULL)
	{
		/* do not allocate a stack for the initial task */
		task->stack = NULL;
		memset(&task->ctx, 0, sizeof(task->ctx));
	}
	else
	{
		/* TODO: use ICV stack size info instead */
		task->stack = malloc(_STARPU_STACKSIZE);
		getcontext(&task->ctx);
		/*
		 * we do not use uc_link, starpu_omp_task_entry will handle
		 * the end of the task
		 */
		task->ctx.uc_link                 = NULL;
		task->ctx.uc_stack.ss_sp          = task->stack;
		task->ctx.uc_stack.ss_size        = _STARPU_STACKSIZE;
		makecontext(&task->ctx, starpu_omp_task_entry, 1, task);
	}

	return task;
}

/*
 * Entry point to be called by the OpenMP runtime constructor
 */
int starpu_omp_init(void)
{
	_starpu_omp_environment_init();
	_global_state.icvs.cancel_var = _starpu_omp_initial_icv_values->cancel_var;
	_global_state.initial_device = create_omp_device_struct();
	_global_state.initial_region = create_omp_region_struct(NULL, _global_state.initial_device, 1);
	_global_state.initial_thread = create_omp_thread_struct(_global_state.initial_region);
	/* TODO: initialize context for initial thread */

	set_master_thread(_global_state.initial_region, _global_state.initial_thread);
	_global_state.initial_task = create_omp_task_struct(NULL,
			_global_state.initial_thread, _global_state.initial_region, 1);
	_starpu_omp_global_state = &_global_state;

	STARPU_PTHREAD_KEY_CREATE(&omp_thread_key, NULL);
	STARPU_PTHREAD_KEY_CREATE(&omp_task_key, NULL);
	int ret = starpu_init(0);
	if(ret < 0)
		return ret;

	/* init clock reference for starpu_omp_get_wtick */
	_starpu_omp_clock_ref = starpu_timing_now();

	return 0;
}

void starpu_omp_shutdown(void)
{
	starpu_shutdown();
	STARPU_PTHREAD_KEY_DELETE(omp_task_key);
	STARPU_PTHREAD_KEY_DELETE(omp_thread_key);
	/* TODO: free ICV variables */
	/* TODO: free task/thread/region/device structures */
}

#endif /* STARPU_OPENMP */
