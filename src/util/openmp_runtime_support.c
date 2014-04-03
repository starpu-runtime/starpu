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
/*
 * locally disable -Wdeprecated-declarations to avoid
 * lots of deprecated warnings for ucontext related functions
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
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

static void omp_initial_thread_func(struct starpu_omp_thread *init_thread, struct starpu_omp_task *init_task)
{
	while (1)
	{
		starpu_driver_run_once(&init_thread->starpu_driver);

		/* TODO: check if we are leaving the first nested region or not
		 *
		 * if we are leaving the first nested region we give control back to initial task
		 * otherwise, we should continue to execute work */
		swapcontext(&init_thread->ctx, &init_task->ctx);
	}
}

/*
 * setup the main application thread to handle the possible preemption of the initial task
 */
static void omp_initial_thread_setup(void)
{
	struct starpu_omp_thread *initial_thread = _global_state.initial_thread;
	struct starpu_omp_task *initial_task = _global_state.initial_task;
	/* .current_task */
	initial_thread->current_task = initial_task;
	/* .owner_region already set in create_omp_thread_struct */
	/* .initial_thread_stack */
	initial_thread->initial_thread_stack = malloc(_STARPU_STACKSIZE);
	if (initial_thread->initial_thread_stack == NULL)
		_STARPU_ERROR("memory allocation failed");
	/* .ctx */
	getcontext(&initial_thread->ctx);
	/*
	 * we do not use uc_link, the initial thread always should give hand back to the initial task
	 */
	initial_thread->ctx.uc_link          = NULL;
	initial_thread->ctx.uc_stack.ss_sp   = initial_thread->initial_thread_stack;
	initial_thread->ctx.uc_stack.ss_size = _STARPU_STACKSIZE;
	makecontext(&initial_thread->ctx, omp_initial_thread_func, 2, initial_thread, initial_task);
	/* .starpu_driver */
	/*
	 * we configure starpu to not launch CPU worker 0
	 * because we will use the main thread to play the role of worker 0
	 */
	struct starpu_conf conf;
	int ret = starpu_conf_init(&conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_conf_init");
	initial_thread->starpu_driver.type = STARPU_CPU_WORKER;
	initial_thread->starpu_driver.id.cpu_id = 0;
	conf.not_launched_drivers = &initial_thread->starpu_driver;
	conf.n_not_launched_drivers = 1;
	ret = starpu_init(&conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_driver_init(&initial_thread->starpu_driver);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_driver_init");
}

static void omp_initial_thread_exit()
{
	struct starpu_omp_thread *initial_thread = _global_state.initial_thread;
	int ret = starpu_driver_deinit(&initial_thread->starpu_driver);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_driver_deinit");
	starpu_shutdown();

	/* TODO: free initial_thread data structures */
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
	/* .current_task */
	thread->current_task = NULL;
	/* .owner_region */
	thread->owner_region = owner_region;
	/* .init_thread_stack */
	thread->initial_thread_stack = NULL;
	/* .ctx */
	memset(&thread->ctx, 0, sizeof(thread->ctx));
	/* .starpu_driver will be initialized later on */
	return thread;
}

static void starpu_omp_task_entry(struct starpu_omp_task *task)
{
	task->f(task->starpu_buffers, task->starpu_cl_arg);
	task->state = starpu_omp_task_state_terminated;
	struct starpu_omp_thread *thread = STARPU_PTHREAD_GETSPECIFIC(omp_thread_key);
	/* 
	 * the task reached the terminated state, definitively give hand back to the worker code.
	 *
	 * about to run on the worker stack...
	 */
	setcontext(&thread->ctx);
	STARPU_ASSERT(0); /* unreachable code */
}

/*
 * stop executing a task that is about to block
 * and give hand back to the thread
 */
static void starpu_omp_task_preempt(void)
{
	struct starpu_omp_task *task = STARPU_PTHREAD_GETSPECIFIC(omp_task_key);
	struct starpu_omp_thread *thread = STARPU_PTHREAD_GETSPECIFIC(omp_thread_key);
	task->state = starpu_omp_task_state_preempted;

	/* 
	 * the task reached a blocked state, give hand back to the worker code.
	 *
	 * about to run on the worker stack...
	 */
	swapcontext(&task->ctx, &thread->ctx);
	/* now running on the task stack again */
}

/*
 * wrap a task function to allow the task to be preempted
 */
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

	/* 
	 * start the task execution, or restore a previously preempted task.
	 * about to run on the task stack...
	 * */
	swapcontext(&thread->ctx, &task->ctx);
	/* now running on the worker stack again */

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
		if (task->stack == NULL)
			_STARPU_ERROR("memory allocation failed");
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

	omp_initial_thread_setup();

	/* init clock reference for starpu_omp_get_wtick */
	_starpu_omp_clock_ref = starpu_timing_now();

	return 0;
}

void starpu_omp_shutdown(void)
{
	omp_initial_thread_exit();
	STARPU_PTHREAD_KEY_DELETE(omp_task_key);
	STARPU_PTHREAD_KEY_DELETE(omp_thread_key);
	/* TODO: free ICV variables */
	/* TODO: free task/thread/region/device structures */
}
/*
 * restore deprecated diagnostics (-Wdeprecated-declarations)
 */
#pragma GCC diagnostic pop
#endif /* STARPU_OPENMP */
