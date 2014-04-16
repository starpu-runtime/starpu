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
#include <core/task.h>
#include <core/workers.h>
#include <common/list.h>
#include <common/starpu_spinlock.h>
#include <common/uthash.h>
#include <stdlib.h>
#include <ctype.h>
#include <strings.h>

#define _STARPU_STACKSIZE 2097152

static struct starpu_omp_global _global_state;
static starpu_pthread_key_t omp_thread_key;
static starpu_pthread_key_t omp_task_key;

struct starpu_omp_global *_starpu_omp_global_state = NULL;
double _starpu_omp_clock_ref = 0.0; /* clock reference for starpu_omp_get_wtick */

static struct starpu_omp_critical *create_omp_critical_struct(void);
static void destroy_omp_critical_struct(struct starpu_omp_critical *critical);
static struct starpu_omp_device *create_omp_device_struct(void);
static void destroy_omp_device_struct(struct starpu_omp_device *device);
static struct starpu_omp_region *create_omp_region_struct(struct starpu_omp_region *parent_region, struct starpu_omp_device *owner_device);
static void destroy_omp_region_struct(struct starpu_omp_region *region);
static struct starpu_omp_thread *create_omp_thread_struct(struct starpu_omp_region *owner_region);
static void destroy_omp_thread_struct(struct starpu_omp_thread *thread);
static struct starpu_omp_task *create_omp_task_struct(struct starpu_omp_task *parent_task,
		struct starpu_omp_thread *owner_thread, struct starpu_omp_region *owner_region, int is_implicit);
static void destroy_omp_task_struct(struct starpu_omp_task *task);
static void _wake_up_locked_task(struct starpu_omp_task *task);
static void wake_up_barrier(struct starpu_omp_region *parallel_region);

static void register_thread_worker(struct starpu_omp_thread *thread)
{
	STARPU_ASSERT(thread->worker != NULL);
	_starpu_spin_lock(&_global_state.hash_workers_lock);
	struct _starpu_worker *check = thread->worker;
	struct starpu_omp_thread *tmp = NULL;
	HASH_FIND_PTR(_global_state.hash_workers, &check, tmp);
	STARPU_ASSERT(tmp == NULL);
	HASH_ADD_PTR(_global_state.hash_workers, worker, thread);
	_starpu_spin_unlock(&_global_state.hash_workers_lock);
}
static struct starpu_omp_thread *get_local_thread(void)
{
	struct starpu_omp_thread *thread = STARPU_PTHREAD_GETSPECIFIC(omp_thread_key);
	if (thread == NULL)
	{
		struct _starpu_worker *starpu_worker = _starpu_get_local_worker_key();
		STARPU_ASSERT(starpu_worker != NULL);
		_starpu_spin_lock(&_global_state.hash_workers_lock);
		HASH_FIND_PTR(_global_state.hash_workers, &starpu_worker, thread);
		STARPU_ASSERT(thread != NULL);
		_starpu_spin_unlock(&_global_state.hash_workers_lock);
		STARPU_PTHREAD_SETSPECIFIC(omp_thread_key, thread);
	}
	return thread;
}

static struct starpu_omp_critical *create_omp_critical_struct(void)
{
	struct starpu_omp_critical *critical = malloc(sizeof(*critical));
	memset(critical, 0, sizeof(*critical));
	_starpu_spin_init(&critical->lock);
	return critical;
}

static void destroy_omp_critical_struct(struct starpu_omp_critical *critical)
{
	STARPU_ASSERT(critical->state == 0);
	STARPU_ASSERT(critical->contention_list_head == NULL);
	_starpu_spin_destroy(&critical->lock);
	critical->name = NULL;
	free(critical);
}

static struct starpu_omp_device *create_omp_device_struct(void)
{
	struct starpu_omp_device *device = malloc(sizeof(*device));
	if (device == NULL)
		_STARPU_ERROR("memory allocation failed");
	memset(device, 0, sizeof(*device));

	/* TODO: initialize device->icvs with proper values */ 
	memset(&device->icvs, 0, sizeof(device->icvs));

	return device;
}

static void destroy_omp_device_struct(struct starpu_omp_device *device)
{
	memset(device, 0, sizeof(*device));
	free(device);
}

static struct starpu_omp_region *create_omp_region_struct(struct starpu_omp_region *parent_region, struct starpu_omp_device *owner_device)
{
	struct starpu_omp_region *region = malloc(sizeof(*region));
	if (region == NULL)
		_STARPU_ERROR("memory allocation failed");

	memset(region, 0, sizeof(*region));
	region->parent_region = parent_region;
	region->owner_device = owner_device;
	region->thread_list = starpu_omp_thread_list_new();
	region->implicit_task_list = starpu_omp_task_list_new();

	_starpu_spin_init(&region->lock);
	region->level = (parent_region != NULL)?parent_region->level+1:0;
	return region;
}

static void destroy_omp_region_struct(struct starpu_omp_region *region)
{
	STARPU_ASSERT(region->nb_threads == 0);
	STARPU_ASSERT(starpu_omp_thread_list_empty(region->thread_list));
	STARPU_ASSERT(starpu_omp_task_list_empty(region->implicit_task_list));
	STARPU_ASSERT(region->continuation_starpu_task == NULL);
	starpu_omp_thread_list_delete(region->thread_list);
	starpu_omp_task_list_delete(region->implicit_task_list);
	_starpu_spin_destroy(&region->lock);
	memset(region, 0, sizeof(*region));
	free(region);
}

static void omp_initial_thread_func(void)
{
	struct starpu_omp_thread *initial_thread = _global_state.initial_thread;
	struct starpu_omp_task *initial_task = _global_state.initial_task;
	while (1)
	{
		struct starpu_task *continuation_starpu_task = initial_task->nested_region->continuation_starpu_task;
		starpu_driver_run_once(&initial_thread->starpu_driver);

		/*
		 * if we are leaving the first nested region we give control back to initial task
		 * otherwise, we should continue to execute work
		 */
		if (_starpu_task_test_termination(continuation_starpu_task))
		{
			initial_task->nested_region->continuation_starpu_task = NULL;
			STARPU_PTHREAD_SETSPECIFIC(omp_task_key, initial_task);
			swapcontext(&initial_thread->ctx, &initial_task->ctx);
		}
	}
}

static struct starpu_omp_thread *create_omp_thread_struct(struct starpu_omp_region *owner_region)
{
	struct starpu_omp_thread *thread = starpu_omp_thread_new();
	if (thread == NULL)
		_STARPU_ERROR("memory allocation failed");
	memset(thread, 0, sizeof(*thread));
	thread->owner_region = owner_region;
	return thread;
}

static void destroy_omp_thread_struct(struct starpu_omp_thread *thread)
{
	STARPU_ASSERT(thread->current_task == NULL);
	if (thread->worker)
	{
		struct _starpu_worker *check = thread->worker;
		struct starpu_omp_thread *_tmp;
		_starpu_spin_lock(&_global_state.hash_workers_lock);
		HASH_FIND_PTR(_global_state.hash_workers, &check, _tmp);
		STARPU_ASSERT(_tmp == thread);
		HASH_DEL(_global_state.hash_workers, _tmp);
		_starpu_spin_unlock(&_global_state.hash_workers_lock);
	}
	memset(thread, 0, sizeof(*thread));
	starpu_omp_thread_delete(thread);
}

static void starpu_omp_explicit_task_entry(struct starpu_omp_task *task)
{
	STARPU_ASSERT(!task->is_implicit);
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

static void starpu_omp_implicit_task_entry(struct starpu_omp_task *task)
{
	STARPU_ASSERT(task->is_implicit);
	task->f(task->starpu_buffers, task->starpu_cl_arg);
	starpu_omp_barrier();
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
static void starpu_omp_implicit_task_exec(void *buffers[], void *cl_arg)
{
	struct starpu_omp_task *task = starpu_task_get_current()->omp_task;
	STARPU_ASSERT(task->is_implicit);
	STARPU_PTHREAD_SETSPECIFIC(omp_task_key, task);
	struct starpu_omp_thread *thread = get_local_thread();
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
	if (task->state == starpu_omp_task_state_terminated)
	{
		task->starpu_task->omp_task = NULL;
		task->starpu_task = NULL;
	}
	else if (task->state != starpu_omp_task_state_preempted)
		_STARPU_ERROR("invalid omp task state");
}
/*
 * wrap a task function to allow the task to be preempted
 */
static void starpu_omp_explicit_task_exec(void *buffers[], void *cl_arg)
{
	struct starpu_omp_task *task = starpu_task_get_current()->omp_task;
	STARPU_ASSERT(!task->is_implicit);
	STARPU_PTHREAD_SETSPECIFIC(omp_task_key, task);
	struct starpu_omp_thread *thread = get_local_thread();
	if (task->state != starpu_omp_task_state_preempted)
	{
		if (!task->is_untied)
		{
			struct _starpu_worker *starpu_worker = _starpu_get_local_worker_key();
			task->starpu_task->workerid = starpu_worker->workerid;
			task->starpu_task->execute_on_a_specific_worker = 1;
		}
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
	if (task->state == starpu_omp_task_state_terminated)
	{
		struct starpu_omp_task *parent_task = task->parent_task;
		struct starpu_omp_region *parallel_region = task->owner_region;
		_starpu_spin_lock(&parent_task->lock);
		if (STARPU_ATOMIC_ADD(&parent_task->child_task_count, -1) == 0)
		{
			if (parent_task->wait_on & starpu_omp_task_wait_on_task_childs)
			{
				parent_task->wait_on &= ~starpu_omp_task_wait_on_task_childs;
				_wake_up_locked_task(parent_task);
			}
		}
		_starpu_spin_unlock(&parent_task->lock);
		_starpu_spin_lock(&parallel_region->lock);
		if (STARPU_ATOMIC_ADD(&parallel_region->bound_explicit_task_count, -1) == 0)
		{
			struct starpu_omp_task *waiting_task = parallel_region->waiting_task;
			_starpu_spin_unlock(&parallel_region->lock);

			if (waiting_task)
			{
				_starpu_spin_lock(&waiting_task->lock);
				_starpu_spin_lock(&parallel_region->lock);
				parallel_region->waiting_task = NULL;
				STARPU_ASSERT(waiting_task->wait_on & starpu_omp_task_wait_on_region_tasks);
				waiting_task->wait_on &= ~starpu_omp_task_wait_on_region_tasks;
				_wake_up_locked_task(waiting_task);
				_starpu_spin_unlock(&parallel_region->lock);
				_starpu_spin_unlock(&waiting_task->lock);
			}
		}
		else
		{
			_starpu_spin_unlock(&parallel_region->lock);
		}
	}
	else if (task->state != starpu_omp_task_state_preempted)
		_STARPU_ERROR("invalid omp task state");
}

static struct starpu_omp_task *create_omp_task_struct(struct starpu_omp_task *parent_task,
		struct starpu_omp_thread *owner_thread, struct starpu_omp_region *owner_region, int is_implicit)
{
	struct starpu_omp_task *task = starpu_omp_task_new();
	if (task == NULL)
		_STARPU_ERROR("memory allocation failed");

	memset(task, 0, sizeof(*task));
	task->parent_task = parent_task;
	task->owner_thread = owner_thread;
	task->owner_region = owner_region;
	task->is_implicit = is_implicit;
	_starpu_spin_init(&task->lock);
	/* TODO: initialize task->data_env_icvs with proper values */ 
	memset(&task->data_env_icvs, 0, sizeof(task->data_env_icvs));
	if (is_implicit)
	{
	  /* TODO: initialize task->implicit_task_icvs with proper values */ 
		memset(&task->implicit_task_icvs, 0, sizeof(task->implicit_task_icvs));
	}

	if (parent_task != NULL)
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
		if (is_implicit)
		{
			makecontext(&task->ctx, (void (*) ()) starpu_omp_implicit_task_entry, 1, task);
		}
		else
		{
			makecontext(&task->ctx, (void (*) ()) starpu_omp_explicit_task_entry, 1, task);
		}
	}

	return task;
}

static void destroy_omp_task_struct(struct starpu_omp_task *task)
{
	STARPU_ASSERT(task->state == starpu_omp_task_state_terminated);
	STARPU_ASSERT(task->nested_region == NULL);
	STARPU_ASSERT(task->starpu_task == NULL);
	if (task->stack != NULL)
	{
		free(task->stack);
	}
	_starpu_spin_destroy(&task->lock);
	memset(task, 0, sizeof(*task));
	starpu_omp_task_delete(task);
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
	makecontext(&initial_thread->ctx, omp_initial_thread_func, 0);
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
	/* we are now ready to start StarPU */
	ret = starpu_init(&conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_driver_init(&initial_thread->starpu_driver);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_driver_init");
	STARPU_PTHREAD_SETSPECIFIC(omp_task_key, initial_task);
	initial_thread->worker = _starpu_get_worker_struct(0);
	STARPU_ASSERT(initial_thread->worker);
	STARPU_PTHREAD_SETSPECIFIC(omp_thread_key, initial_thread);
	register_thread_worker(initial_thread);
}

static void omp_initial_thread_exit()
{
	struct starpu_omp_thread *initial_thread = _global_state.initial_thread;
	int ret = starpu_driver_deinit(&initial_thread->starpu_driver);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_driver_deinit");
	memset(&initial_thread->starpu_driver, 0, sizeof (initial_thread->starpu_driver));
	/* the driver for the main thread is now de-inited, we can shutdown Starpu */
	starpu_shutdown();
	free(initial_thread->initial_thread_stack);
	initial_thread->initial_thread_stack = NULL;
	memset(&initial_thread->ctx, 0, sizeof (initial_thread->ctx));
	initial_thread->current_task = NULL;
}

static void omp_initial_region_setup(void)
{
	_global_state.initial_region->master_thread = _global_state.initial_thread;
	_global_state.initial_region->nb_threads++;
	starpu_omp_task_list_push_back(_global_state.initial_region->implicit_task_list,
			_global_state.initial_task);
	omp_initial_thread_setup();
}

static void omp_initial_region_exit(void)
{
	omp_initial_thread_exit();
	_global_state.initial_task->state = starpu_omp_task_state_terminated;
	starpu_omp_task_list_pop_front(_global_state.initial_region->implicit_task_list);
	_global_state.initial_region->master_thread = NULL;
	_global_state.initial_region->nb_threads--;
}


/*
 * Entry point to be called by the OpenMP runtime constructor
 */
int starpu_omp_init(void)
{
	STARPU_PTHREAD_KEY_CREATE(&omp_thread_key, NULL);
	STARPU_PTHREAD_KEY_CREATE(&omp_task_key, NULL);
	_starpu_omp_environment_init();
	_global_state.icvs.cancel_var = _starpu_omp_initial_icv_values->cancel_var;
	_global_state.initial_device = create_omp_device_struct();
	_global_state.initial_region = create_omp_region_struct(NULL, _global_state.initial_device);
	_global_state.initial_thread = create_omp_thread_struct(_global_state.initial_region);
	_global_state.initial_task = create_omp_task_struct(NULL,
			_global_state.initial_thread, _global_state.initial_region, 1);
	_global_state.default_critical = create_omp_critical_struct();
	_global_state.named_criticals = NULL;
	_starpu_spin_init(&_global_state.named_criticals_lock);
	_global_state.hash_workers = NULL;
	_starpu_spin_init(&_global_state.hash_workers_lock);
	_starpu_omp_global_state = &_global_state;

	omp_initial_region_setup();

	/* init clock reference for starpu_omp_get_wtick */
	_starpu_omp_clock_ref = starpu_timing_now();

	return 0;
}

void starpu_omp_shutdown(void)
{
	omp_initial_region_exit();
	/* TODO: free ICV variables */
	/* TODO: free task/thread/region/device structures */
	destroy_omp_task_struct(_global_state.initial_task);
	_global_state.initial_task = NULL;
	destroy_omp_thread_struct(_global_state.initial_thread);
	_global_state.initial_thread = NULL;
	destroy_omp_region_struct(_global_state.initial_region);
	_global_state.initial_region = NULL;
	destroy_omp_device_struct(_global_state.initial_device);
	_global_state.initial_device = NULL;
	destroy_omp_critical_struct(_global_state.default_critical);
	_global_state.default_critical = NULL;
	_starpu_spin_lock(&_global_state.named_criticals_lock);
	{
		struct starpu_omp_critical *critical, *tmp;
		HASH_ITER(hh, _global_state.named_criticals, critical, tmp)
		{
			STARPU_ASSERT(critical != NULL);
			HASH_DEL(_global_state.named_criticals, critical);
			destroy_omp_critical_struct(critical);
		}
	}
	STARPU_ASSERT(_global_state.named_criticals == NULL);
	_starpu_spin_unlock(&_global_state.named_criticals_lock);
	_starpu_spin_destroy(&_global_state.named_criticals_lock);
	_starpu_spin_lock(&_global_state.hash_workers_lock);
	{
		struct starpu_omp_thread *thread, *tmp;
		HASH_ITER(hh, _global_state.hash_workers, thread, tmp)
		{
			STARPU_ASSERT(thread != NULL);
			HASH_DEL(_global_state.hash_workers, thread);
			destroy_omp_thread_struct(thread);
		}
	}
	STARPU_ASSERT(_global_state.hash_workers == NULL);
	_starpu_spin_unlock(&_global_state.hash_workers_lock);
	_starpu_spin_destroy(&_global_state.hash_workers_lock);
	STARPU_PTHREAD_KEY_DELETE(omp_task_key);
	STARPU_PTHREAD_KEY_DELETE(omp_thread_key);
}

void starpu_omp_parallel_region(const struct starpu_codelet * const _parallel_region_cl,
		void * const parallel_region_cl_arg)
{
	struct starpu_omp_thread *master_thread = STARPU_PTHREAD_GETSPECIFIC(omp_thread_key);
	struct starpu_omp_task *task = STARPU_PTHREAD_GETSPECIFIC(omp_task_key);
	struct starpu_omp_region *region = task->owner_region;
	int ret;

	/* TODO: compute the proper nb_threads and launch additional workers as needed.
	 * for now, the level 1 parallel region spans all the threads
	 * and level >= 2 parallel regions have only one thread */
	int nb_threads = (region->level == 0)?starpu_cpu_worker_get_count():1;

	struct starpu_omp_region *new_region = 
		create_omp_region_struct(region, _global_state.initial_device);

	int i;
	for (i = 0; i < nb_threads; i++)
	{
		struct starpu_omp_thread *new_thread;
		
		if (i == 0)
		{
			new_thread = master_thread;
			new_region->master_thread = master_thread;
		}
		else
		{
			/* TODO: specify actual starpu worker */
			new_thread = create_omp_thread_struct(new_region);

			/* TODO: use a less arbitrary thread/worker mapping scheme */
			if (region->level == 0)
			{
				new_thread->worker = _starpu_get_worker_struct(i);
				register_thread_worker(new_thread);
			}
			else
			{
				new_thread->worker = master_thread->worker;
			}
			starpu_omp_thread_list_push_back(new_region->thread_list, new_thread);
		}

		new_region->nb_threads++;
		struct starpu_omp_task *new_task = create_omp_task_struct(task, new_thread, new_region, 1);
		starpu_omp_task_list_push_back(new_region->implicit_task_list, new_task);

	}
	STARPU_ASSERT(new_region->nb_threads == nb_threads);

	/* 
	 * if task == initial_task, create a starpu task as a continuation to all the implicit
	 * tasks of the new region, else prepare the task for preemption,
	 * to become itself a continuation to the implicit tasks of the new region
	 */
	if (task == _global_state.initial_task)
	{
		new_region->continuation_starpu_task = starpu_task_create();
		/* in that case, the continuation starpu task is only used for synchronisation */
		new_region->continuation_starpu_task->cl = NULL;
		new_region->continuation_starpu_task->workerid = master_thread->worker->workerid;
		new_region->continuation_starpu_task->execute_on_a_specific_worker = 1;
		/* this sync task will be tested for completion in omp_initial_thread_func() */
		new_region->continuation_starpu_task->detach = 0;

	}
	else
	{
		/* through the preemption, the parent starpu task becomes the continuation task */
		_starpu_task_prepare_for_continuation();
		new_region->continuation_starpu_task = task->starpu_task;
	}
	task->nested_region = new_region;

	/*
	 * create the starpu tasks for the implicit omp tasks,
	 * create explicit dependencies between these starpu tasks and the continuation starpu task
	 */
	struct starpu_omp_task * implicit_task;
	for (implicit_task  = starpu_omp_task_list_begin(new_region->implicit_task_list);
			implicit_task != starpu_omp_task_list_end(new_region->implicit_task_list);
			implicit_task  = starpu_omp_task_list_next(implicit_task))
	{
		implicit_task->cl = *_parallel_region_cl;
		/*
		 * save pointer to the regions user function from the parallel region codelet
		 *
		 * TODO: add support for multiple/heterogeneous implementations
		 */
		implicit_task->f = implicit_task->cl.cpu_funcs[0];

		/*
		 * plug the task wrapper into the parallel region codelet instead, to support task preemption
		 */
		implicit_task->cl.cpu_funcs[0] = starpu_omp_implicit_task_exec;

		implicit_task->starpu_task = starpu_task_create();
		implicit_task->starpu_task->cl = &implicit_task->cl;
		implicit_task->starpu_task->cl_arg = parallel_region_cl_arg;
		implicit_task->starpu_task->omp_task = implicit_task;
		implicit_task->starpu_task->workerid = implicit_task->owner_thread->worker->workerid;
		implicit_task->starpu_task->execute_on_a_specific_worker = 1;
		starpu_task_declare_deps_array(new_region->continuation_starpu_task, 1, &implicit_task->starpu_task);
	}

	/*
	 * submit all the region implicit starpu tasks
	 */
	for (implicit_task  = starpu_omp_task_list_begin(new_region->implicit_task_list);
			implicit_task != starpu_omp_task_list_end(new_region->implicit_task_list);
			implicit_task  = starpu_omp_task_list_next(implicit_task))
	{
		ret = starpu_task_submit(implicit_task->starpu_task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/*
	 * submit the region continuation starpu task if task == initial_task
	 */
	if (task == _global_state.initial_task)
	{
		ret = _starpu_task_submit_internally(new_region->continuation_starpu_task);
		STARPU_CHECK_RETURN_VALUE(ret, "_starpu_task_submit_internally");
	}

	/*
	 * preempt for completion of the region
	 */
	starpu_omp_task_preempt();
	if (task == _global_state.initial_task)
	{
		STARPU_ASSERT(new_region->continuation_starpu_task == NULL);
	}
	else
	{
		STARPU_ASSERT(new_region->continuation_starpu_task != NULL);
		new_region->continuation_starpu_task = NULL;
	}
	/*
	 * TODO: free region resources
	 */
	for (i = 0; i < nb_threads; i++)
	{
		if (i == 0)
		{
			new_region->master_thread = NULL;
		}
		else
		{
			struct starpu_omp_thread *region_thread = starpu_omp_thread_list_pop_front(new_region->thread_list);
			destroy_omp_thread_struct(region_thread);
		}
		new_region->nb_threads--;
		struct starpu_omp_task *implicit_task = starpu_omp_task_list_pop_front(new_region->implicit_task_list);
		destroy_omp_task_struct(implicit_task);
	}
	STARPU_ASSERT(new_region->nb_threads == 0);
	task->nested_region = NULL;
	destroy_omp_region_struct(new_region);
}

static void _wake_up_locked_task(struct starpu_omp_task *task)
{
	if (task->wait_on == 0)
	{
		int ret = starpu_task_submit(task->starpu_task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
}

static void wake_up_barrier(struct starpu_omp_region *parallel_region)
{
	struct starpu_omp_task *task = STARPU_PTHREAD_GETSPECIFIC(omp_task_key);
	struct starpu_omp_task *implicit_task;
	for (implicit_task  = starpu_omp_task_list_begin(parallel_region->implicit_task_list);
			implicit_task != starpu_omp_task_list_end(parallel_region->implicit_task_list);
			implicit_task  = starpu_omp_task_list_next(implicit_task))
	{
		if (implicit_task == task)
			continue;
		_starpu_spin_lock(&implicit_task->lock);
		STARPU_ASSERT(implicit_task->wait_on & starpu_omp_task_wait_on_barrier);
		implicit_task->wait_on &= ~starpu_omp_task_wait_on_barrier;
		_wake_up_locked_task(implicit_task);
		_starpu_spin_unlock(&implicit_task->lock);
	}
}

static void barrier__sleep_callback(void *_task)
{
	struct starpu_omp_task *task = _task;
	_starpu_spin_unlock(&task->lock);
}

static void region_tasks__sleep_callback(void *_task)
{
	struct starpu_omp_task *task = _task;
	struct starpu_omp_region *parallel_region = task->owner_region;
	_starpu_spin_unlock(&task->lock);
	_starpu_spin_unlock(&parallel_region->lock);
}

void starpu_omp_barrier(void)
{
	struct starpu_omp_task *task = STARPU_PTHREAD_GETSPECIFIC(omp_task_key);
	/* Assume barriers are performed in by the implicit tasks of a parallel_region */
	STARPU_ASSERT(task->is_implicit);
	struct starpu_omp_region *parallel_region = task->owner_region;
	_starpu_spin_lock(&task->lock);
	int inc_barrier_count = STARPU_ATOMIC_ADD(&parallel_region->barrier_count, 1);

	if (inc_barrier_count == parallel_region->nb_threads)
	{
		/* last task reaching the barrier */
		_starpu_spin_lock(&parallel_region->lock);
		parallel_region->barrier_count = 0;
		if (parallel_region->bound_explicit_task_count > 0)
		{
			task->wait_on |= starpu_omp_task_wait_on_region_tasks;
			parallel_region->waiting_task = task;
			_starpu_task_prepare_for_continuation_ext(0, region_tasks__sleep_callback, task);
			starpu_omp_task_preempt();
		}
		else
		{
			_starpu_spin_unlock(&task->lock);
			_starpu_spin_unlock(&parallel_region->lock);
		}
		wake_up_barrier(parallel_region);
	}
	else
	{
		/* not the last task reaching the barrier
		 * . prepare for conditional continuation 
		 * . sleep
		 */

		task->wait_on |= starpu_omp_task_wait_on_barrier;
		_starpu_task_prepare_for_continuation_ext(0, barrier__sleep_callback, task);
		starpu_omp_task_preempt();
		STARPU_ASSERT(task->child_task_count == 0);
	}
}

void starpu_omp_master(void (*f)(void *arg), void *arg, int nowait)
{
	struct starpu_omp_task *task = STARPU_PTHREAD_GETSPECIFIC(omp_task_key);
	struct starpu_omp_thread *thread = STARPU_PTHREAD_GETSPECIFIC(omp_thread_key);
	/* Assume singles are performed in by the implicit tasks of a region */
	STARPU_ASSERT(task->is_implicit);
	struct starpu_omp_region *region = task->owner_region;

	if (thread == region->master_thread)
	{
		f(arg);
	}

	if (!nowait)
	{
		starpu_omp_barrier();
	}
}

void starpu_omp_single(void (*f)(void *arg), void *arg, int nowait)
{
	struct starpu_omp_task *task = STARPU_PTHREAD_GETSPECIFIC(omp_task_key);
	/* Assume singles are performed in by the implicit tasks of a region */
	STARPU_ASSERT(task->is_implicit);
	struct starpu_omp_region *region = task->owner_region;
	int first = STARPU_BOOL_COMPARE_AND_SWAP(&region->single_id, task->single_id, task->single_id+1);
	task->single_id++;

	if (first)
	{
		f(arg);
	}

	if (!nowait)
	{
		starpu_omp_barrier();
	}
}

static void critical__sleep_callback(void *_critical)
{
	struct starpu_omp_critical *critical = _critical;
	_starpu_spin_unlock(&critical->lock);
}

void starpu_omp_critical(void (*f)(void *arg), void *arg, const char *name)
{
	struct starpu_omp_task *task = STARPU_PTHREAD_GETSPECIFIC(omp_task_key);
	struct starpu_omp_critical *critical = NULL;
	struct starpu_omp_task_link link;

	if (name)
	{
		_starpu_spin_lock(&_global_state.named_criticals_lock);
		HASH_FIND_STR(_global_state.named_criticals, name, critical);
		if (critical == NULL)
		{
			critical = create_omp_critical_struct();
			critical->name = name;
			HASH_ADD_STR(_global_state.named_criticals, name, critical);
		}
		_starpu_spin_unlock(&_global_state.named_criticals_lock);
	}
	else
	{
		critical = _global_state.default_critical;
	}

	_starpu_spin_lock(&critical->lock);
	while (critical->state != 0)
	{
		_starpu_spin_lock(&task->lock);
		task->wait_on |= starpu_omp_task_wait_on_critical;
		_starpu_spin_unlock(&task->lock);
		link.task = task;
		link.next = critical->contention_list_head;
		critical->contention_list_head = &link;
		_starpu_task_prepare_for_continuation_ext(0, critical__sleep_callback, critical);
		starpu_omp_task_preempt();

		/* re-acquire the spin lock */
		_starpu_spin_lock(&critical->lock);
	}
	critical->state = 1;
	_starpu_spin_unlock(&critical->lock);

	f(arg);

	_starpu_spin_lock(&critical->lock);
	STARPU_ASSERT(critical->state == 1);
	critical->state = 0;
	if (critical->contention_list_head != NULL)
	{
		struct starpu_omp_task *next_task = critical->contention_list_head->task;
		critical->contention_list_head = critical->contention_list_head->next;
		_starpu_spin_lock(&next_task->lock);
		STARPU_ASSERT(next_task->wait_on & starpu_omp_task_wait_on_critical);
		next_task->wait_on &= ~starpu_omp_task_wait_on_critical;
		_wake_up_locked_task(next_task);
		_starpu_spin_unlock(&next_task->lock);
	}
	_starpu_spin_unlock(&critical->lock);
}

static void explicit_task__destroy_callback(void *_task)
{
	struct starpu_omp_task *task = _task;
	STARPU_ASSERT(!task->is_implicit);
	task->starpu_task->omp_task = NULL;
	task->starpu_task = NULL;
	destroy_omp_task_struct(task);
}

void starpu_omp_task_region(const struct starpu_codelet * const _task_region_cl,
		void * const task_region_cl_arg,
		int if_clause, int final_clause, int untied_clause, int mergeable_clause)
{
	struct starpu_omp_task *generating_task = STARPU_PTHREAD_GETSPECIFIC(omp_task_key);
	struct starpu_omp_region *parallel_region = generating_task->owner_region;
	int is_undeferred = 0;
	int is_final = 0;
	int is_included = 0;
	int is_merged = 0;
	int is_untied = 0;
	int ret;

	if (!if_clause)
	{
		is_undeferred = 1;
	}
	if (generating_task->is_final)
	{
		is_final = 1;
		is_included = 1;
	}
	else if (final_clause)
	{
		is_final = 1;
	}
	if (is_included)
	{
		is_undeferred = 1;
	}
	if ((is_undeferred || is_included) & mergeable_clause)
	{
		is_merged = 1;
	}
	if (is_merged)
	{
		struct starpu_codelet task_region_cl = *_task_region_cl;
		(void)task_region_cl;
		_STARPU_ERROR("omp merged task unimplemented\n");
	}
	else
	{
		struct starpu_omp_task *generated_task =
			create_omp_task_struct(generating_task, NULL, parallel_region, 0);
		generated_task->cl = *_task_region_cl;
		if (untied_clause)
		{
			is_untied = 1;
		}
		generated_task->is_undeferred = is_undeferred;
		generated_task->is_final = is_final;
		generated_task->is_untied = is_untied;
		generated_task->task_group = generating_task->task_group;

		/*
		 * save pointer to the regions user function from the task region codelet
		 *
		 * TODO: add support for multiple/heterogeneous implementations
		 */
		generated_task->f = generated_task->cl.cpu_funcs[0];

		/*
		 * plug the task wrapper into the task region codelet instead, to support task preemption
		 */
		generated_task->cl.cpu_funcs[0] = starpu_omp_explicit_task_exec;

		generated_task->starpu_task = starpu_task_create();
		generated_task->starpu_task->cl = &generated_task->cl;
		generated_task->starpu_task->cl_arg = task_region_cl_arg;
		generated_task->starpu_task->omp_task = generated_task;
		_starpu_task_set_omp_cleanup_callback(generated_task->starpu_task, explicit_task__destroy_callback, generated_task);
		/* if the task is tied, execute_on_a_specific_worker will be changed to 1
		 * upon the first preemption of the generated task, once we know
		 * which worker thread has been selected */
		generated_task->starpu_task->execute_on_a_specific_worker = 0;

		if (is_included)
		{
			_STARPU_ERROR("omp included task unimplemented\n");
		}
		else
		{
			(void)STARPU_ATOMIC_ADD(&generating_task->child_task_count, 1);
			(void)STARPU_ATOMIC_ADD(&parallel_region->bound_explicit_task_count, 1);
			{
				struct starpu_omp_task_group *_task_group = generated_task->task_group;
				while (_task_group)
				{
					(void)STARPU_ATOMIC_ADD(&_task_group->descendent_task_count, 1);
					_task_group = _task_group->next;
				}
			}
			if (is_undeferred)
			{
				_starpu_task_prepare_for_continuation();
				starpu_task_declare_deps_array(generating_task->starpu_task, 1,
						&generated_task->starpu_task);
			}
			ret = starpu_task_submit(generated_task->starpu_task);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
			if (is_undeferred)
			{
				starpu_omp_task_preempt();
			}
		}
	}
}

static void task_childs__sleep_callback(void *_task)
{
	struct starpu_omp_task *task = _task;
	_starpu_spin_unlock(&task->lock);
}

void starpu_omp_taskwait(void)
{
	struct starpu_omp_task *task = STARPU_PTHREAD_GETSPECIFIC(omp_task_key);
	_starpu_spin_lock(&task->lock);
	if (task->child_task_count > 0)
	{
		task->wait_on |= starpu_omp_task_wait_on_task_childs;
		_starpu_task_prepare_for_continuation_ext(0, task_childs__sleep_callback, task);
		starpu_omp_task_preempt();
		STARPU_ASSERT(task->child_task_count == 0);
	}
	else
	{
		_starpu_spin_unlock(&task->lock);
	}
}

/*
 * restore deprecated diagnostics (-Wdeprecated-declarations)
 */
#pragma GCC diagnostic pop
#endif /* STARPU_OPENMP */
