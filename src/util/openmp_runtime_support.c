/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <datawizard/interfaces/data_interface.h>
#include <stdlib.h>
#include <ctype.h>
#include <strings.h>

#define _STARPU_INITIAL_THREAD_STACKSIZE 2097152

static struct starpu_omp_global _global_state;
starpu_pthread_key_t omp_thread_key;
starpu_pthread_key_t omp_task_key;

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
static void wake_up_and_unlock_task(struct starpu_omp_task *task);
static void wake_up_barrier(struct starpu_omp_region *parallel_region);
static void starpu_omp_task_preempt(void);

struct starpu_omp_thread *_starpu_omp_get_thread(void)
{
	struct starpu_omp_thread *thread = STARPU_PTHREAD_GETSPECIFIC(omp_thread_key);
	return thread;
}

static inline void _starpu_omp_set_thread(struct starpu_omp_thread *thread)
{
	STARPU_PTHREAD_SETSPECIFIC(omp_thread_key, thread);
}

struct starpu_omp_task *_starpu_omp_get_task(void)
{
	struct starpu_omp_task *task = STARPU_PTHREAD_GETSPECIFIC(omp_task_key);
	return task;
}

static inline void _starpu_omp_set_task(struct starpu_omp_task *task)
{
	STARPU_PTHREAD_SETSPECIFIC(omp_task_key, task);
}

struct starpu_omp_region *_starpu_omp_get_region_at_level(int level)
{
	const struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_region *parallel_region;

	if (!task)
		return NULL;

	parallel_region = task->owner_region;
	if (level < 0 || level > parallel_region->icvs.levels_var)
		return NULL;

	while (level < parallel_region->icvs.levels_var)
	{
		parallel_region = parallel_region->parent_region;
	}

	return parallel_region;
}

int _starpu_omp_get_region_thread_num(const struct starpu_omp_region * const region)
{
	struct starpu_omp_thread *thread = _starpu_omp_get_thread();
	STARPU_ASSERT(thread != NULL);
	if (thread == region->master_thread)
		return 0;
	int tid = starpu_omp_thread_list_member(&region->thread_list, thread);
	if (tid >= 0)
		return tid+1;
	_STARPU_ERROR("unrecognized omp thread\n");
}

static void weak_task_lock(struct starpu_omp_task *task)
{
	_starpu_spin_lock(&task->lock);
	while (task->transaction_pending)
	{
		_starpu_spin_unlock(&task->lock);
		STARPU_UYIELD();
		_starpu_spin_lock(&task->lock);
	}
}

static void weak_task_unlock(struct starpu_omp_task *task)
{
	_starpu_spin_unlock(&task->lock);
}

static void wake_up_and_unlock_task(struct starpu_omp_task *task)
{
	STARPU_ASSERT(task->transaction_pending == 0);
	if (task->wait_on == 0)
	{
		weak_task_unlock(task);
		int ret = starpu_task_submit(task->starpu_task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	else
	{
		weak_task_unlock(task);
	}
}

static void transaction_callback(void *_task)
{
	struct starpu_omp_task *task = _task;
	_starpu_spin_lock(&task->lock);
	STARPU_ASSERT(task->transaction_pending != 0);
	task->transaction_pending = 0;
	_starpu_spin_unlock(&task->lock);
}

static void condition_init(struct starpu_omp_condition *condition)
{
	condition->contention_list_head = NULL;
}

static void condition_exit(struct starpu_omp_condition *condition)
{
	STARPU_ASSERT(condition->contention_list_head == NULL);
	condition->contention_list_head = NULL;
}

static void condition_wait(struct starpu_omp_condition *condition, struct _starpu_spinlock *lock, enum starpu_omp_task_wait_on flag)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_task_link link;
	_starpu_spin_lock(&task->lock);
	task->wait_on |= flag;
	link.task = task;
	link.next = condition->contention_list_head;
	condition->contention_list_head = &link;
	task->transaction_pending = 1;
	_starpu_spin_unlock(&task->lock);
	_starpu_spin_unlock(lock);
	_starpu_task_prepare_for_continuation_ext(0, transaction_callback, task);
	starpu_omp_task_preempt();

	/* re-acquire the lock released by the callback */
	_starpu_spin_lock(lock);
}

#if 0
/* unused for now */
static void condition_signal(struct starpu_omp_condition *condition)
{
	if (condition->contention_list_head != NULL)
	{
		struct starpu_omp_task *next_task = condition->contention_list_head->task;
		weak_task_lock(next_task);
		condition->contention_list_head = condition->contention_list_head->next;
		STARPU_ASSERT(next_task->wait_on & starpu_omp_task_wait_on_condition);
		next_task->wait_on &= ~starpu_omp_task_wait_on_condition;
		wake_up_and_unlock_task(next_task);
	}
}
#endif

static void condition_broadcast(struct starpu_omp_condition *condition, enum starpu_omp_task_wait_on flag)
{
	while (condition->contention_list_head != NULL)
	{
		struct starpu_omp_task *next_task = condition->contention_list_head->task;
		weak_task_lock(next_task);
		condition->contention_list_head = condition->contention_list_head->next;
		STARPU_ASSERT(next_task->wait_on & flag);
		next_task->wait_on &= ~flag;
		wake_up_and_unlock_task(next_task);
	}
}

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
static struct starpu_omp_thread *get_worker_thread(struct _starpu_worker *starpu_worker)
{
	struct starpu_omp_thread *thread = NULL;
	_starpu_spin_lock(&_global_state.hash_workers_lock);
	HASH_FIND_PTR(_global_state.hash_workers, &starpu_worker, thread);
	_starpu_spin_unlock(&_global_state.hash_workers_lock);
	return thread;
}
static struct starpu_omp_thread *get_local_thread(void)
{
	struct starpu_omp_thread *thread = _starpu_omp_get_thread();
	if (thread == NULL)
	{
		struct _starpu_worker *starpu_worker = _starpu_get_local_worker_key();
		STARPU_ASSERT(starpu_worker != NULL);
		thread = get_worker_thread(starpu_worker);

		if (
#ifdef STARPU_USE_CUDA
				(starpu_worker->arch != STARPU_CUDA_WORKER)
				&&
#endif
#ifdef STARPU_USE_OPENCL
				(starpu_worker->arch != STARPU_OPENCL_WORKER)
				&&
#endif
				1
		   )
		{
			STARPU_ASSERT(thread != NULL);
		}

		if (thread != NULL)
		{
			_starpu_omp_set_thread(thread);
		}
	}
	return thread;
}

static struct starpu_omp_critical *create_omp_critical_struct(void)
{
	struct starpu_omp_critical *critical;

	_STARPU_CALLOC(critical, 1, sizeof(*critical));
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
	struct starpu_omp_device *device;

	_STARPU_CALLOC(device, 1, sizeof(*device));
	_starpu_spin_init(&device->atomic_lock);
	return device;
}

static void destroy_omp_device_struct(struct starpu_omp_device *device)
{
	_starpu_spin_destroy(&device->atomic_lock);
	memset(device, 0, sizeof(*device));
	free(device);
}

static struct starpu_omp_device *get_caller_device(void)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_device *device;
	if (task)
	{
		STARPU_ASSERT(task->owner_region != NULL);
		device = task->owner_region->owner_device;
	}
	else
	{
		device = _global_state.initial_device;
	}
	STARPU_ASSERT(device != NULL);
	return device;
}

static struct starpu_omp_region *create_omp_region_struct(struct starpu_omp_region *parent_region, struct starpu_omp_device *owner_device)
{
	struct starpu_omp_region *region;

	_STARPU_CALLOC(region, 1, sizeof(*region));
	region->parent_region = parent_region;
	region->owner_device = owner_device;
	starpu_omp_thread_list_init(&region->thread_list);

	_starpu_spin_init(&region->lock);
	_starpu_spin_init(&region->registered_handles_lock);
	region->level = (parent_region != NULL)?parent_region->level+1:0;
	return region;
}

static void destroy_omp_region_struct(struct starpu_omp_region *region)
{
	STARPU_ASSERT(region->nb_threads == 0);
	STARPU_ASSERT(starpu_omp_thread_list_empty(&region->thread_list));
	STARPU_ASSERT(region->continuation_starpu_task == NULL);
	_starpu_spin_destroy(&region->registered_handles_lock);
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
			_starpu_omp_set_task(initial_task);
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
	memset(thread, 0, sizeof(*thread));
	starpu_omp_thread_delete(thread);
}

static void starpu_omp_explicit_task_entry(struct starpu_omp_task *task)
{
	STARPU_ASSERT(!(task->flags & STARPU_OMP_TASK_FLAGS_IMPLICIT));
	struct _starpu_worker *starpu_worker = _starpu_get_local_worker_key();
	/* XXX on work */
	if (task->is_loop)
	{
		starpu_omp_for_inline_first_alt(task->nb_iterations, task->chunk, starpu_omp_sched_static, 1, &task->begin_i, &task->end_i);
	}
	if (starpu_worker->arch == STARPU_CPU_WORKER)
	{
		task->cpu_f(task->starpu_buffers, task->starpu_cl_arg);
	}
#ifdef STARPU_USE_CUDA
	else if (starpu_worker->arch == STARPU_CUDA_WORKER)
	{
		task->cuda_f(task->starpu_buffers, task->starpu_cl_arg);
	}
#endif
#ifdef STARPU_USE_OPENCL
	else if (starpu_worker->arch == STARPU_OPENCL_WORKER)
	{
		task->opencl_f(task->starpu_buffers, task->starpu_cl_arg);
	}
#endif
	else
		_STARPU_ERROR("invalid worker architecture");
	/**/
	_starpu_omp_unregister_task_handles(task);
	_starpu_spin_lock(&task->lock);
	task->state = starpu_omp_task_state_terminated;
	task->transaction_pending=1;
	_starpu_spin_unlock(&task->lock);
	struct starpu_omp_thread *thread = _starpu_omp_get_thread();
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
	struct starpu_omp_thread *thread = _starpu_omp_get_thread();
	STARPU_ASSERT(task->flags & STARPU_OMP_TASK_FLAGS_IMPLICIT);
	task->cpu_f(task->starpu_buffers, task->starpu_cl_arg);
	starpu_omp_barrier();
	if (thread == task->owner_region->master_thread)
	{
		_starpu_omp_unregister_region_handles(task->owner_region);
	}
	task->state = starpu_omp_task_state_terminated;
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
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_thread *thread = _starpu_omp_get_thread();
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
	STARPU_ASSERT(task->flags & STARPU_OMP_TASK_FLAGS_IMPLICIT);
	_starpu_omp_set_task(task);
	struct starpu_omp_thread *thread = get_local_thread();
	if (task->state != starpu_omp_task_state_preempted)
	{
		task->starpu_buffers = buffers;
		task->starpu_cl_arg = cl_arg;
		STARPU_ASSERT(task->stack == NULL);
		STARPU_ASSERT(task->stacksize > 0);
		_STARPU_MALLOC(task->stack, task->stacksize);
		getcontext(&task->ctx);
		/*
		 * we do not use uc_link, starpu_omp_task_entry will handle
		 * the end of the task
		 */
		task->ctx.uc_link                 = NULL;
		task->ctx.uc_stack.ss_sp          = task->stack;
		task->ctx.uc_stack.ss_size        = task->stacksize;
		task->stack_vg_id                 = VALGRIND_STACK_REGISTER(task->stack, task->stack+task->stacksize);
		makecontext(&task->ctx, (void (*) ()) starpu_omp_implicit_task_entry, 1, task);
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
	_starpu_omp_set_task(NULL);

	/* TODO: analyse the cause of the return and take appropriate steps */
	if (task->state == starpu_omp_task_state_terminated)
	{
		task->starpu_task->omp_task = NULL;
		task->starpu_task = NULL;
		VALGRIND_STACK_DEREGISTER(task->stack_vg_id);
		task->stack_vg_id = 0;
		free(task->stack);
		task->stack = NULL;
		memset(&task->ctx, 0, sizeof(task->ctx));
	}
	else if (task->state != starpu_omp_task_state_preempted)
		_STARPU_ERROR("invalid omp task state");
}
static void starpu_omp_task_completion_accounting(struct starpu_omp_task *task)
{
	struct starpu_omp_task *parent_task = task->parent_task;
	struct starpu_omp_region *parallel_region = task->owner_region;

	weak_task_lock(parent_task);
	if (STARPU_ATOMIC_ADD(&parent_task->child_task_count, -1) == 0)
	{
		if (parent_task->state == starpu_omp_task_state_zombie)
		{
			STARPU_ASSERT(!(parent_task->flags & STARPU_OMP_TASK_FLAGS_IMPLICIT));
			weak_task_unlock(parent_task);
			destroy_omp_task_struct(parent_task);
		}
		else if (parent_task->wait_on & starpu_omp_task_wait_on_task_childs)
		{
			parent_task->wait_on &= ~starpu_omp_task_wait_on_task_childs;
			wake_up_and_unlock_task(parent_task);
		}
		else
		{
			weak_task_unlock(parent_task);
		}
	}
	else
	{
		weak_task_unlock(parent_task);
	}
	_starpu_spin_lock(&parallel_region->lock);
	if (STARPU_ATOMIC_ADD(&parallel_region->bound_explicit_task_count, -1) == 0)
	{
		struct starpu_omp_task *waiting_task = parallel_region->waiting_task;
		_starpu_spin_unlock(&parallel_region->lock);

		if (waiting_task)
		{
			weak_task_lock(waiting_task);
			_starpu_spin_lock(&parallel_region->lock);
			parallel_region->waiting_task = NULL;
			STARPU_ASSERT(waiting_task->wait_on & starpu_omp_task_wait_on_region_tasks);
			waiting_task->wait_on &= ~starpu_omp_task_wait_on_region_tasks;
			_starpu_spin_unlock(&parallel_region->lock);
			wake_up_and_unlock_task(waiting_task);
		}
	}
	else
	{
		_starpu_spin_unlock(&parallel_region->lock);
	}
	if (task->task_group)
	{
		struct starpu_omp_task *leader_task = task->task_group->leader_task;
		STARPU_ASSERT(leader_task != task);
		weak_task_lock(leader_task);
		if (STARPU_ATOMIC_ADD(&task->task_group->descendent_task_count, -1) == 0)
		{
			if (leader_task->wait_on & starpu_omp_task_wait_on_group
				&& task->task_group == leader_task->task_group)
				/* only wake the leader_task if it is actually
				 * waiting for the current task's task_group */
			{
				leader_task->wait_on &= ~starpu_omp_task_wait_on_group;
				wake_up_and_unlock_task(leader_task);
			}
			else
			{
				weak_task_unlock(leader_task);
			}
		}
		else
		{
			weak_task_unlock(leader_task);
		}
	}
}
/*
 * wrap a task function to allow the task to be preempted
 */
static void starpu_omp_explicit_task_exec(void *buffers[], void *cl_arg)
{
	struct starpu_omp_task *task = starpu_task_get_current()->omp_task;
	STARPU_ASSERT(!(task->flags & STARPU_OMP_TASK_FLAGS_IMPLICIT));
	_starpu_omp_set_task(task);

	struct starpu_omp_thread *thread = get_local_thread();
	if (task->state != starpu_omp_task_state_preempted)
	{
		if (thread == NULL)
		{
			struct _starpu_worker *starpu_worker = _starpu_get_local_worker_key();
			if (starpu_worker->arch != STARPU_CPU_WORKER)
			{
				if (
#ifdef STARPU_USE_CUDA
						(starpu_worker->arch != STARPU_CUDA_WORKER)
						&&
#endif
#ifdef STARPU_USE_OPENCL
						(starpu_worker->arch != STARPU_OPENCL_WORKER)
						&&
#endif
						1
				   )
				{
					_STARPU_ERROR("invalid worker architecture");
				}

				struct starpu_omp_thread *new_thread;
				new_thread = create_omp_thread_struct(NULL);
				new_thread->worker = starpu_worker;
				register_thread_worker(new_thread);

				thread = get_local_thread();
				STARPU_ASSERT(thread == new_thread);
			}
			else
			{
				_STARPU_ERROR("orphaned CPU thread");
			}
		}
		STARPU_ASSERT(thread != NULL);
		if (!(task->flags & STARPU_OMP_TASK_FLAGS_UNTIED))
		{
			struct _starpu_worker *starpu_worker = _starpu_get_local_worker_key();
			task->starpu_task->workerid = starpu_worker->workerid;
			task->starpu_task->execute_on_a_specific_worker = 1;
		}
		task->starpu_buffers = buffers;
		task->starpu_cl_arg = cl_arg;
		STARPU_ASSERT(task->stack == NULL);
		STARPU_ASSERT(task->stacksize > 0);
		_STARPU_MALLOC(task->stack, task->stacksize);
		getcontext(&task->ctx);
		/*
		 * we do not use uc_link, starpu_omp_task_entry will handle
		 * the end of the task
		 */
		task->ctx.uc_link                 = NULL;
		task->ctx.uc_stack.ss_sp          = task->stack;
		task->ctx.uc_stack.ss_size        = task->stacksize;
		makecontext(&task->ctx, (void (*) ()) starpu_omp_explicit_task_entry, 1, task);
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
	_starpu_omp_set_task(NULL);
	/* TODO: analyse the cause of the return and take appropriate steps */
	if (task->state == starpu_omp_task_state_terminated)
	{
		free(task->stack);
		task->stack = NULL;
		memset(&task->ctx, 0, sizeof(task->ctx));

		starpu_omp_task_completion_accounting(task);
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
	if (is_implicit)
	{
		task->flags |= STARPU_OMP_TASK_FLAGS_IMPLICIT;
	}
	_starpu_spin_init(&task->lock);
	/* TODO: initialize task->data_env_icvs with proper values */
	memset(&task->data_env_icvs, 0, sizeof(task->data_env_icvs));
	if (is_implicit)
	{
	  /* TODO: initialize task->implicit_task_icvs with proper values */
		memset(&task->implicit_task_icvs, 0, sizeof(task->implicit_task_icvs));
	}

	if (owner_region->level > 0)
	{
		STARPU_ASSERT(owner_region->owner_device->icvs.stacksize_var > 0);
		task->stacksize = owner_region->owner_device->icvs.stacksize_var;
	}

	return task;
}

static void destroy_omp_task_struct(struct starpu_omp_task *task)
{
	STARPU_ASSERT(task->state == starpu_omp_task_state_terminated || (task->state == starpu_omp_task_state_zombie && task->child_task_count == 0) || task->state == starpu_omp_task_state_target);
	if (task->state == starpu_omp_task_state_target)
	{
		starpu_omp_task_completion_accounting(task);
	}
	STARPU_ASSERT(task->nested_region == NULL);
	STARPU_ASSERT(task->starpu_task == NULL);
	STARPU_ASSERT(task->stack == NULL);
	_starpu_spin_destroy(&task->lock);
	memset(task, 0, sizeof(*task));
	starpu_omp_task_delete(task);
}

/*
 * setup the main application thread to handle the possible preemption of the initial task
 */
static int omp_initial_thread_setup(void)
{
	struct starpu_omp_thread *initial_thread = _global_state.initial_thread;
	struct starpu_omp_task *initial_task = _global_state.initial_task;
	/* .current_task */
	initial_thread->current_task = initial_task;
	/* .owner_region already set in create_omp_thread_struct */
	/* .initial_thread_stack */
	_STARPU_MALLOC(initial_thread->initial_thread_stack, _STARPU_INITIAL_THREAD_STACKSIZE);
	if (initial_thread->initial_thread_stack == NULL)
		_STARPU_ERROR("memory allocation failed");
	/* .ctx */
	getcontext(&initial_thread->ctx);
	/*
	 * we do not use uc_link, the initial thread always should give hand back to the initial task
	 */
	initial_thread->ctx.uc_link          = NULL;
	initial_thread->ctx.uc_stack.ss_sp   = initial_thread->initial_thread_stack;
	initial_thread->ctx.uc_stack.ss_size = _STARPU_INITIAL_THREAD_STACKSIZE;
	initial_thread->initial_thread_stack_vg_id = VALGRIND_STACK_REGISTER(initial_thread->initial_thread_stack, initial_thread->initial_thread_stack+_STARPU_INITIAL_THREAD_STACKSIZE);
	makecontext(&initial_thread->ctx, omp_initial_thread_func, 0);
	/* .starpu_driver */
	/*
	 * we configure starpu to not launch CPU worker 0
	 * because we will use the main thread to play the role of worker 0
	 */
	struct starpu_conf omp_starpu_conf;
	int ret = starpu_conf_init(&omp_starpu_conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_conf_init");
	initial_thread->starpu_driver.type = STARPU_CPU_WORKER;
	initial_thread->starpu_driver.id.cpu_id = 0;
	omp_starpu_conf.not_launched_drivers = &initial_thread->starpu_driver;
	omp_starpu_conf.n_not_launched_drivers = 1;
	/* we are now ready to start StarPU */
	ret = starpu_init(&omp_starpu_conf);
	int check = _starpu_omp_environment_check();
	if (check == 0)
	{
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
		ret = starpu_driver_init(&initial_thread->starpu_driver);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_driver_init");
		_starpu_omp_set_task(initial_task);

		_global_state.nb_starpu_cpu_workers = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
		_STARPU_MALLOC(_global_state.starpu_cpu_worker_ids, _global_state.nb_starpu_cpu_workers * sizeof(int));
		if (_global_state.starpu_cpu_worker_ids == NULL)
			_STARPU_ERROR("memory allocation failed");
		unsigned n = starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, _global_state.starpu_cpu_worker_ids, _global_state.nb_starpu_cpu_workers);
		STARPU_ASSERT(n == _global_state.nb_starpu_cpu_workers);
		initial_thread->worker = _starpu_get_worker_struct(_global_state.starpu_cpu_worker_ids[0]);
		STARPU_ASSERT(initial_thread->worker);
		STARPU_ASSERT(initial_thread->worker->arch == STARPU_CPU_WORKER);
		_starpu_omp_set_thread(initial_thread);
		register_thread_worker(initial_thread);
	}
	return check;
}

static void omp_initial_thread_exit()
{
	struct starpu_omp_thread *initial_thread = _global_state.initial_thread;
	int ret = starpu_driver_deinit(&initial_thread->starpu_driver);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_driver_deinit");
	memset(&initial_thread->starpu_driver, 0, sizeof (initial_thread->starpu_driver));
	/* the driver for the main thread is now de-inited, we can shutdown Starpu */
	starpu_shutdown();
	free(_global_state.starpu_cpu_worker_ids);
	_global_state.starpu_cpu_worker_ids = NULL;
	_global_state.nb_starpu_cpu_workers = 0;
	VALGRIND_STACK_DEREGISTER(initial_thread->initial_thread_stack_vg_id);
	free(initial_thread->initial_thread_stack);
	initial_thread->initial_thread_stack = NULL;
	memset(&initial_thread->ctx, 0, sizeof (initial_thread->ctx));
	initial_thread->current_task = NULL;
}

static int omp_initial_region_setup(void)
{
	int ret = omp_initial_thread_setup();
	if (ret != 0) return ret;

	const int max_active_levels = _starpu_omp_initial_icv_values->max_active_levels_var;
	const int max_threads = (int)starpu_cpu_worker_get_count();

	/* implementation specific initial ICV values override */
	if (_starpu_omp_initial_icv_values->nthreads_var[0] == 0)
	{
		_starpu_omp_initial_icv_values->nthreads_var[0] = max_threads;
		_starpu_omp_initial_icv_values->nthreads_var[1] = 0;
	}
	else
	{
		int i;
		for (i = 0; i < max_active_levels; i++)
		{
			if (_starpu_omp_initial_icv_values->nthreads_var[i] == 0)
				break;
			if (_starpu_omp_initial_icv_values->nthreads_var[i] > max_threads)
			{
				_starpu_omp_initial_icv_values->nthreads_var[i] = max_threads;
			}
		}
	}
	_starpu_omp_initial_icv_values->dyn_var = 0;
	_starpu_omp_initial_icv_values->nest_var = 0;

	_global_state.initial_device->icvs.max_active_levels_var = max_active_levels;
	_global_state.initial_device->icvs.def_sched_var = _starpu_omp_initial_icv_values->def_sched_var;
	_global_state.initial_device->icvs.def_sched_chunk_var = _starpu_omp_initial_icv_values->def_sched_chunk_var;
	_global_state.initial_device->icvs.stacksize_var = _starpu_omp_initial_icv_values->stacksize_var;
	_global_state.initial_device->icvs.wait_policy_var = _starpu_omp_initial_icv_values->wait_policy_var;

	_global_state.initial_region->master_thread = _global_state.initial_thread;
	_global_state.initial_region->nb_threads++;
	_global_state.initial_region->icvs.dyn_var = _starpu_omp_initial_icv_values->dyn_var;
	_global_state.initial_region->icvs.nest_var = _starpu_omp_initial_icv_values->nest_var;
	if (_starpu_omp_initial_icv_values->nthreads_var[1] != 0)
	{
		_STARPU_MALLOC(_global_state.initial_region->icvs.nthreads_var, (1+max_active_levels-_global_state.initial_region->level) * sizeof(*_global_state.initial_region->icvs.nthreads_var));
		int i,j;
		for (i = _global_state.initial_region->level, j = 0; i < max_active_levels; i++, j++)
		{
			_global_state.initial_region->icvs.nthreads_var[j] = _starpu_omp_initial_icv_values->nthreads_var[j];
		}
		_global_state.initial_region->icvs.nthreads_var[j] = 0;
	}
	else
	{
		_STARPU_MALLOC(_global_state.initial_region->icvs.nthreads_var, 2 * sizeof(*_global_state.initial_region->icvs.nthreads_var));
		_global_state.initial_region->icvs.nthreads_var[0] = _starpu_omp_initial_icv_values->nthreads_var[0];
		_global_state.initial_region->icvs.nthreads_var[1] = 0;
	}

	if (_starpu_omp_initial_icv_values->bind_var[1] != starpu_omp_proc_bind_undefined)
	{
		_STARPU_MALLOC(_global_state.initial_region->icvs.bind_var, (1+max_active_levels-_global_state.initial_region->level) * sizeof(*_global_state.initial_region->icvs.bind_var));
		int i,j;
		for (i = _global_state.initial_region->level, j = 0; i < max_active_levels; i++, j++)
		{
			_global_state.initial_region->icvs.bind_var[j] = _starpu_omp_initial_icv_values->bind_var[j];
		}
		_global_state.initial_region->icvs.bind_var[j] = starpu_omp_proc_bind_undefined;
	}
	else
	{
		_STARPU_MALLOC(_global_state.initial_region->icvs.bind_var, 2 * sizeof(*_global_state.initial_region->icvs.bind_var));
		_global_state.initial_region->icvs.bind_var[0] = _starpu_omp_initial_icv_values->bind_var[0];
		_global_state.initial_region->icvs.bind_var[1] = starpu_omp_proc_bind_undefined;
	}
	_global_state.initial_region->icvs.thread_limit_var = _starpu_omp_initial_icv_values->thread_limit_var;
	_global_state.initial_region->icvs.active_levels_var = 0;
	_global_state.initial_region->icvs.levels_var = 0;
	_global_state.initial_region->icvs.run_sched_var = _starpu_omp_initial_icv_values->run_sched_var;
	_global_state.initial_region->icvs.run_sched_chunk_var = _starpu_omp_initial_icv_values->run_sched_chunk_var;
	_global_state.initial_region->icvs.default_device_var = _starpu_omp_initial_icv_values->default_device_var;
	_global_state.initial_region->icvs.max_task_priority_var = _starpu_omp_initial_icv_values->max_task_priority_var;
	_global_state.initial_region->implicit_task_array = &_global_state.initial_task;
	return 0;
}

static void omp_initial_region_exit(void)
{
	omp_initial_thread_exit();
	_global_state.initial_task->state = starpu_omp_task_state_terminated;
	_global_state.initial_region->implicit_task_array = NULL;
	_global_state.initial_region->master_thread = NULL;
	free(_global_state.initial_region->icvs.nthreads_var);
	free(_global_state.initial_region->icvs.bind_var);
	_global_state.initial_region->nb_threads--;
}

/*
 * If StarPU was compiled with --enable-openmp, but the OpenMP runtime support
 * is not in use, starpu_init() may have been called directly instead of
 * through starpu_omp_init(). However, some starpu_omp functions may be still
 * be called such as _starpu_omp_get_task(). So let's setup a basic environment
 * for them.
 */
void _starpu_omp_dummy_init(void)
{
	if (_starpu_omp_global_state != &_global_state)
	{
		STARPU_PTHREAD_KEY_CREATE(&omp_thread_key, NULL);
		STARPU_PTHREAD_KEY_CREATE(&omp_task_key, NULL);
	}
}

/*
 * Free data structures allocated by _starpu_omp_dummy_init().
 */
void _starpu_omp_dummy_shutdown(void)
{
	if (_starpu_omp_global_state != &_global_state)
	{
		STARPU_PTHREAD_KEY_DELETE(omp_thread_key);
		STARPU_PTHREAD_KEY_DELETE(omp_task_key);
	}
}

/*
 * Entry point to be called by the OpenMP runtime constructor
 */
int starpu_omp_init(void)
{
#ifdef STARPU_SIMGRID
	/* XXX: ideally we'd pass the real argc/argv.  */
	/* We have to tell simgrid to avoid cleaning up at exit, since that's before our destructor :/ */
#  if SIMGRID_VERSION >= 32300
	char *argv[] = { "program", "--cfg=debug/clean-atexit:0", NULL };
#  else
	char *argv[] = { "program", "--cfg=clean-atexit:0", NULL };
#  endif
	int argc = sizeof(argv) / sizeof(argv[0]) - 1;
	char **_argv = argv;
	/* Initialize simgrid before anything else.  */
	_starpu_simgrid_init_early(&argc, &_argv);
#endif

	_starpu_omp_global_state = &_global_state;

	STARPU_PTHREAD_KEY_CREATE(&omp_thread_key, NULL);
	STARPU_PTHREAD_KEY_CREATE(&omp_task_key, NULL);
	_global_state.initial_device = create_omp_device_struct();
	_global_state.initial_region = create_omp_region_struct(NULL, _global_state.initial_device);
	_global_state.initial_thread = create_omp_thread_struct(_global_state.initial_region);
	_global_state.initial_task = create_omp_task_struct(NULL,
			_global_state.initial_thread, _global_state.initial_region, 1);
	_global_state.default_critical = create_omp_critical_struct();
        _global_state.default_arbiter = starpu_arbiter_create();
	_global_state.named_criticals = NULL;
	_starpu_spin_init(&_global_state.named_criticals_lock);
	_global_state.hash_workers = NULL;
	_starpu_spin_init(&_global_state.hash_workers_lock);

	_starpu_omp_environment_init();
	_global_state.icvs.cancel_var = _starpu_omp_initial_icv_values->cancel_var;
	_global_state.environment_valid = omp_initial_region_setup();

	/* init clock reference for starpu_omp_get_wtick */
	_starpu_omp_clock_ref = starpu_timing_now();

	return _global_state.environment_valid;
}

void starpu_omp_shutdown(void)
{
	if (_global_state.environment_valid != 0) return;

	omp_initial_region_exit();
	/* TODO: free ICV variables */
	/* TODO: free task/thread/region/device structures */
	destroy_omp_task_struct(_global_state.initial_task);
	_global_state.initial_task = NULL;
	_global_state.initial_thread = NULL;
	destroy_omp_region_struct(_global_state.initial_region);
	_global_state.initial_region = NULL;
	destroy_omp_device_struct(_global_state.initial_device);
	_global_state.initial_device = NULL;
	destroy_omp_critical_struct(_global_state.default_critical);
	_global_state.default_critical = NULL;
        starpu_arbiter_destroy(_global_state.default_arbiter);
        _global_state.default_arbiter = NULL;
	_starpu_spin_lock(&_global_state.named_criticals_lock);
	{
		struct starpu_omp_critical *critical=NULL, *tmp=NULL;
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
		struct starpu_omp_thread *thread=NULL, *tmp=NULL;
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
	_starpu_omp_environment_exit();
	STARPU_PTHREAD_KEY_DELETE(omp_task_key);
	STARPU_PTHREAD_KEY_DELETE(omp_thread_key);
#ifdef STARPU_SIMGRID
	_starpu_simgrid_deinit_late();
#endif
}

static void implicit_task__destroy_callback(void *_task)
{
	struct starpu_omp_task *task = _task;
	destroy_omp_task_struct(task);
}

void starpu_omp_parallel_region(const struct starpu_omp_parallel_region_attr *attr)
{
	struct starpu_omp_thread *master_thread = _starpu_omp_get_thread();
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_region *generating_region = task->owner_region;
	const int max_active_levels = generating_region->owner_device->icvs.max_active_levels_var;
	struct starpu_omp_region *new_region =
		create_omp_region_struct(generating_region, _global_state.initial_device);
	int ret;
	int nb_threads = 1;

	/* TODO: for now, nested parallel sections are not supported, thus we
	 * open an active parallel section only if the generating region is the
	 * initial region */
	if (attr->if_clause != 0)
	{
		const int max_threads = (int)starpu_cpu_worker_get_count();
		if (attr->num_threads > 0)
		{
			nb_threads = attr->num_threads;
		}
		else
		{
			nb_threads = generating_region->icvs.nthreads_var[0];
		}
		if (nb_threads > max_threads)
		{
			nb_threads = max_threads;
		}
		if (nb_threads > 1 && generating_region->icvs.active_levels_var+1 > max_active_levels)
		{
			nb_threads = 1;
		}
	}
	STARPU_ASSERT(nb_threads > 0);

	new_region->icvs.dyn_var = generating_region->icvs.dyn_var;
	new_region->icvs.nest_var = generating_region->icvs.nest_var;
	/* the nthreads_var and bind_var arrays do not hold more than
	 * max_active_levels entries at most, even if some in-between levels
	 * are inactive */
	if (new_region->level < max_active_levels)
	{
		if (generating_region->icvs.nthreads_var[1] != 0)
		{
			_STARPU_MALLOC(new_region->icvs.nthreads_var, (1+max_active_levels-new_region->level) * sizeof(*new_region->icvs.nthreads_var));
			int i,j;
			for (i = new_region->level, j = 0; i < max_active_levels; i++, j++)
			{
				new_region->icvs.nthreads_var[j] = generating_region->icvs.nthreads_var[j+1];
			}
			new_region->icvs.nthreads_var[j] = 0;
		}
		else
		{
			_STARPU_MALLOC(new_region->icvs.nthreads_var, 2 * sizeof(*new_region->icvs.nthreads_var));
			new_region->icvs.nthreads_var[0] = generating_region->icvs.nthreads_var[0];
			new_region->icvs.nthreads_var[1] = 0;
		}

		if (generating_region->icvs.bind_var[1] != starpu_omp_proc_bind_undefined)
		{
			_STARPU_MALLOC(new_region->icvs.bind_var, (1+max_active_levels-new_region->level) * sizeof(*new_region->icvs.bind_var));
			int i,j;
			for (i = new_region->level, j = 0; i < max_active_levels; i++, j++)
			{
				new_region->icvs.bind_var[j] = generating_region->icvs.bind_var[j+1];
			}
			new_region->icvs.bind_var[j] = starpu_omp_proc_bind_undefined;
		}
		else
		{
			_STARPU_MALLOC(new_region->icvs.bind_var, 2 * sizeof(*new_region->icvs.bind_var));
			new_region->icvs.bind_var[0] = generating_region->icvs.bind_var[0];
			new_region->icvs.bind_var[1] = starpu_omp_proc_bind_undefined;
		}
	}
	else
	{
		_STARPU_MALLOC(new_region->icvs.nthreads_var, sizeof(*new_region->icvs.nthreads_var));
		new_region->icvs.nthreads_var[0] = generating_region->icvs.nthreads_var[0];

		_STARPU_MALLOC(new_region->icvs.bind_var, sizeof(*new_region->icvs.bind_var));
		new_region->icvs.bind_var[0] = generating_region->icvs.bind_var[0];
	}
	new_region->icvs.thread_limit_var = generating_region->icvs.thread_limit_var;
	new_region->icvs.active_levels_var = (nb_threads > 1)?generating_region->icvs.active_levels_var+1:generating_region->icvs.active_levels_var;
	new_region->icvs.levels_var = generating_region->icvs.levels_var+1;
	new_region->icvs.run_sched_var = generating_region->icvs.run_sched_var;
	new_region->icvs.run_sched_chunk_var = generating_region->icvs.run_sched_chunk_var;
	new_region->icvs.default_device_var = generating_region->icvs.default_device_var;
	new_region->icvs.max_task_priority_var = generating_region->icvs.max_task_priority_var;
	_STARPU_CALLOC(new_region->implicit_task_array, nb_threads, sizeof(*new_region->implicit_task_array));

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

			/* TODO: use a less arbitrary thread/worker mapping scheme */
			if (generating_region->level == 0)
			{
				struct _starpu_worker *worker = _starpu_get_worker_struct(_global_state.starpu_cpu_worker_ids[i]);
				new_thread = get_worker_thread(worker);
				if (new_thread == NULL)
				{
					new_thread = create_omp_thread_struct(new_region);
					new_thread->worker = _starpu_get_worker_struct(_global_state.starpu_cpu_worker_ids[i]);
					register_thread_worker(new_thread);
				}
			}
			else
			{
				new_thread = master_thread;
			}
			starpu_omp_thread_list_push_back(&new_region->thread_list, new_thread);
		}

		struct starpu_omp_task *new_task = create_omp_task_struct(task, new_thread, new_region, 1);
		new_task->rank = new_region->nb_threads;
		new_region->nb_threads++;
		new_region->implicit_task_array[i] = new_task;

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
	for (i = 0; i < nb_threads; i++)
	{
		struct starpu_omp_task * implicit_task = new_region->implicit_task_array[i];
		implicit_task->cl = attr->cl;
		/*
		 * save pointer to the regions user function from the parallel region codelet
		 *
		 * TODO: add support for multiple/heterogeneous implementations
		 */
		implicit_task->cpu_f = implicit_task->cl.cpu_funcs[0];

		/*
		 * plug the task wrapper into the parallel region codelet instead, to support task preemption
		 */
		implicit_task->cl.cpu_funcs[0] = starpu_omp_implicit_task_exec;

		implicit_task->starpu_task = starpu_task_create();
		_starpu_task_set_omp_cleanup_callback(implicit_task->starpu_task, implicit_task__destroy_callback, implicit_task);
		implicit_task->starpu_task->cl = &implicit_task->cl;
		{
			int j;
			for (j = 0; j < implicit_task->cl.nbuffers; j++)
			{
				implicit_task->starpu_task->handles[j] = attr->handles[j];
			}
		}
		implicit_task->starpu_task->cl_arg = attr->cl_arg;
		implicit_task->starpu_task->cl_arg_size = attr->cl_arg_size;
		implicit_task->starpu_task->cl_arg_free = attr->cl_arg_free;
		implicit_task->starpu_task->omp_task = implicit_task;
		implicit_task->starpu_task->workerid = implicit_task->owner_thread->worker->workerid;
		implicit_task->starpu_task->execute_on_a_specific_worker = 1;
		starpu_task_declare_deps_array(new_region->continuation_starpu_task, 1, &implicit_task->starpu_task);
	}

	attr = NULL;

	/*
	 * submit all the region implicit starpu tasks
	 */
	for (i = 0; i < nb_threads; i++)
	{
		struct starpu_omp_task * implicit_task = new_region->implicit_task_array[i];
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
			starpu_omp_thread_list_pop_front(&new_region->thread_list);
			/* TODO: cleanup unused threads */
		}
		new_region->nb_threads--;
	}
	/* implicit tasks will be freed in implicit_task__destroy_callback() */
	free(new_region->implicit_task_array);
	STARPU_ASSERT(new_region->nb_threads == 0);
	task->nested_region = NULL;
	free(new_region->icvs.bind_var);
	free(new_region->icvs.nthreads_var);
	destroy_omp_region_struct(new_region);
}

static void wake_up_barrier(struct starpu_omp_region *parallel_region)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	int i;
	for (i = 0; i < parallel_region->nb_threads; i++)
	{
		struct starpu_omp_task * implicit_task = parallel_region->implicit_task_array[i];
		if (implicit_task == task)
			continue;
		weak_task_lock(implicit_task);
		STARPU_ASSERT(implicit_task->wait_on & starpu_omp_task_wait_on_barrier);
		implicit_task->wait_on &= ~starpu_omp_task_wait_on_barrier;
		wake_up_and_unlock_task(implicit_task);
	}
}

void starpu_omp_barrier(void)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	/* Assume barriers are performed in by the implicit tasks of a parallel_region */
	STARPU_ASSERT(task->flags & STARPU_OMP_TASK_FLAGS_IMPLICIT);
	struct starpu_omp_region *parallel_region = task->owner_region;
	_starpu_spin_lock(&task->lock);
	int inc_barrier_count = STARPU_ATOMIC_ADD(&parallel_region->barrier_count, 1);

	if (inc_barrier_count == parallel_region->nb_threads)
	{
		/* last task reaching the barrier */
		_starpu_spin_lock(&parallel_region->lock);
		ANNOTATE_HAPPENS_AFTER(&parallel_region->barrier_count);
		ANNOTATE_HAPPENS_BEFORE_FORGET_ALL(&parallel_region->barrier_count);
		parallel_region->barrier_count = 0;
		ANNOTATE_HAPPENS_AFTER(&parallel_region->barrier_count);
		ANNOTATE_HAPPENS_BEFORE_FORGET_ALL(&parallel_region->barrier_count);
		if (parallel_region->bound_explicit_task_count > 0)
		{
			task->wait_on |= starpu_omp_task_wait_on_region_tasks;
			parallel_region->waiting_task = task;
			task->transaction_pending = 1;
			_starpu_spin_unlock(&parallel_region->lock);
			_starpu_spin_unlock(&task->lock);
			_starpu_task_prepare_for_continuation_ext(0, transaction_callback, task);
			starpu_omp_task_preempt();
		}
		else
		{
			_starpu_spin_unlock(&parallel_region->lock);
			_starpu_spin_unlock(&task->lock);
		}
		wake_up_barrier(parallel_region);
	}
	else
	{
		ANNOTATE_HAPPENS_BEFORE(&parallel_region->barrier_count);
		/* not the last task reaching the barrier
		 * . prepare for conditional continuation
		 * . sleep
		 */

		task->wait_on |= starpu_omp_task_wait_on_barrier;
		task->transaction_pending = 1;
		_starpu_spin_unlock(&task->lock);
		_starpu_task_prepare_for_continuation_ext(0, transaction_callback, task);
		starpu_omp_task_preempt();
		STARPU_ASSERT(task->child_task_count == 0);
	}
}

void starpu_omp_master(void (*f)(void *arg), void *arg)
{
	if (starpu_omp_master_inline())
		f(arg);
}

/* variant of omp_master for inlined code
 * return !0 for the task that should perform the master section
 * return  0 for the tasks that should not perform the master section */
int starpu_omp_master_inline(void)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_thread *thread = _starpu_omp_get_thread();
	/* Assume master is performed in by the implicit tasks of a region */
	STARPU_ASSERT(task->flags & STARPU_OMP_TASK_FLAGS_IMPLICIT);
	struct starpu_omp_region *region = task->owner_region;

	return thread == region->master_thread;
}

void starpu_omp_single(void (*f)(void *arg), void *arg, int nowait)
{
	if (starpu_omp_single_inline())
		f(arg);
	if (!nowait)
		starpu_omp_barrier();
}

/* variant of omp_single for inlined code
 * return !0 for the task that should perform the single section
 * return  0 for the tasks that should not perform the single section
 * wait/nowait should be handled directly by the calling code using starpu_omp_barrier */
int starpu_omp_single_inline(void)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	/* Assume singles are performed in by the implicit tasks of a region */
	STARPU_ASSERT(task->flags & STARPU_OMP_TASK_FLAGS_IMPLICIT);
	struct starpu_omp_region *region = task->owner_region;
	int first = STARPU_BOOL_COMPARE_AND_SWAP(&region->single_id, task->single_id, task->single_id+1);
	task->single_id++;

	return first;
}

void starpu_omp_single_copyprivate(void (*f)(void *arg, void *data, unsigned long long data_size), void *arg, void *data, unsigned long long data_size)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_region *region = task->owner_region;
	int first = starpu_omp_single_inline();

	if (first)
	{
		region->copy_private_data = data;
		f(arg, data, data_size);
	}
	starpu_omp_barrier();
	if (!first)
		memcpy(data, region->copy_private_data, data_size);
	starpu_omp_barrier();
}

void *starpu_omp_single_copyprivate_inline_begin(void *data)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_region *region = task->owner_region;
	int first = starpu_omp_single_inline();

	if (first)
	{
		task->single_first = 1;
		region->copy_private_data = data;
		return NULL;
	}

	starpu_omp_barrier();
	return region->copy_private_data;
}

void starpu_omp_single_copyprivate_inline_end(void)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	/* Assume singles are performed in by the implicit tasks of a region */
	STARPU_ASSERT(task->flags & STARPU_OMP_TASK_FLAGS_IMPLICIT);
	if (task->single_first)
	{
		task->single_first = 0;
		starpu_omp_barrier();
	}
	starpu_omp_barrier();
}

void starpu_omp_critical(void (*f)(void *arg), void *arg, const char *name)
{
	starpu_omp_critical_inline_begin(name);
	f(arg);
	starpu_omp_critical_inline_end(name);
}

void starpu_omp_critical_inline_begin(const char *name)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
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
		task->transaction_pending = 1;
		link.task = task;
		link.next = critical->contention_list_head;
		critical->contention_list_head = &link;
		_starpu_spin_unlock(&task->lock);
		_starpu_spin_unlock(&critical->lock);
		_starpu_task_prepare_for_continuation_ext(0, transaction_callback, task);
		starpu_omp_task_preempt();

		/* re-acquire the spin lock */
		_starpu_spin_lock(&critical->lock);
	}
	critical->state = 1;
	_starpu_spin_unlock(&critical->lock);
}

void starpu_omp_critical_inline_end(const char *name)
{
	struct starpu_omp_critical *critical = NULL;

	if (name)
	{
		_starpu_spin_lock(&_global_state.named_criticals_lock);
		HASH_FIND_STR(_global_state.named_criticals, name, critical);
		_starpu_spin_unlock(&_global_state.named_criticals_lock);
	}
	else
	{
		critical = _global_state.default_critical;
	}

	STARPU_ASSERT(critical != NULL);
	_starpu_spin_lock(&critical->lock);
	STARPU_ASSERT(critical->state == 1);
	critical->state = 0;
	if (critical->contention_list_head != NULL)
	{
		struct starpu_omp_task *next_task = critical->contention_list_head->task;
		weak_task_lock(next_task);
		critical->contention_list_head = critical->contention_list_head->next;
		STARPU_ASSERT(next_task->wait_on & starpu_omp_task_wait_on_critical);
		next_task->wait_on &= ~starpu_omp_task_wait_on_critical;
		wake_up_and_unlock_task(next_task);
	}
	_starpu_spin_unlock(&critical->lock);
}

static void explicit_task__destroy_callback(void *_task)
{
	struct starpu_omp_task *task = _task;
	STARPU_ASSERT(!(task->flags & STARPU_OMP_TASK_FLAGS_IMPLICIT));
	task->starpu_task->omp_task = NULL;
	task->starpu_task = NULL;
	_starpu_spin_lock(&task->lock);
	if (task->state != starpu_omp_task_state_target)
	{
		STARPU_ASSERT(task->transaction_pending == 1);
		task->transaction_pending = 0;
		if (task->child_task_count != 0)
		{
			task->state = starpu_omp_task_state_zombie;
			_starpu_spin_unlock(&task->lock);
			return;
		}
	}
	_starpu_spin_unlock(&task->lock);
	destroy_omp_task_struct(task);
}

void starpu_omp_task_region(const struct starpu_omp_task_region_attr *attr)
{
	struct starpu_omp_task *generating_task = _starpu_omp_get_task();
	struct starpu_omp_region *parallel_region = generating_task->owner_region;
	int is_undeferred = 0;
	int is_final = 0;
	int is_included = 0;
	int is_merged = 0;
	int ret;

	if (generating_task == _global_state.initial_task)
	{
		is_undeferred = 1;
		is_final = 1;
		is_included = 1;
	}
	else
	{
		if (!attr->if_clause)
		{
			is_undeferred = 1;
		}
		if (generating_task->flags & STARPU_OMP_TASK_FLAGS_FINAL)
		{
			is_final = 1;
			is_included = 1;
		}
		else if (attr->final_clause)
		{
			is_final = 1;
		}
		if (is_included)
		{
			is_undeferred = 1;
		}
		if ((is_undeferred || is_included) & attr->mergeable_clause)
		{
			is_merged = 1;
		}
	}
	if (is_merged || is_included)
	{
		if (is_included)
		{
			/* TODO: backup current ICVs and setup new ICVs for the included task */
		}
		int i;
		unsigned n = attr->cl.nbuffers;
		if (n == 0)
			n = 1;
		void *data_interfaces[n];
		for (i = 0; i < attr->cl.nbuffers; i++)
		{
			starpu_data_handle_t handle = attr->handles[i];
			ret = starpu_data_acquire(handle, attr->cl.modes[i]);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
			data_interfaces[i] = starpu_data_get_interface_on_node(handle, handle->home_node);
		}
		void (*f)(void **starpu_buffers, void *starpu_cl_arg) = attr->cl.cpu_funcs[0];
		f(data_interfaces, attr->cl_arg);
		for (i = 0; i < attr->cl.nbuffers; i++)
		{
			starpu_data_release(attr->handles[i]);
		}
		if (attr->cl_arg_free)
		{
			free(attr->cl_arg);
		}
		if (is_included)
		{
			/* TODO: restore backuped ICVs */
		}
	}
	else
	{
		struct starpu_omp_task *generated_task =
			create_omp_task_struct(generating_task, NULL, parallel_region, 0);
		generated_task->cl = attr->cl;
		if (attr->untied_clause)
		{
			generated_task->flags |= STARPU_OMP_TASK_FLAGS_UNTIED;
		}
		if (is_final)
		{
			generated_task->flags |= STARPU_OMP_TASK_FLAGS_FINAL;
		}
		if (is_undeferred)
		{
			generated_task->flags |= STARPU_OMP_TASK_FLAGS_UNDEFERRED;
		}
      // XXX taskgroup exist
      if (!attr->nogroup_clause)
      {
         generated_task->task_group = generating_task->task_group;
      }
      generated_task->rank = -1;

      /* XXX taskloop attributes */
      generated_task->is_loop = attr->is_loop;
      generated_task->nb_iterations = attr->nb_iterations;
      generated_task->grainsize = attr->grainsize;
      generated_task->chunk = attr->chunk;
      generated_task->begin_i = attr->begin_i;
      generated_task->end_i = attr->end_i;

		/*
		 * save pointer to the regions user function from the task region codelet
		 *
		 * TODO: add support for multiple/heterogeneous implementations
		 */
		if (generated_task->cl.cpu_funcs[0])
		{
			generated_task->cpu_f = generated_task->cl.cpu_funcs[0];

			/*
			 * plug the task wrapper into the task region codelet instead, to support task preemption
			 */
			generated_task->cl.cpu_funcs[0] = starpu_omp_explicit_task_exec;
		}
#ifdef STARPU_USE_CUDA
		if (generated_task->cl.cuda_funcs[0])
		{
			generated_task->cuda_f = generated_task->cl.cuda_funcs[0];
#if 1
			/* we assume for now that Cuda task won't block, thus we don't need
			 * to initialize the StarPU OpenMP Runtime Support context for enabling
			 * continuations on Cuda tasks */
			generated_task->state  = starpu_omp_task_state_target;
#else
			generated_task->cl.cuda_funcs[0] = starpu_omp_explicit_task_exec;
#endif
		}
#endif
#ifdef STARPU_USE_OPENCL
		if (generated_task->cl.opencl_funcs[0])
		{
			generated_task->opencl_f = generated_task->cl.opencl_funcs[0];
#if 1
			/* we assume for now that OpenCL task won't block, thus we don't need
			 * to initialize the StarPU OpenMP Runtime Support context for enabling
			 * continuations on OpenCL tasks */
			generated_task->state  = starpu_omp_task_state_target;
#else
			generated_task->cl.opencl_funcs[0] = starpu_omp_explicit_task_exec;
#endif
		}
#endif
		/* TODO: add other accelerator support */

		generated_task->starpu_task = starpu_task_create();
		generated_task->starpu_task->cl = &generated_task->cl;
		generated_task->starpu_task->cl_arg = attr->cl_arg;
		generated_task->starpu_task->cl_arg_size = attr->cl_arg_size;
		generated_task->starpu_task->cl_arg_free = attr->cl_arg_free;
		generated_task->starpu_task->priority = attr->priority;
		{
			int i;
			for (i = 0; i < generated_task->cl.nbuffers; i++)
			{
				generated_task->starpu_task->handles[i] = attr->handles[i];
			}
		}
		generated_task->starpu_task->omp_task = generated_task;
		_starpu_task_set_omp_cleanup_callback(generated_task->starpu_task, explicit_task__destroy_callback, generated_task);
		/* if the task is tied, execute_on_a_specific_worker will be changed to 1
		 * upon the first preemption of the generated task, once we know
		 * which worker thread has been selected */
		generated_task->starpu_task->execute_on_a_specific_worker = 0;

		(void)STARPU_ATOMIC_ADD(&generating_task->child_task_count, 1);
		(void)STARPU_ATOMIC_ADD(&parallel_region->bound_explicit_task_count, 1);
		if (generated_task->task_group)
		{
			(void)STARPU_ATOMIC_ADD(&generated_task->task_group->descendent_task_count, 1);
		}

		/* do not use the attribute struct afterward as it may become out of scope */
		attr = NULL;

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

void starpu_omp_taskwait(void)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	_starpu_spin_lock(&task->lock);
	if (task->child_task_count > 0)
	{
		task->wait_on |= starpu_omp_task_wait_on_task_childs;
		task->transaction_pending = 1;
		_starpu_spin_unlock(&task->lock);
		_starpu_task_prepare_for_continuation_ext(0, transaction_callback, task);
		starpu_omp_task_preempt();
		STARPU_ASSERT(task->child_task_count == 0);
	}
	else
	{
		_starpu_spin_unlock(&task->lock);
	}
}

void starpu_omp_taskgroup(void (*f)(void *arg), void *arg)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_task_group task_group;
	task_group.p_previous_task_group = task->task_group;
	task_group.descendent_task_count = 0;
	task_group.leader_task = task;
	task->task_group = &task_group;
	f(arg);
	_starpu_spin_lock(&task->lock);
	if (task_group.descendent_task_count > 0)
	{
		task->wait_on |= starpu_omp_task_wait_on_group;
		task->transaction_pending = 1;
		_starpu_spin_unlock(&task->lock);
		_starpu_task_prepare_for_continuation_ext(0, transaction_callback, task);
		starpu_omp_task_preempt();
		STARPU_ASSERT(task_group.descendent_task_count == 0);
	}
	else
	{
		_starpu_spin_unlock(&task->lock);
	}
	task->task_group = task_group.p_previous_task_group;
}

void starpu_omp_taskgroup_inline_begin(void)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_task_group *p_task_group;
	_STARPU_MALLOC(p_task_group, sizeof(*p_task_group));
	p_task_group->p_previous_task_group = task->task_group;
	p_task_group->descendent_task_count = 0;
	p_task_group->leader_task = task;
	task->task_group = p_task_group;
}

void starpu_omp_taskgroup_inline_end(void)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	_starpu_spin_lock(&task->lock);
	struct starpu_omp_task_group *p_task_group = task->task_group;
	if (p_task_group->descendent_task_count > 0)
	{
		task->wait_on |= starpu_omp_task_wait_on_group;
		task->transaction_pending = 1;
		_starpu_spin_unlock(&task->lock);
		_starpu_task_prepare_for_continuation_ext(0, transaction_callback, task);
		starpu_omp_task_preempt();
		STARPU_ASSERT(p_task_group->descendent_task_count == 0);
	}
	else
	{
		_starpu_spin_unlock(&task->lock);
	}
	task->task_group = p_task_group->p_previous_task_group;
	free(p_task_group);
}

// XXX on work
void starpu_omp_taskloop_inline_begin(struct starpu_omp_task_region_attr *attr)
{
	if (!attr->nogroup_clause)
	{
		starpu_omp_taskgroup_inline_begin();
	}

	int nb_subloop;
	if (attr->num_tasks)
	{
		nb_subloop = attr->num_tasks;
	}
	else if (attr->grainsize)
	{
		nb_subloop = attr->nb_iterations / attr->grainsize;
	}
	else
	{
		nb_subloop = 4;
	}

	attr->is_loop = 1;

	int i;
	int nb_iter_i = attr->nb_iterations / nb_subloop;
	for (i = 0; i < nb_subloop; i++)
	{
		attr->begin_i = nb_iter_i * i;
		attr->end_i = attr->begin_i + nb_iter_i;
		attr->end_i += (i+1 != nb_subloop) ? 0 : (attr->nb_iterations % nb_subloop);
		attr->chunk = attr->end_i - attr->begin_i;
		starpu_omp_task_region(attr);
	}
}

// XXX on work
void starpu_omp_taskloop_inline_end(const struct starpu_omp_task_region_attr *attr)
{
	if (!attr->nogroup_clause)
	{
		starpu_omp_taskgroup_inline_end();
	}
}

static inline void _starpu_omp_for_loop(struct starpu_omp_region *parallel_region, struct starpu_omp_task *task,
		struct starpu_omp_loop *loop, int first_call,
		unsigned long long nb_iterations, unsigned long long chunk, int schedule, int ordered, unsigned long long *_first_i, unsigned long long *_nb_i)
{
	*_nb_i = 0;
	if (schedule == starpu_omp_sched_undefined)
	{
		schedule = parallel_region->owner_device->icvs.def_sched_var;
		chunk = parallel_region->owner_device->icvs.def_sched_chunk_var;
	}
	else if (schedule == starpu_omp_sched_runtime)
	{
		schedule = parallel_region->icvs.run_sched_var;
		chunk = parallel_region->icvs.run_sched_chunk_var;
	}
	STARPU_ASSERT(     schedule == starpu_omp_sched_static
			|| schedule == starpu_omp_sched_dynamic
			|| schedule == starpu_omp_sched_guided
			|| schedule == starpu_omp_sched_auto);
	if (schedule == starpu_omp_sched_auto)
	{
		schedule = starpu_omp_sched_static;
		chunk = 0;
	}
	if (schedule == starpu_omp_sched_static)
	{
		if (chunk > 0)
		{
			if (first_call)
			{
				*_first_i = task->rank * chunk;
			}
			else
			{
				*_first_i += parallel_region->nb_threads * chunk;
			}

			if (*_first_i < nb_iterations)
			{
				if (*_first_i + chunk > nb_iterations)
				{
					*_nb_i = nb_iterations - *_first_i;
				}
				else
				{
					*_nb_i = chunk;
				}
			}
		}
		else
		{
			if (first_call)
			{
				*_nb_i = nb_iterations / parallel_region->nb_threads;
				*_first_i = (unsigned)task->rank * (*_nb_i);
				unsigned long long remainder = nb_iterations % parallel_region->nb_threads;

				if (remainder > 0)
				{
					if ((unsigned)task->rank < remainder)
					{
						(*_nb_i)++;
						*_first_i += (unsigned)task->rank;
					}
					else
					{
						*_first_i += remainder;
					}
				}
			}
		}
	}
	else if (schedule == starpu_omp_sched_dynamic)
	{
		if (chunk == 0)
		{
			chunk = 1;
		}
		if (first_call)
		{
			*_first_i = 0;
		}
		_starpu_spin_lock(&parallel_region->lock);
		if (loop->next_iteration < nb_iterations)
		{
			*_first_i = loop->next_iteration;
			if (*_first_i + chunk > nb_iterations)
			{
				*_nb_i = nb_iterations - *_first_i;
			}
			else
			{
				*_nb_i = chunk;
			}
			loop->next_iteration += *_nb_i;
		}
		_starpu_spin_unlock(&parallel_region->lock);
	}
	else if (schedule == starpu_omp_sched_guided)
	{
		if (chunk == 0)
		{
			chunk = 1;
		}
		if (first_call)
		{
			*_first_i = 0;
		}
		_starpu_spin_lock(&parallel_region->lock);
		if (loop->next_iteration < nb_iterations)
		{
			*_first_i = loop->next_iteration;
			*_nb_i = (nb_iterations - *_first_i)/parallel_region->nb_threads;
			if (*_nb_i < chunk)
			{
				if (*_first_i+chunk > nb_iterations)
				{
					*_nb_i = nb_iterations - *_first_i;
				}
				else
				{
					*_nb_i = chunk;
				}
			}
			loop->next_iteration += *_nb_i;
		}
		_starpu_spin_unlock(&parallel_region->lock);
	}
	if (ordered)
	{
		task->ordered_first_i = *_first_i;
		task->ordered_nb_i = *_nb_i;
	}
}

static inline struct starpu_omp_loop *_starpu_omp_for_get_loop(struct starpu_omp_region *parallel_region, struct starpu_omp_task *task)
{
	struct starpu_omp_loop *loop;
	loop = parallel_region->loop_list;
	while (loop && loop->id != task->loop_id)
	{
		loop = loop->next_loop;
	}
	return loop;
}

static inline struct starpu_omp_loop *_starpu_omp_for_loop_begin(struct starpu_omp_region *parallel_region, struct starpu_omp_task *task,
		int ordered)
{
	struct starpu_omp_loop *loop;
	_starpu_spin_lock(&parallel_region->lock);
	loop = _starpu_omp_for_get_loop(parallel_region, task);
	if (!loop)
	{
		_STARPU_MALLOC(loop, sizeof(*loop));
		loop->id = task->loop_id;
		loop->next_iteration = 0;
		loop->nb_completed_threads = 0;
		loop->next_loop = parallel_region->loop_list;
		parallel_region->loop_list = loop;
		if (ordered)
		{
			loop->ordered_iteration = 0;
			_starpu_spin_init(&loop->ordered_lock);
			condition_init(&loop->ordered_cond);
		}
	}
	_starpu_spin_unlock(&parallel_region->lock);
	return loop;
}
static inline void _starpu_omp_for_loop_end(struct starpu_omp_region *parallel_region, struct starpu_omp_task *task,
		struct starpu_omp_loop *loop, int ordered)
{
	_starpu_spin_lock(&parallel_region->lock);
	loop->nb_completed_threads++;
	if (loop->nb_completed_threads == parallel_region->nb_threads)
	{
		struct starpu_omp_loop **p_loop;
		if (ordered)
		{
			loop->ordered_iteration = 0;
			condition_exit(&loop->ordered_cond);
			_starpu_spin_destroy(&loop->ordered_lock);
		}
		STARPU_ASSERT(loop->next_loop == NULL);
		p_loop = &(parallel_region->loop_list);
		while (*p_loop != loop)
		{
			p_loop = &((*p_loop)->next_loop);
		}
		*p_loop = NULL;
		free(loop);
	}
	_starpu_spin_unlock(&parallel_region->lock);
	task->loop_id++;
}

int starpu_omp_for_inline_first(unsigned long long nb_iterations, unsigned long long chunk, int schedule, int ordered, unsigned long long *_first_i, unsigned long long *_nb_i)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_region *parallel_region = task->owner_region;
	struct starpu_omp_loop *loop = _starpu_omp_for_loop_begin(parallel_region, task, ordered);

	_starpu_omp_for_loop(parallel_region, task, loop, 1, nb_iterations, chunk, schedule, ordered, _first_i, _nb_i);
	if (*_nb_i == 0)
	{
		_starpu_omp_for_loop_end(parallel_region, task, loop, ordered);
	}
	return *_nb_i != 0;
}

int starpu_omp_for_inline_next(unsigned long long nb_iterations, unsigned long long chunk, int schedule, int ordered, unsigned long long *_first_i, unsigned long long *_nb_i)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_region *parallel_region = task->owner_region;
	struct starpu_omp_loop *loop = _starpu_omp_for_loop_begin(parallel_region, task, ordered);

	_starpu_omp_for_loop(parallel_region, task, loop, 0, nb_iterations, chunk, schedule, ordered, _first_i, _nb_i);
	if (*_nb_i == 0)
	{
		_starpu_omp_for_loop_end(parallel_region, task, loop, ordered);
	}
	return *_nb_i != 0;
}

int starpu_omp_for_inline_first_alt(unsigned long long nb_iterations, unsigned long long chunk, int schedule, int ordered, unsigned long long *_begin_i, unsigned long long *_end_i)
{
	unsigned long long nb_i;
	int end = starpu_omp_for_inline_first(nb_iterations, chunk, schedule, ordered, _begin_i, &nb_i);
	*_end_i = *_begin_i + nb_i;
	return end;
}

int starpu_omp_for_inline_next_alt(unsigned long long nb_iterations, unsigned long long chunk, int schedule, int ordered, unsigned long long *_begin_i, unsigned long long *_end_i)
{
	unsigned long long nb_i;
	int end = starpu_omp_for_inline_next(nb_iterations, chunk, schedule, ordered, _begin_i, &nb_i);
	*_end_i = *_begin_i + nb_i;
	return end;
}

void starpu_omp_for(void (*f)(unsigned long long _first_i, unsigned long long _nb_i, void *arg), void *arg, unsigned long long nb_iterations, unsigned long long chunk, int schedule, int ordered, int nowait)
{
	unsigned long long _first_i = 0;
	unsigned long long _nb_i = 0;
	if (starpu_omp_for_inline_first(nb_iterations, chunk, schedule, ordered, &_first_i, &_nb_i))
	{
		do
		{
			f(_first_i, _nb_i, arg);
		}
		while (starpu_omp_for_inline_next(nb_iterations, chunk, schedule, ordered, &_first_i, &_nb_i));
	}
	if (!nowait)
	{
		starpu_omp_barrier();
	}
}

void starpu_omp_for_alt(void (*f)(unsigned long long _begin_i, unsigned long long _end_i, void *arg), void *arg, unsigned long long nb_iterations, unsigned long long chunk, int schedule, int ordered, int nowait)
{
	unsigned long long _begin_i = 0;
	unsigned long long _end_i = 0;
	if (starpu_omp_for_inline_first_alt(nb_iterations, chunk, schedule, ordered, &_begin_i, &_end_i))
	{
		do
		{
			f(_begin_i, _end_i, arg);
		}
		while (starpu_omp_for_inline_next_alt(nb_iterations, chunk, schedule, ordered, &_begin_i, &_end_i));
	}
	if (!nowait)
	{
		starpu_omp_barrier();
	}
}

void starpu_omp_ordered(void (*f)(void *arg), void *arg)
{
	starpu_omp_ordered_inline_begin();
	f(arg);
	starpu_omp_ordered_inline_end();
}

void starpu_omp_ordered_inline_begin(void)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_region *parallel_region = task->owner_region;
	struct starpu_omp_loop *loop = _starpu_omp_for_get_loop(parallel_region, task);
	unsigned long long i;
	STARPU_ASSERT(task->ordered_nb_i > 0);
	i = task->ordered_first_i;
	task->ordered_first_i++;
	task->ordered_nb_i--;
	_starpu_spin_lock(&loop->ordered_lock);
	while (i != loop->ordered_iteration)
	{
		STARPU_ASSERT(i > loop->ordered_iteration);
		condition_wait(&loop->ordered_cond, &loop->ordered_lock, starpu_omp_task_wait_on_ordered);
	}
}

void starpu_omp_ordered_inline_end(void)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_region *parallel_region = task->owner_region;
	struct starpu_omp_loop *loop = _starpu_omp_for_get_loop(parallel_region, task);

	loop->ordered_iteration++;
	condition_broadcast(&loop->ordered_cond, starpu_omp_task_wait_on_ordered);
	_starpu_spin_unlock(&loop->ordered_lock);
}

static inline struct starpu_omp_sections *_starpu_omp_get_sections(struct starpu_omp_region *parallel_region, struct starpu_omp_task *task)
{
	struct starpu_omp_sections *sections;
	sections = parallel_region->sections_list;
	while (sections && sections->id != task->sections_id)
	{
		sections = sections->next_sections;
	}
	return sections;
}

static inline struct starpu_omp_sections *_starpu_omp_sections_begin(struct starpu_omp_region *parallel_region, struct starpu_omp_task *task)
{
	struct starpu_omp_sections *sections;
	_starpu_spin_lock(&parallel_region->lock);
	sections = _starpu_omp_get_sections(parallel_region, task);
	if (!sections)
	{
		_STARPU_MALLOC(sections, sizeof(*sections));
		sections->id = task->sections_id;
		sections->next_section_num = 0;
		sections->nb_completed_threads = 0;
		sections->next_sections = parallel_region->sections_list;
		parallel_region->sections_list = sections;
	}
	_starpu_spin_unlock(&parallel_region->lock);
	return sections;
}
static inline void _starpu_omp_sections_end(struct starpu_omp_region *parallel_region, struct starpu_omp_task *task,
		struct starpu_omp_sections *sections)
{
	_starpu_spin_lock(&parallel_region->lock);
	sections->nb_completed_threads++;
	if (sections->nb_completed_threads == parallel_region->nb_threads)
	{
		struct starpu_omp_sections **p_sections;
		STARPU_ASSERT(sections->next_sections == NULL);
		p_sections = &(parallel_region->sections_list);
		while (*p_sections != sections)
		{
			p_sections = &((*p_sections)->next_sections);
		}
		*p_sections = NULL;
		free(sections);
	}
	_starpu_spin_unlock(&parallel_region->lock);
	task->sections_id++;
}

void starpu_omp_sections(unsigned long long nb_sections, void (**section_f)(void *arg), void **section_arg, int nowait)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_region *parallel_region = task->owner_region;
	struct starpu_omp_sections *sections = _starpu_omp_sections_begin(parallel_region, task);
	for (;;)
	{
		void (*f)(void *arg) = NULL;
		void *arg = NULL;
		_starpu_spin_lock(&parallel_region->lock);
		if (sections->next_section_num < nb_sections)
		{
			f = section_f[sections->next_section_num];
			arg = section_arg[sections->next_section_num];
			sections->next_section_num ++;
		}
		_starpu_spin_unlock(&parallel_region->lock);
		if (f == NULL)
			break;
		f(arg);
	}
	_starpu_omp_sections_end(parallel_region, task, sections);
	if (!nowait)
	{
		starpu_omp_barrier();
	}
}

void starpu_omp_sections_combined(unsigned long long nb_sections, void (*section_f)(unsigned long long section_num, void *arg), void *section_arg, int nowait)
{
	struct starpu_omp_task *task = _starpu_omp_get_task();
	struct starpu_omp_region *parallel_region = task->owner_region;
	struct starpu_omp_sections *sections = _starpu_omp_sections_begin(parallel_region, task);
	for (;;)
	{
		unsigned long long section_num;
		void *arg = NULL;
		_starpu_spin_lock(&parallel_region->lock);
		if (sections->next_section_num < nb_sections)
		{
			section_num = sections->next_section_num;
			arg = section_arg;
			sections->next_section_num ++;
		}
		else
		{
			_starpu_spin_unlock(&parallel_region->lock);
			break;
		}
		_starpu_spin_unlock(&parallel_region->lock);
		section_f(section_num, arg);
	}
	_starpu_omp_sections_end(parallel_region, task, sections);
	if (!nowait)
	{
		starpu_omp_barrier();
	}
}

static void _starpu_omp_lock_init(void **_internal)
{
	struct _starpu_omp_lock_internal *_lock;

	_STARPU_CALLOC(_lock, 1, sizeof(*_lock));
	_starpu_spin_init(&_lock->lock);
	condition_init(&_lock->cond);
	*_internal = _lock;
}

static void _starpu_omp_lock_destroy(void **_internal)
{
	struct _starpu_omp_lock_internal * const _lock = *_internal;
	STARPU_ASSERT(_lock->state == 0);
	condition_exit(&_lock->cond);
	_starpu_spin_destroy(&_lock->lock);
	memset(_lock, 0, sizeof(*_lock));
	free(_lock);
	*_internal = NULL;
}

static void _starpu_omp_lock_set(void **_internal)
{
	struct _starpu_omp_lock_internal * const _lock = *_internal;
	_starpu_spin_lock(&_lock->lock);
	while (_lock->state != 0)
	{
		condition_wait(&_lock->cond, &_lock->lock, starpu_omp_task_wait_on_lock);
	}
	_lock->state = 1;
	_starpu_spin_unlock(&_lock->lock);
}

static void _starpu_omp_lock_unset(void **_internal)
{
	struct _starpu_omp_lock_internal * const _lock = *_internal;
	_starpu_spin_lock(&_lock->lock);
	STARPU_ASSERT(_lock->state == 1);
	_lock->state = 0;
	condition_broadcast(&_lock->cond, starpu_omp_task_wait_on_lock);
	_starpu_spin_unlock(&_lock->lock);
}

static int _starpu_omp_lock_test(void **_internal)
{
	struct _starpu_omp_lock_internal * const _lock = *_internal;
	int ret = 0;
	_starpu_spin_lock(&_lock->lock);
	if (_lock->state == 0)
	{
		_lock->state = 1;
		ret = 1;
	}
	_starpu_spin_unlock(&_lock->lock);
	return ret;
}

static void _starpu_omp_nest_lock_init(void **_internal)
{
	struct _starpu_omp_nest_lock_internal *_nest_lock;

	_STARPU_CALLOC(_nest_lock, 1, sizeof(*_nest_lock));
	_starpu_spin_init(&_nest_lock->lock);
	condition_init(&_nest_lock->cond);
	*_internal = _nest_lock;
}

static void _starpu_omp_nest_lock_destroy(void **_internal)
{
	struct _starpu_omp_nest_lock_internal * const _nest_lock = *_internal;
	STARPU_ASSERT(_nest_lock->state == 0);
	STARPU_ASSERT(_nest_lock->nesting == 0);
	STARPU_ASSERT(_nest_lock->owner_task == NULL);
	condition_exit(&_nest_lock->cond);
	_starpu_spin_destroy(&_nest_lock->lock);
	memset(_nest_lock, 0, sizeof(*_nest_lock));
	free(_nest_lock);
	*_internal = NULL;
}

static void _starpu_omp_nest_lock_set(void **_internal)
{
	struct _starpu_omp_nest_lock_internal * const _nest_lock = *_internal;
	struct starpu_omp_task * const task = _starpu_omp_get_task();
	_starpu_spin_lock(&_nest_lock->lock);
	if (_nest_lock->owner_task == task)
	{
		STARPU_ASSERT(_nest_lock->state == 1);
		STARPU_ASSERT(_nest_lock->nesting > 0);
		_nest_lock->nesting++;
	}
	else
	{
		while (_nest_lock->state != 0)
		{
			condition_wait(&_nest_lock->cond, &_nest_lock->lock, starpu_omp_task_wait_on_nest_lock);
		}
		STARPU_ASSERT(_nest_lock->nesting == 0);
		STARPU_ASSERT(_nest_lock->owner_task == NULL);
		_nest_lock->state = 1;
		_nest_lock->owner_task = task;
		_nest_lock->nesting = 1;
	}
	_starpu_spin_unlock(&_nest_lock->lock);
}

static void _starpu_omp_nest_lock_unset(void **_internal)
{
	struct _starpu_omp_nest_lock_internal * const _nest_lock = *_internal;
	struct starpu_omp_task * const task = _starpu_omp_get_task();
	_starpu_spin_lock(&_nest_lock->lock);
	STARPU_ASSERT(_nest_lock->owner_task == task);
	STARPU_ASSERT(_nest_lock->state == 1);
	STARPU_ASSERT(_nest_lock->nesting > 0);
	_nest_lock->nesting--;
	if (_nest_lock->nesting == 0)
	{
		_nest_lock->state = 0;
		_nest_lock->owner_task = NULL;
		condition_broadcast(&_nest_lock->cond, starpu_omp_task_wait_on_nest_lock);
	}
	_starpu_spin_unlock(&_nest_lock->lock);
}

static int _starpu_omp_nest_lock_test(void **_internal)
{
	struct _starpu_omp_nest_lock_internal * const _nest_lock = *_internal;
	struct starpu_omp_task * const task = _starpu_omp_get_task();
	int ret = 0;
	_starpu_spin_lock(&_nest_lock->lock);
	if (_nest_lock->state == 0)
	{
		STARPU_ASSERT(_nest_lock->nesting == 0);
		STARPU_ASSERT(_nest_lock->owner_task == NULL);
		_nest_lock->state = 1;
		_nest_lock->owner_task = task;
		_nest_lock->nesting = 1;
		ret = _nest_lock->nesting;
	}
	else if (_nest_lock->owner_task == task)
	{
		STARPU_ASSERT(_nest_lock->state == 1);
		STARPU_ASSERT(_nest_lock->nesting > 0);
		_nest_lock->nesting++;
		ret = _nest_lock->nesting;
	}
	_starpu_spin_unlock(&_nest_lock->lock);
	return ret;
}

void starpu_omp_init_lock (starpu_omp_lock_t *lock)
{
	_starpu_omp_lock_init(&lock->internal);
}

void starpu_omp_destroy_lock (starpu_omp_lock_t *lock)
{
	_starpu_omp_lock_destroy(&lock->internal);
}

void starpu_omp_set_lock (starpu_omp_lock_t *lock)
{
	_starpu_omp_lock_set(&lock->internal);
}

void starpu_omp_unset_lock (starpu_omp_lock_t *lock)
{
	_starpu_omp_lock_unset(&lock->internal);
}

int starpu_omp_test_lock (starpu_omp_lock_t *lock)
{
	return _starpu_omp_lock_test(&lock->internal);
}

void starpu_omp_init_nest_lock (starpu_omp_nest_lock_t *nest_lock)
{
	_starpu_omp_nest_lock_init(&nest_lock->internal);
}

void starpu_omp_destroy_nest_lock (starpu_omp_nest_lock_t *nest_lock)
{
	_starpu_omp_nest_lock_destroy(&nest_lock->internal);
}

void starpu_omp_set_nest_lock (starpu_omp_nest_lock_t *nest_lock)
{
	_starpu_omp_nest_lock_set(&nest_lock->internal);
}

void starpu_omp_unset_nest_lock (starpu_omp_nest_lock_t *nest_lock)
{
	_starpu_omp_nest_lock_unset(&nest_lock->internal);
}

int starpu_omp_test_nest_lock (starpu_omp_nest_lock_t *nest_lock)
{
	return _starpu_omp_nest_lock_test(&nest_lock->internal);
}

void starpu_omp_atomic_fallback_inline_begin(void)
{
	struct starpu_omp_device *device = get_caller_device();
	_starpu_spin_lock(&device->atomic_lock);

}

void starpu_omp_atomic_fallback_inline_end(void)
{
	struct starpu_omp_device *device = get_caller_device();
	_starpu_spin_unlock(&device->atomic_lock);
}

void starpu_omp_vector_annotate(starpu_data_handle_t handle, uint32_t slice_base)
{
	/* FIXME Oli: rather iterate over all nodes? */
	int node = starpu_data_get_home_node(handle);
	if (node < 0 || (starpu_node_get_kind(node) != STARPU_CPU_RAM))
		node = STARPU_MAIN_RAM;
	struct starpu_vector_interface *vector_interface = (struct starpu_vector_interface *)
		starpu_data_get_interface_on_node(handle, node);
	assert(vector_interface->id == STARPU_VECTOR_INTERFACE_ID);
	vector_interface->slice_base = slice_base;
}

struct starpu_arbiter *starpu_omp_get_default_arbiter(void)
{
	return _global_state.default_arbiter;
}

/*
 * restore deprecated diagnostics (-Wdeprecated-declarations)
 */
#pragma GCC diagnostic pop
#endif /* STARPU_OPENMP */
