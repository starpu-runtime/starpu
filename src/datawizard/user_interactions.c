/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2021-2021  Federal University of Rio Grande do Sul (UFRGS)
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

#include <common/config.h>
#include <common/utils.h>
#include <core/task.h>
#include <datawizard/coherency.h>
#include <datawizard/copy_driver.h>
#include <datawizard/write_back.h>
#include <core/dependencies/data_concurrency.h>
#include <core/sched_policy.h>
#include <datawizard/memory_nodes.h>

static void _starpu_data_check_initialized(starpu_data_handle_t handle, enum starpu_data_access_mode mode)
{
	if (((handle->nplans && !handle->nchildren) || handle->siblings)
		&& !(mode & STARPU_NOPLAN))
	{
		_STARPU_RECURSIVE_TASKS_DEBUG("Call access_submit() for %p\n", handle);
		_starpu_data_partition_access_submit(handle, (mode & STARPU_W) != 0, (mode & STARPU_W) != 0 && (mode & STARPU_R) == 0, NULL);
	}

	if (!(mode & STARPU_R))
		return;

	int initialized = handle->initialized;
#ifdef STARPU_RECURSIVE_TASKS
	if (!initialized && _starpu_recursive_task_which_generate_dag() != NULL)
	{
		initialized += _starpu_get_initialized_state_on_parent_task_parent_data(handle, _starpu_recursive_task_which_generate_dag());
	}
#endif
	if (!initialized && handle->init_cl)
	{
		int ret = starpu_task_insert(handle->init_cl,
			STARPU_CL_ARGS_NFREE, handle->init_cl_arg, 0,
			STARPU_W, handle, 0);
		STARPU_ASSERT(ret == 0);
		initialized = 1;
	}
	STARPU_ASSERT_MSG(initialized, "handle %p is not initialized while trying to read it\n", handle);
}

/* Explicitly ask StarPU to allocate room for a piece of data on the specified
 * memory node. */
int starpu_data_request_allocation(starpu_data_handle_t handle, unsigned node)
{
	struct _starpu_data_request *r;

	STARPU_ASSERT(handle);

	_starpu_spin_lock(&handle->header_lock);

	r = _starpu_create_data_request(handle, NULL, &handle->per_node[node], node, STARPU_NONE, 0, NULL, STARPU_PREFETCH, 0, 0, "starpu_data_request_allocation");

	/* we do not increase the refcnt associated to the request since we are
	 * not waiting for its termination */

	_starpu_post_data_request(r);

	_starpu_spin_unlock(&handle->header_lock);

	return 0;
}

struct user_interaction_wrapper
{
	starpu_data_handle_t handle;
	enum starpu_data_access_mode mode;
	int node;
	starpu_pthread_cond_t cond;
	starpu_pthread_mutex_t lock;
	unsigned finished;
	unsigned detached;
	enum starpu_is_prefetch prefetch;
	unsigned async;
	int prio;
	void (*callback_acquired)(void *, int *node, enum starpu_data_access_mode mode);
	void (*callback)(void *);
	void *callback_arg;
	struct starpu_task *pre_sync_task;
	struct starpu_task *post_sync_task;
};

static inline void _starpu_data_acquire_wrapper_init(struct user_interaction_wrapper *wrapper, starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode)
{
	memset(wrapper, 0, sizeof(*wrapper));
	wrapper->handle = handle;
	wrapper->node = node;
	wrapper->mode = mode;
	//wrapper->finished = 0;
	STARPU_PTHREAD_COND_INIT0(&wrapper->cond, NULL);
	STARPU_PTHREAD_MUTEX_INIT0(&wrapper->lock, NULL);
}

/* Called to signal completion of asynchronous data acquisition */
static inline void _starpu_data_acquire_wrapper_finished(struct user_interaction_wrapper *wrapper)
{
	STARPU_PTHREAD_MUTEX_LOCK(&wrapper->lock);
	wrapper->finished = 1;
	STARPU_PTHREAD_COND_SIGNAL(&wrapper->cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&wrapper->lock);
}

/* Called to wait for completion of asynchronous data acquisition */
static inline void _starpu_data_acquire_wrapper_wait(struct user_interaction_wrapper *wrapper)
{
	STARPU_PTHREAD_MUTEX_LOCK(&wrapper->lock);
	while (!wrapper->finished)
		STARPU_PTHREAD_COND_WAIT(&wrapper->cond, &wrapper->lock);
	STARPU_PTHREAD_MUTEX_UNLOCK(&wrapper->lock);
}

static inline void _starpu_data_acquire_wrapper_fini(struct user_interaction_wrapper *wrapper)
{
	STARPU_PTHREAD_COND_DESTROY(&wrapper->cond);
	STARPU_PTHREAD_MUTEX_DESTROY(&wrapper->lock);
}

/* Called when the data acquisition is done, to launch the fetch into target memory */
static inline void _starpu_data_acquire_launch_fetch(struct user_interaction_wrapper *wrapper, int async, void (*callback)(void *), void *callback_arg)
{
	int node = wrapper->node;
	starpu_data_handle_t handle = wrapper->handle;
	struct _starpu_data_replicate *replicate = node >= 0 ? &handle->per_node[node] : NULL;

	int ret = _starpu_fetch_data_on_node(handle, node, replicate, wrapper->mode, wrapper->detached, NULL, wrapper->prefetch, async, callback, callback_arg, wrapper->prio, "_starpu_data_acquire_launch_fetch");
	STARPU_ASSERT(!ret);
}



/*
 *	Non Blocking data request from application
 */


/* Called when fetch is done, call the callback */
static void _starpu_data_acquire_fetch_data_callback(void *arg)
{
	struct user_interaction_wrapper *wrapper = (struct user_interaction_wrapper *) arg;
	starpu_data_handle_t handle = wrapper->handle;

//	_STARPU_DEBUG("Calling acquire callback for task %p(%s) on handle %p\n", wrapper->pre_sync_task, wrapper->pre_sync_task->name, wrapper->handle);
	/* At that moment, the caller holds a reference to the piece of data.
	 * We enqueue the "post" sync task in the list associated to the handle
	 * so that it is submitted by the starpu_data_release
	 * function. */

	_STARPU_RECURSIVE_TASKS_DEBUG("Acquire has post_sync_task ? %p(%s)\n", wrapper->post_sync_task, wrapper->post_sync_task ? wrapper->post_sync_task->name : NULL);
	if (wrapper->post_sync_task)
		_starpu_add_post_sync_tasks(wrapper->post_sync_task, handle);

	wrapper->callback(wrapper->callback_arg);

	_starpu_data_acquire_wrapper_fini(wrapper);
	free(wrapper);
}

/* Called when the data acquisition is done, launch the fetch into target memory */
static void _starpu_data_acquire_continuation_non_blocking(void *arg)
{
	struct user_interaction_wrapper *wrapper = (struct user_interaction_wrapper *) arg;

	if (wrapper->callback_acquired)
		/* This can change the node at will according to the current data situation */
		wrapper->callback_acquired(wrapper->callback_arg, &wrapper->node, wrapper->mode);

//	_STARPU_RECURSIVE_TASKS_DEBUG("Calling acquire for task %p(%s) on handle %p\n", wrapper->pre_sync_task, wrapper->pre_sync_task->name, wrapper->handle);
	_starpu_data_acquire_launch_fetch(arg, 1, _starpu_data_acquire_fetch_data_callback, arg);
}

/* Called when the implicit data dependencies are done, launch the data acquisition */
static void starpu_data_acquire_cb_pre_sync_callback(void *arg)
{
	struct user_interaction_wrapper *wrapper = (struct user_interaction_wrapper *) arg;

//	_STARPU_RECURSIVE_TASKS_DEBUG("Calling callback for task %p(%s) on handle %p\n", wrapper->pre_sync_task, wrapper->pre_sync_task->name, wrapper->handle);
	/*
	 * we try to get the data, if we do not succeed immediately,
	 * we set a callback function that will be executed
	 * automatically when the data is available again, otherwise we
	 * fetch the data directly
	 */
	if (!_starpu_attempt_to_submit_data_request_from_apps(wrapper->handle, wrapper->mode,
			_starpu_data_acquire_continuation_non_blocking, wrapper))
	{
		/* no one has locked this data yet, so we proceed immediately */
		_STARPU_RECURSIVE_TASKS_DEBUG("data %p available on task %p\n", wrapper->handle, wrapper->pre_sync_task);
		_starpu_data_acquire_continuation_non_blocking(wrapper);
	}
}

#ifdef STARPU_RECURSIVE_TASKS
static void empty_function(STARPU_ATTRIBUTE_UNUSED void * buffers[], STARPU_ATTRIBUTE_UNUSED void * cl_arg)
{
}

static struct starpu_codelet control_cl =
{ // this function is only used before mpi_ssend, so we do not need any cuda or opencl function : the send will be done by the cpu
        .cpu_funcs = {empty_function},
        .nbuffers = STARPU_VARIABLE_NBUFFERS
};
#endif

/* The data must be released by calling starpu_data_release later on */
int _starpu_data_acquire_on_node_cb_sequential_consistency_sync_jobids(starpu_data_handle_t handle, int node,
							  enum starpu_data_access_mode mode,
							  void (*callback_soon)(void *arg, double delay),
							  void (*callback_acquired)(void *arg, int *node, enum starpu_data_access_mode mode),
							  void (*callback)(void *arg),
							  void *arg,
							  int sequential_consistency, int quick,
							  long *pre_sync_jobid, long *post_sync_jobid, int prio, int need_to_be_unpart_or_part)
{
#ifndef STARPU_RECURSIVE_TASKS
	(void)need_to_be_unpart_or_part;
#endif
	STARPU_ASSERT(handle);
	STARPU_ASSERT_MSG(handle->nchildren == 0, "Acquiring a partitioned data (%p) is not possible", handle);
	_STARPU_LOG_IN();

	/* Check that previous tasks have set a value if needed */

#ifdef STARPU_RECURSIVE_TASKS
	if (!need_to_be_unpart_or_part)
		// only when we do not need to part or unpart -> if we need, maybe data is not initialized, but it is normal
#endif
		_starpu_data_check_initialized(handle, mode);

	struct user_interaction_wrapper *wrapper;
	_STARPU_MALLOC(wrapper, sizeof(struct user_interaction_wrapper));

	_starpu_data_acquire_wrapper_init(wrapper, handle, node, mode);
	wrapper->async = 1;

	wrapper->callback_acquired = callback_acquired;
	wrapper->callback = callback;
	wrapper->callback_arg = arg;
	wrapper->pre_sync_task = NULL;
	wrapper->post_sync_task = NULL;
	wrapper->prio = prio;

#ifdef STARPU_RECURSIVE_TASKS
	_STARPU_RECURSIVE_TASKS_DEBUG( "ACQUIRING HANDLE %p in mode %d\n", handle, mode);
	assert(mode != 0);
#endif

	STARPU_PTHREAD_MUTEX_LOCK(&handle->sequential_consistency_mutex);
	int handle_sequential_consistency = handle->sequential_consistency;
	if (handle_sequential_consistency && sequential_consistency)
	{
		struct starpu_task *new_task = NULL;
		struct _starpu_job *pre_sync_job, *post_sync_job;
#ifdef STARPU_RECURSIVE_TASKS
		int submit_pre_sync = 1; //need_to_be_unpart_or_part;
#else
		int submit_pre_sync = 0;
#endif
		wrapper->pre_sync_task = starpu_task_create();
#ifdef STARPU_RECURSIVE_TASKS_VERBOSE
		asprintf(&wrapper->pre_sync_task->name, "_starpu_data_acquire_cb_pre(%p)", handle);
#else
		wrapper->pre_sync_task->name = "_starpu_data_acquire_cb_pre";
#endif
		wrapper->pre_sync_task->detach = 1;
		wrapper->pre_sync_task->callback_func = starpu_data_acquire_cb_pre_sync_callback;
		wrapper->pre_sync_task->callback_arg = wrapper;
		wrapper->pre_sync_task->soon_callback_func = callback_soon;
		wrapper->pre_sync_task->soon_callback_arg = arg;
		wrapper->pre_sync_task->type = STARPU_TASK_TYPE_DATA_ACQUIRE;
		wrapper->pre_sync_task->priority = prio;
		pre_sync_job = _starpu_get_job_associated_to_task(wrapper->pre_sync_task);
		if (pre_sync_jobid)
			*pre_sync_jobid = pre_sync_job->job_id;

		wrapper->post_sync_task = starpu_task_create();
#ifdef STARPU_RECURSIVE_TASKS_VERBOSE
		asprintf(&wrapper->post_sync_task->name, "_starpu_data_acquire_cb_release(%p)", handle);
#else
		wrapper->post_sync_task->name = "_starpu_data_acquire_cb_release";
#endif
		wrapper->post_sync_task->detach = 1;
		wrapper->post_sync_task->type = STARPU_TASK_TYPE_DATA_ACQUIRE;
		wrapper->post_sync_task->priority = prio;
		post_sync_job = _starpu_get_job_associated_to_task(wrapper->post_sync_task);
		if (post_sync_jobid)
			*post_sync_jobid = post_sync_job->job_id;

		if (quick)
			pre_sync_job->quick_next = post_sync_job;

#ifdef STARPU_RECURSIVE_TASKS
		struct starpu_task *control_task = NULL, *control_task_end = NULL;
		if (need_to_be_unpart_or_part)
		{
			submit_pre_sync = 1;
			control_task = starpu_task_create();
                        control_task->name = "control_sync_jobids";
                        control_task->cl = &control_cl;
                        control_task->handles[0] = handle;
                        control_task->modes[0] = mode;
			control_task->nbuffers = 1;
                        starpu_task_declare_deps(wrapper->pre_sync_task, 1, control_task);
			control_task_end = starpu_task_create();
			control_task_end ->name = "control_sync_jobids_end";
			struct _starpu_job *j_end = _starpu_get_job_associated_to_task (control_task_end);
			j_end->recursive.need_part_unpart = 0;
			control_task_end->handles[0] = handle;
			control_task_end->modes[0] = mode;
			control_task_end->nbuffers = 1;
			starpu_task_declare_deps(control_task_end, 1, wrapper->post_sync_task);
			STARPU_PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);
			starpu_task_submit(control_task);
			STARPU_PTHREAD_MUTEX_LOCK(&handle->sequential_consistency_mutex);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);
		STARPU_PTHREAD_MUTEX_LOCK(handle->partition_mutex);
		STARPU_PTHREAD_MUTEX_LOCK(&handle->sequential_consistency_mutex);
		/* recursive_task unpartitioning */
/*		if (handle->ctrl_unpartition_children)
		{
			_STARPU_DEBUG("EEEEEEEEEEEEEEEEEEEEEEEEEE\nacquire on unpartition handle %p\nEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE\n", handle);
			// plug the post_sync tasks before the control task
			starpu_task_declare_deps(handle->ctrl_unpartition_children, 1, wrapper->post_sync_task);
			_STARPU_DEBUG("[%p] Set dependencies rel(%p) -> ctrl(%p)\n", handle, wrapper->post_sync_task, handle->ctrl_unpartition_children);
			handle->ctrl_unpartition_children = NULL;
		}*/
		if (handle->ctrl_unpartition)
		{
			starpu_task_declare_deps(wrapper->pre_sync_task, 1, handle->ctrl_unpartition);
			_STARPU_RECURSIVE_TASKS_DEBUG("Set dependencies rel(%p) -> ctrl(%p)\n", handle->ctrl_unpartition, wrapper->pre_sync_task);
//			submit_pre_sync = 1;
		}

		/* recursive_task partitioning */
		if (handle->last_partition)
		{
			_STARPU_RECURSIVE_TASKS_DEBUG("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\nacquire on partition handle %p\nCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n", handle);
			/* plug the pre_sync task after the partition needing it */
			starpu_task_declare_deps(wrapper->pre_sync_task, 1, handle->last_partition);
			_STARPU_RECURSIVE_TASKS_DEBUG("[%p] Set dependencies %s(%p) -> acq(%p)\n", handle, starpu_task_get_name(handle->last_partition), handle->last_partition, wrapper->pre_sync_task);
			starpu_task_declare_deps(wrapper->post_sync_task, 1, wrapper->pre_sync_task);
			_STARPU_RECURSIVE_TASKS_DEBUG("[%p] Set dependencies acq(%p) -> rel(%p)\n", handle, wrapper->pre_sync_task, wrapper->post_sync_task);
			submit_pre_sync = 1;
			/* handle->last_partition = NULL; */
			/* Est-il possible que des acquires issus d'une autre tâche que last_partition arrive ici ? */
		}
		else
#endif
		{
			_STARPU_RECURSIVE_TASKS_DEBUG("JJJJJJJJJJJJJJJJJJJJJJJJJJJ\ndetecting implicit data deps normally for handle %p\nJJJJJJJJJJJJJJJJJJJJJJJJJJJJ\n", handle);
			new_task = _starpu_detect_implicit_data_deps_with_handle(wrapper->pre_sync_task, &submit_pre_sync, wrapper->post_sync_task, &_starpu_get_job_associated_to_task(wrapper->post_sync_task)->implicit_dep_slot, handle, mode, sequential_consistency);
		}

		STARPU_PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);
#ifdef STARPU_RECURSIVE_TASKS
		STARPU_PTHREAD_MUTEX_UNLOCK(handle->partition_mutex);
#endif


		if (STARPU_UNLIKELY(new_task))
		{
			int ret = _starpu_task_submit_internally(new_task);
			STARPU_ASSERT(!ret);
		}

		if (submit_pre_sync)
		{
			int ret = _starpu_task_submit_internally(wrapper->pre_sync_task);
			STARPU_ASSERT(!ret);
		}
		else
		{
			wrapper->pre_sync_task->detach = 0;
			starpu_task_destroy(wrapper->pre_sync_task);
			starpu_data_acquire_cb_pre_sync_callback(wrapper);
		}
#ifdef STARPU_RECURSIVE_TASKS
		if (control_task_end)
		{
			starpu_task_submit(control_task_end);
		}
#endif
	}
	else
	{
		if (pre_sync_jobid)
			*pre_sync_jobid = -1;
		if (post_sync_jobid)
			*post_sync_jobid = -1;
		STARPU_PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);

		starpu_data_acquire_cb_pre_sync_callback(wrapper);
	}

	_STARPU_LOG_OUT();
	return 0;
}

int starpu_data_acquire_on_node_cb_sequential_consistency_sync_jobids(starpu_data_handle_t handle, int node,
							  enum starpu_data_access_mode mode,
							  void (*callback_soon)(void *arg, double delay),
							  void (*callback_acquired)(void *arg, int *node, enum starpu_data_access_mode mode),
							  void (*callback)(void *arg),
							  void *arg,
							  int sequential_consistency, int quick,
							  long *pre_sync_jobid, long *post_sync_jobid, int prio)
{
	return 	_starpu_data_acquire_on_node_cb_sequential_consistency_sync_jobids(handle, node,
										   mode,
										   callback_soon,
										   callback_acquired,
										   callback,
										   arg,
										   sequential_consistency, quick,
										   pre_sync_jobid, post_sync_jobid, prio, 1);
}

int starpu_data_acquire_on_node_cb_sequential_consistency_quick(starpu_data_handle_t handle, int node,
								enum starpu_data_access_mode mode, void (*callback)(void *), void *arg,
								int sequential_consistency, int quick)
{
	return starpu_data_acquire_on_node_cb_sequential_consistency_sync_jobids(handle, node, mode, NULL, NULL, callback, arg, sequential_consistency, quick, NULL, NULL, STARPU_DEFAULT_PRIO);
}

int starpu_data_acquire_on_node_cb_sequential_consistency(starpu_data_handle_t handle, int node,
							  enum starpu_data_access_mode mode, void (*callback)(void *), void *arg,
							  int sequential_consistency)
{
	return starpu_data_acquire_on_node_cb_sequential_consistency_quick(handle, node, mode, callback, arg, sequential_consistency, 0);
}

int starpu_data_acquire_on_node_cb(starpu_data_handle_t handle, int node,
				   enum starpu_data_access_mode mode, void (*callback)(void *), void *arg)
{
	return starpu_data_acquire_on_node_cb_sequential_consistency(handle, node, mode, callback, arg, 1);
}

int starpu_data_acquire_cb(starpu_data_handle_t handle,
			   enum starpu_data_access_mode mode, void (*callback)(void *), void *arg)
{
	int home_node = handle->home_node;
	if (home_node < 0)
		home_node = STARPU_MAIN_RAM;
	return starpu_data_acquire_on_node_cb(handle, home_node, mode, callback, arg);
}

int starpu_data_acquire_cb_sequential_consistency(starpu_data_handle_t handle,
						  enum starpu_data_access_mode mode, void (*callback)(void *), void *arg, int sequential_consistency)
{
	int home_node = handle->home_node;
	if (home_node < 0)
		home_node = STARPU_MAIN_RAM;
	return starpu_data_acquire_on_node_cb_sequential_consistency(handle, home_node, mode, callback, arg, sequential_consistency);
}


int starpu_data_get_on_node_cb_prio(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg, enum starpu_is_prefetch prefetch, int prio)
{
	if (node == STARPU_ACQUIRE_NO_NODE)
		/* Still no transfer to do.  */
		return 0;
	STARPU_ASSERT_MSG(node != STARPU_ACQUIRE_NO_NODE_LOCK_ALL, "starpu_data_get_on_node_cb_prio does not support STARPU_ACQUIRE_NO_NODE_LOCK_ALL");

#ifndef STARPU_NO_ASSERT
	_starpu_spin_lock(&handle->header_lock);
	STARPU_ASSERT_MSG(((mode & STARPU_W) == STARPU_W && handle->current_mode == STARPU_W) ||
	                  (mode == STARPU_R && handle->current_mode == STARPU_R), "The mode given to starpu_data_get_on_node_cb_prio must be coherent with the mode of the previous acquisition");
	_starpu_spin_unlock(&handle->header_lock);
#endif

	_starpu_fetch_data_on_node(handle, node, &handle->per_node[node], mode, 0, NULL, prefetch, 1, callback, arg, prio, "starpu_data_get_on_node_cb_prio");

	/* _starpu_fetch_data_on_node above takes a busy reference in addition to the one we had taken in the previous data_acquire */
	_starpu_spin_lock(&handle->header_lock);
	handle->busy_count--;
	if (!_starpu_notify_data_dependencies(handle, mode))
		_starpu_spin_unlock(&handle->header_lock);

	return 0;
}

/*
 *	Blocking data request from application
 */



static inline void _starpu_data_acquire_continuation(void *arg)
{
	struct user_interaction_wrapper *wrapper = (struct user_interaction_wrapper *) arg;

	starpu_data_handle_t handle = wrapper->handle;
	STARPU_ASSERT(handle);

	_starpu_data_acquire_launch_fetch(wrapper, 0, NULL, NULL);
	_starpu_data_acquire_wrapper_finished(wrapper);
}

int _starpu_recursive_tasks_disable_sequential_consistency = 0;

/* The data must be released by calling starpu_data_release later on */
int starpu_data_acquire_on_node(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode)
{
	STARPU_ASSERT(handle);
	STARPU_ASSERT_MSG(handle->nchildren == 0, "Acquiring a partitioned data is not possible");
	_STARPU_LOG_IN();

	/* unless asynchronous, it is forbidden to call this function from a callback or a codelet */
	STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "Acquiring a data synchronously is not possible from a codelet or from a task callback, use starpu_data_acquire_cb instead.");

	/* Check that previous tasks have set a value if needed */
	_starpu_data_check_initialized(handle, mode);

	if (node >= 0 && _starpu_data_is_multiformat_handle(handle) &&
	    _starpu_handle_needs_conversion_task(handle, node))
	{
		struct starpu_task *task = _starpu_create_conversion_task(handle, node);
		int ret;
		_starpu_spin_lock(&handle->header_lock);
		handle->refcnt--;
		_STARPU_RECURSIVE_TASKS_DEBUG("Release refcnt on data %p\n", handle);
		handle->busy_count--;
		handle->mf_node = node;
		_starpu_spin_unlock(&handle->header_lock);
		task->synchronous = 1;
		ret = _starpu_task_submit_internally(task);
		STARPU_ASSERT(!ret);
	}

	struct user_interaction_wrapper wrapper;
	_starpu_data_acquire_wrapper_init(&wrapper, handle, node, mode);

//	_STARPU_DEBUG("TAKE sequential_consistency_mutex starpu_data_acquire\n");
	STARPU_PTHREAD_MUTEX_LOCK(&handle->sequential_consistency_mutex);
	int sequential_consistency = handle->sequential_consistency;
	if (sequential_consistency)
	{
		struct starpu_task *new_task;
		int submit_pre_sync = 0;
		wrapper.pre_sync_task = starpu_task_create();
		wrapper.pre_sync_task->name = "_starpu_data_acquire_pre";
		wrapper.pre_sync_task->detach = 0;
		wrapper.pre_sync_task->type = STARPU_TASK_TYPE_DATA_ACQUIRE;

		wrapper.post_sync_task = starpu_task_create();
		wrapper.post_sync_task->name = "_starpu_data_acquire_post";
		wrapper.post_sync_task->detach = 1;
		wrapper.post_sync_task->type = STARPU_TASK_TYPE_DATA_ACQUIRE;

		/**
		   The insertion of the first recursive task is going
		   to disable sequential_consistency (even though it
		   might be a bit cheeky, it's needed in the case of
		   recursive task partitioning
		*/
		new_task = _starpu_detect_implicit_data_deps_with_handle(wrapper.pre_sync_task, &submit_pre_sync, wrapper.post_sync_task, &_starpu_get_job_associated_to_task(wrapper.post_sync_task)->implicit_dep_slot, handle, mode,
									 _starpu_recursive_tasks_disable_sequential_consistency ? 0 : sequential_consistency);
		STARPU_PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);

		if (STARPU_UNLIKELY(new_task))
		{
			int ret = _starpu_task_submit_internally(new_task);
			STARPU_ASSERT(!ret);
		}

		if (submit_pre_sync)
		{
			wrapper.pre_sync_task->synchronous = 1;
			int ret = _starpu_task_submit_internally(wrapper.pre_sync_task);
			STARPU_ASSERT(!ret);
		}
		else
		{
			wrapper.pre_sync_task->detach = 0;
			starpu_task_destroy(wrapper.pre_sync_task);
		}
	}
	else
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);
	}

	/*
	 * we try to get the data, if we do not succeed immediately,
	 * we set a callback function that will be executed
	 * automatically when the data is available again, otherwise we
	 * fetch the data directly
	 */
	if (!_starpu_attempt_to_submit_data_request_from_apps(handle, mode, _starpu_data_acquire_continuation, &wrapper))
	{
		/* no one has locked this data yet, so we proceed immediately */
		_starpu_data_acquire_launch_fetch(&wrapper, 0, NULL, NULL);
	}
	else
	{
		_starpu_data_acquire_wrapper_wait(&wrapper);
	}
	_starpu_data_acquire_wrapper_fini(&wrapper);

	/* At that moment, the caller holds a reference to the piece of data.
	 * We enqueue the "post" sync task in the list associated to the handle
	 * so that it is submitted by the starpu_data_release
	 * function. */
	if (sequential_consistency)
		_starpu_add_post_sync_tasks(wrapper.post_sync_task, handle);

	_STARPU_LOG_OUT();
	return 0;
}

int starpu_data_acquire(starpu_data_handle_t handle, enum starpu_data_access_mode mode)
{
	int home_node = handle->home_node;
	if (home_node < 0)
		home_node = STARPU_MAIN_RAM;
	return starpu_data_acquire_on_node(handle, home_node, mode);
}

int starpu_data_acquire_on_node_try(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode)
{
	STARPU_ASSERT(handle);
	STARPU_ASSERT_MSG(handle->nchildren == 0, "Acquiring a partitioned data is not possible");
	/* it is forbidden to call this function from a callback or a codelet */
	STARPU_ASSERT_MSG(_starpu_worker_may_perform_blocking_calls(), "Acquiring a data synchronously is not possible from a codelet or from a task callback, use starpu_data_acquire_cb instead.");

	/* Check that previous tasks have set a value if needed */
	_starpu_data_check_initialized(handle, mode);

	int ret;
	STARPU_ASSERT_MSG(!_starpu_data_is_multiformat_handle(handle), "not supported yet");
	STARPU_PTHREAD_MUTEX_LOCK(&handle->sequential_consistency_mutex);
	ret = _starpu_test_implicit_data_deps_with_handle(handle, mode);
	STARPU_PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);
	if (ret)
		return ret;

	struct user_interaction_wrapper wrapper;
	_starpu_data_acquire_wrapper_init(&wrapper, handle, node, mode);

	/*
	 * we try to get the data, if we do not succeed immediately,
	 * we set a callback function that will be executed
	 * automatically when the data is available again, otherwise we
	 * fetch the data directly
	 */
	if (!_starpu_attempt_to_submit_data_request_from_apps(handle, mode, _starpu_data_acquire_continuation, &wrapper))
	{
		/* no one has locked this data yet, so we proceed immediately */
		_starpu_data_acquire_launch_fetch(&wrapper, 0, NULL, NULL);
	}
	else
	{
		_starpu_data_acquire_wrapper_wait(&wrapper);
	}
	_starpu_data_acquire_wrapper_fini(&wrapper);

	return 0;
}

int starpu_data_acquire_try(starpu_data_handle_t handle, enum starpu_data_access_mode mode)
{
	return starpu_data_acquire_on_node_try(handle, STARPU_MAIN_RAM, mode);
}

/* This function must be called after starpu_data_acquire so that the
 * application release the data */
void starpu_data_release_to_on_node(starpu_data_handle_t handle, enum starpu_data_access_mode mode, int node)
{
	STARPU_ASSERT(handle);

	if (mode == STARPU_RW)
		/* They are equivalent here, and current_mode is never STARPU_RW */
		mode = STARPU_W;

	STARPU_ASSERT_MSG(mode == STARPU_NONE ||
			  mode == handle->current_mode ||
			  (mode == STARPU_R &&
			     handle->current_mode == STARPU_W),
		"We only support releasing from W to R");

	/* In case there are some implicit dependencies, unlock the "post sync" tasks */
	_STARPU_RECURSIVE_TASKS_DEBUG("Unlocking post_sync task for handle %p\n", handle);
	_starpu_unlock_post_sync_tasks(handle, mode);

	/* The application can now release the rw-lock */
	if (node >= 0)
		_starpu_release_data_on_node(handle, 0, mode, &handle->per_node[node]);
	else
	{
		_starpu_spin_lock(&handle->header_lock);
		if (node == STARPU_ACQUIRE_NO_NODE_LOCK_ALL)
		{
			int i;
			for (i = 0; i < STARPU_MAXNODES; i++)
				handle->per_node[i].refcnt--;
		}
		handle->busy_count--;
		if (!_starpu_notify_data_dependencies(handle, mode))
			_starpu_spin_unlock(&handle->header_lock);
	}
}

void starpu_data_release_on_node(starpu_data_handle_t handle, int node)
{
	starpu_data_release_to_on_node(handle, STARPU_NONE, node);
}

void starpu_data_release_to(starpu_data_handle_t handle, enum starpu_data_access_mode mode)
{
	int home_node = handle->home_node;
	if (home_node < 0)
		home_node = STARPU_MAIN_RAM;
	starpu_data_release_to_on_node(handle, mode, home_node);
}

void starpu_data_release(starpu_data_handle_t handle)
{
	starpu_data_release_to(handle, STARPU_NONE);
}

static void _prefetch_data_on_node(void *arg)
{
	struct user_interaction_wrapper *wrapper = (struct user_interaction_wrapper *) arg;
	starpu_data_handle_t handle = wrapper->handle;

	_starpu_data_acquire_launch_fetch(wrapper, wrapper->async, NULL, NULL);

	if (wrapper->async)
		free(wrapper);
	else
		_starpu_data_acquire_wrapper_finished(wrapper);

	_starpu_spin_lock(&handle->header_lock);
	if (!_starpu_notify_data_dependencies(handle, STARPU_NONE))
		_starpu_spin_unlock(&handle->header_lock);
}

/* Prefetch data. This is the execution-time part */
static
int __starpu_prefetch_data_on_node_with_mode(starpu_data_handle_t handle, unsigned node, unsigned async, enum starpu_data_access_mode mode, enum starpu_is_prefetch prefetch, int prio)
{
	STARPU_ASSERT(handle);

	/* it is forbidden to call this function from a callback or a codelet */
	STARPU_ASSERT_MSG(async || _starpu_worker_may_perform_blocking_calls(), "Synchronous prefetch is not possible from a task or a callback");

#ifdef STARPU_RECURSIVE_TASKS
	/* Check that previous tasks have set a value if needed */
	_starpu_data_check_initialized(handle, mode);
#endif

	struct user_interaction_wrapper *wrapper;
	_STARPU_MALLOC(wrapper, sizeof(*wrapper));

	_starpu_data_acquire_wrapper_init(wrapper, handle, node, STARPU_R);

	wrapper->detached = async;
	wrapper->prefetch = prefetch;
	wrapper->async = async;
	wrapper->prio = prio;

	if (!_starpu_attempt_to_submit_data_request_from_apps(handle, mode, _prefetch_data_on_node, wrapper))
	{
		/* we can immediately proceed */
		struct _starpu_data_replicate *replicate = &handle->per_node[node];
		_starpu_data_acquire_launch_fetch(wrapper, async, NULL, NULL);

		_starpu_data_acquire_wrapper_fini(wrapper);
		free(wrapper);

		/* remove the "lock"/reference */

		_starpu_spin_lock(&handle->header_lock);

		if (!async)
		{
			/* Release our refcnt, like _starpu_release_data_on_node would do */
			replicate->refcnt--;
			STARPU_ASSERT(replicate->refcnt >= 0);
			STARPU_ASSERT(handle->busy_count > 0);
			handle->busy_count--;
		}

		/* In case there was a temporary handle (eg. used for reduction), this
		 * handle may have requested to be destroyed when the data is released
		 * */
		if (!_starpu_notify_data_dependencies(handle, STARPU_NONE))
			_starpu_spin_unlock(&handle->header_lock);
	}
	else if (!async)
	{
		_starpu_data_acquire_wrapper_wait(wrapper);
		_starpu_data_acquire_wrapper_fini(wrapper);
		free(wrapper);
	}

	return 0;
}

/* Prefetch data. This is the submission-time part */
static
int _starpu_prefetch_data_on_node_with_mode(starpu_data_handle_t handle, unsigned node, unsigned async, enum starpu_data_access_mode mode, enum starpu_is_prefetch prefetch, int prio)
{
	/* Check that previous tasks have set a value if needed */
	/* Only valid at submission time, not execution time */
	_starpu_data_check_initialized(handle, mode);

	return __starpu_prefetch_data_on_node_with_mode(handle, node, async, mode, prefetch, prio);
}

int starpu_data_fetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async)
{
	return _starpu_prefetch_data_on_node_with_mode(handle, node, async, STARPU_R, STARPU_FETCH, STARPU_DEFAULT_PRIO);
}

int starpu_data_prefetch_on_node_prio(starpu_data_handle_t handle, unsigned node, unsigned async, int prio)
{
	return _starpu_prefetch_data_on_node_with_mode(handle, node, async, STARPU_R, STARPU_PREFETCH, prio);
}

int starpu_data_prefetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async)
{
	return starpu_data_prefetch_on_node_prio(handle, node, async, STARPU_DEFAULT_PRIO);
}

int starpu_data_idle_prefetch_on_node_prio(starpu_data_handle_t handle, unsigned node, unsigned async, int prio)
{
	return _starpu_prefetch_data_on_node_with_mode(handle, node, async, STARPU_R, STARPU_IDLEFETCH, prio);
}

int starpu_data_idle_prefetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async)
{
	return starpu_data_idle_prefetch_on_node_prio(handle, node, async, STARPU_DEFAULT_PRIO);
}

/* Execution-time part */
static void _starpu_data_wont_use(void *data)
{
	unsigned node;
	starpu_data_handle_t handle = data;

	_starpu_trace_data_doing_wont_use(&handle);

	_starpu_spin_lock(&handle->header_lock);
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct _starpu_data_replicate *local = &handle->per_node[node];
		if (local->allocated && local->automatically_allocated)
			_starpu_memchunk_wont_use(local->mc, node);
	}
	if (handle->per_worker)
	{
		unsigned nworkers = starpu_worker_get_count();
		unsigned worker;
		for (worker = 0; worker < nworkers; worker++)
		{
			struct _starpu_data_replicate *local = &handle->per_worker[worker];
			if (local->allocated && local->automatically_allocated)
				_starpu_memchunk_wont_use(local->mc, starpu_worker_get_memory_node(worker));
		}
	}
	_starpu_spin_unlock(&handle->header_lock);
	starpu_data_release_on_node(handle, STARPU_ACQUIRE_NO_NODE_LOCK_ALL);
	if (handle->home_node != -1)
		__starpu_prefetch_data_on_node_with_mode(handle, handle->home_node, 1, STARPU_R, STARPU_IDLEFETCH, STARPU_DEFAULT_PRIO);
	else
	{
		if (handle->ooc)
		{
			/* Try to push it to some disk */
			unsigned i;
			unsigned nnodes = starpu_memory_nodes_get_count();
			for (i = 0; i < nnodes; i++)
			{
				if (starpu_node_get_kind(i) == STARPU_DISK_RAM)
					__starpu_prefetch_data_on_node_with_mode(handle, i, 1, STARPU_R, STARPU_IDLEFETCH, STARPU_DEFAULT_PRIO);
			}
		}
	}
}

#ifdef STARPU_RECURSIVE_TASKS
#if 0
static void flush_func(void *buffers[], void *arg)
{
	(void) buffers;
	(void) arg;
}

static struct starpu_codelet flush_codelet =
{
	.cpu_funcs = {flush_func},
	.nbuffers = 1
};
#endif
#endif

void starpu_data_wont_use(starpu_data_handle_t handle)
{
#ifndef STARPU_RECURSIVE_TASKS
	if (!handle->initialized)
		/* No value atm actually */
		return;
#endif

#ifdef STARPU_RECURSIVE_TASKS
//	char *fname;
//	asprintf(&fname, "Flush(%p)", handle);
//	starpu_task_insert(&flush_codelet,
//			   STARPU_RW, handle,
//			   STARPU_NAME, fname, 0);
#else
	if (starpu_data_get_nb_children(handle) != 0)
	{
		int i;
		for(i=0 ; i<starpu_data_get_nb_children(handle) ; i++)
			starpu_data_wont_use(starpu_data_get_child(handle, i));
		return;
	}

	if (handle->nactive_readonly_children != 0)
	{
		unsigned i;
		for(i=0 ; i<handle->nactive_readonly_children; i++)
		{
			unsigned j;
			for(j=0 ; j<handle->active_readonly_nchildren[i] ; j++)
				starpu_data_wont_use(handle->active_readonly_children[i][j]);
		}
	}

	if (handle->active_nchildren != 0)
	{
		unsigned j;
		for(j=0 ; j<handle->active_nchildren ; j++)
			starpu_data_wont_use(handle->active_children[j]);
		return;
	}
#endif

	_starpu_trace_data_wont_use(&handle);
	starpu_data_acquire_on_node_cb_sequential_consistency_quick(handle, STARPU_ACQUIRE_NO_NODE_LOCK_ALL, STARPU_R, _starpu_data_wont_use, handle, 1, 1);
}

/*
 *	It is possible to specify that a piece of data can be discarded without
 *	impacting the application.
 */
int _starpu_has_not_important_data;
void starpu_data_advise_as_important(starpu_data_handle_t handle, unsigned is_important)
{
	if (!is_important)
		_starpu_has_not_important_data = 1;

	_starpu_spin_lock(&handle->header_lock);

	/* first take all the children lock (in order !) */
	unsigned child;
	for (child = 0; child < handle->nchildren; child++)
	{
		/* make sure the intermediate children is advised as well */
		starpu_data_handle_t child_handle = starpu_data_get_child(handle, child);
		if (child_handle->nchildren > 0)
			starpu_data_advise_as_important(child_handle, is_important);
	}

	handle->is_not_important = !is_important;

	/* now the parent may be used again so we release the lock */
	_starpu_spin_unlock(&handle->header_lock);

}

void starpu_data_set_sequential_consistency_flag(starpu_data_handle_t handle, unsigned flag)
{
	_starpu_spin_lock(&handle->header_lock);

	unsigned child;
	for (child = 0; child < handle->nchildren; child++)
	{
		/* make sure that the flags are applied to the children as well */
		starpu_data_handle_t child_handle = starpu_data_get_child(handle, child);
		if (child_handle->nchildren > 0)
			starpu_data_set_sequential_consistency_flag(child_handle, flag);
	}

	STARPU_PTHREAD_MUTEX_LOCK(&handle->sequential_consistency_mutex);
	handle->sequential_consistency = flag;
	STARPU_PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);

	_starpu_spin_unlock(&handle->header_lock);
}

unsigned starpu_data_get_sequential_consistency_flag(starpu_data_handle_t handle)
{
	return handle->sequential_consistency;
}

void starpu_data_set_ooc_flag(starpu_data_handle_t handle, unsigned flag)
{
	handle->ooc = flag;
}

unsigned starpu_data_get_ooc_flag(starpu_data_handle_t handle)
{
	return handle->ooc;
}

/* By default, sequential consistency is enabled */
static unsigned default_sequential_consistency_flag = 1;

unsigned starpu_data_get_default_sequential_consistency_flag(void)
{
	return default_sequential_consistency_flag;
}

void starpu_data_set_default_sequential_consistency_flag(unsigned flag)
{
	default_sequential_consistency_flag = flag;
}

/* Query the status of the handle on the specified memory node. */
void starpu_data_query_status2(starpu_data_handle_t handle, int memory_node, int *is_allocated, int *is_valid, int *is_loading, int *is_requested)
{
// XXX : this is just a hint, so we don't take the lock ...
//	_starpu_spin_lock(&handle->header_lock);

	if (is_allocated)
		*is_allocated = handle->per_node[memory_node].allocated || handle->per_node[memory_node].mapped != STARPU_UNMAPPED;

	if (is_valid)
		*is_valid = (handle->per_node[memory_node].state != STARPU_INVALID);

	if (is_loading)
		*is_loading = handle->per_node[memory_node].load_request != NULL;

	if (is_requested)
	{
		int requested = 0;

		unsigned node;
		for (node = 0; node < STARPU_MAXNODES; node++)
		{
			if (handle->per_node[memory_node].request[node])
			{
				requested = 1;
				break;
			}
		}

		*is_requested = requested;
	}

//	_starpu_spin_unlock(&handle->header_lock);
}

void starpu_data_query_status(starpu_data_handle_t handle, int memory_node, int *is_allocated, int *is_valid, int *is_requested)
{
	return starpu_data_query_status2(handle, memory_node, is_allocated, is_valid, NULL, is_requested);
}
