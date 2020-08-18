/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static void _starpu_data_check_initialized(starpu_data_handle_t handle, enum starpu_data_access_mode mode)
{
	if (!(mode & STARPU_R))
		return;

	if (!handle->initialized && handle->init_cl)
	{
		int ret = starpu_task_insert(handle->init_cl, STARPU_W, handle, 0);
		STARPU_ASSERT(ret == 0);
	}
	STARPU_ASSERT_MSG(handle->initialized, "handle %p is not initialized while trying to read it\n", handle);
}

/* Explicitly ask StarPU to allocate room for a piece of data on the specified
 * memory node. */
int starpu_data_request_allocation(starpu_data_handle_t handle, unsigned node)
{
	struct _starpu_data_request *r;

	STARPU_ASSERT(handle);

	_starpu_spin_lock(&handle->header_lock);

	r = _starpu_create_data_request(handle, NULL, &handle->per_node[node], node, STARPU_NONE, 0, STARPU_PREFETCH, 0, 0, "starpu_data_request_allocation");

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
	enum _starpu_is_prefetch prefetch;
	unsigned async;
	int prio;
	void (*callback)(void *);
	void (*callback_fetch_data)(void *); // called after fetch_data
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
	wrapper->finished = 0;
	STARPU_PTHREAD_COND_INIT(&wrapper->cond, NULL);
	STARPU_PTHREAD_MUTEX_INIT(&wrapper->lock, NULL);
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

/* Called when the fetch into target memory is done, we're done! */
static inline void _starpu_data_acquire_fetch_done(struct user_interaction_wrapper *wrapper)
{
	if (wrapper->node >= 0)
	{
		struct _starpu_data_replicate *replicate = &wrapper->handle->per_node[wrapper->node];
		if (replicate->mc)
			replicate->mc->diduse = 1;
	}
}

/* Called when the data acquisition is done, to launch the fetch into target memory */
static inline void _starpu_data_acquire_launch_fetch(struct user_interaction_wrapper *wrapper, int async, void (*callback)(void *), void *callback_arg)
{
	int node = wrapper->node;
	starpu_data_handle_t handle = wrapper->handle;
	struct _starpu_data_replicate *replicate = node >= 0 ? &handle->per_node[node] : NULL;

	int ret = _starpu_fetch_data_on_node(handle, node, replicate, wrapper->mode, wrapper->detached, wrapper->prefetch, async, callback, callback_arg, wrapper->prio, "_starpu_data_acquire_launch_fetch");
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

	/* At that moment, the caller holds a reference to the piece of data.
	 * We enqueue the "post" sync task in the list associated to the handle
	 * so that it is submitted by the starpu_data_release
	 * function. */
	if (wrapper->post_sync_task)
		_starpu_add_post_sync_tasks(wrapper->post_sync_task, handle);

	_starpu_data_acquire_fetch_done(wrapper);

	wrapper->callback(wrapper->callback_arg);

	_starpu_data_acquire_wrapper_fini(wrapper);
	free(wrapper);
}

/* Called when the data acquisition is done, launch the fetch into target memory */
static void _starpu_data_acquire_continuation_non_blocking(void *arg)
{
	_starpu_data_acquire_launch_fetch(arg, 1, _starpu_data_acquire_fetch_data_callback, arg);
}

/* Called when the implicit data dependencies are done, launch the data acquisition */
static void starpu_data_acquire_cb_pre_sync_callback(void *arg)
{
	struct user_interaction_wrapper *wrapper = (struct user_interaction_wrapper *) arg;

	/* we try to get the data, if we do not succeed immediately, we set a
 	* callback function that will be executed automatically when the data is
 	* available again, otherwise we fetch the data directly */
	if (!_starpu_attempt_to_submit_data_request_from_apps(wrapper->handle, wrapper->mode,
			_starpu_data_acquire_continuation_non_blocking, wrapper))
	{
		/* no one has locked this data yet, so we proceed immediately */
		_starpu_data_acquire_continuation_non_blocking(wrapper);
	}
}

/* The data must be released by calling starpu_data_release later on */
int starpu_data_acquire_on_node_cb_sequential_consistency_sync_jobids(starpu_data_handle_t handle, int node,
							  enum starpu_data_access_mode mode, void (*callback)(void *), void *arg,
							  int sequential_consistency, int quick,
							  long *pre_sync_jobid, long *post_sync_jobid)
{
	STARPU_ASSERT(handle);
	STARPU_ASSERT_MSG(handle->nchildren == 0, "Acquiring a partitioned data (%p) is not possible", handle);
        _STARPU_LOG_IN();

	/* Check that previous tasks have set a value if needed */
	_starpu_data_check_initialized(handle, mode);

	struct user_interaction_wrapper *wrapper;
	_STARPU_MALLOC(wrapper, sizeof(struct user_interaction_wrapper));

	_starpu_data_acquire_wrapper_init(wrapper, handle, node, mode);
	wrapper->async = 1;

	wrapper->callback = callback;
	wrapper->callback_arg = arg;
	wrapper->pre_sync_task = NULL;
	wrapper->post_sync_task = NULL;

	STARPU_PTHREAD_MUTEX_LOCK(&handle->sequential_consistency_mutex);
	int handle_sequential_consistency = handle->sequential_consistency;
	if (handle_sequential_consistency && sequential_consistency)
	{
		struct starpu_task *new_task;
		struct _starpu_job *pre_sync_job, *post_sync_job;
		wrapper->pre_sync_task = starpu_task_create();
		wrapper->pre_sync_task->name = "_starpu_data_acquire_cb_pre";
		wrapper->pre_sync_task->detach = 1;
		wrapper->pre_sync_task->callback_func = starpu_data_acquire_cb_pre_sync_callback;
		wrapper->pre_sync_task->callback_arg = wrapper;
		wrapper->pre_sync_task->type = STARPU_TASK_TYPE_DATA_ACQUIRE;
		pre_sync_job = _starpu_get_job_associated_to_task(wrapper->pre_sync_task);
		if (pre_sync_jobid)
			*pre_sync_jobid = pre_sync_job->job_id;

		wrapper->post_sync_task = starpu_task_create();
		wrapper->post_sync_task->name = "_starpu_data_acquire_cb_release";
		wrapper->post_sync_task->detach = 1;
		wrapper->post_sync_task->type = STARPU_TASK_TYPE_DATA_ACQUIRE;
		post_sync_job = _starpu_get_job_associated_to_task(wrapper->post_sync_task);
		if (post_sync_jobid)
			*post_sync_jobid = post_sync_job->job_id;

		if (quick)
			pre_sync_job->quick_next = post_sync_job;

		new_task = _starpu_detect_implicit_data_deps_with_handle(wrapper->pre_sync_task, wrapper->post_sync_task, &_starpu_get_job_associated_to_task(wrapper->post_sync_task)->implicit_dep_slot, handle, mode, sequential_consistency);
		STARPU_PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);

		if (new_task)
		{
			int ret = _starpu_task_submit_internally(new_task);
			STARPU_ASSERT(!ret);
		}

		/* TODO detect if this is superflous */
		int ret = _starpu_task_submit_internally(wrapper->pre_sync_task);
		STARPU_ASSERT(!ret);
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

int starpu_data_acquire_on_node_cb_sequential_consistency_quick(starpu_data_handle_t handle, int node,
							  enum starpu_data_access_mode mode, void (*callback)(void *), void *arg,
							  int sequential_consistency, int quick)
{
	return starpu_data_acquire_on_node_cb_sequential_consistency_sync_jobids(handle, node, mode, callback, arg, sequential_consistency, quick, NULL, NULL);
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


/*
 *	Blocking data request from application
 */



static inline void _starpu_data_acquire_continuation(void *arg)
{
	struct user_interaction_wrapper *wrapper = (struct user_interaction_wrapper *) arg;

	starpu_data_handle_t handle = wrapper->handle;
	STARPU_ASSERT(handle);

	_starpu_data_acquire_launch_fetch(wrapper, 0, NULL, NULL);
	_starpu_data_acquire_fetch_done(wrapper);
	_starpu_data_acquire_wrapper_finished(wrapper);
}

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
		wrapper.pre_sync_task = starpu_task_create();
		wrapper.pre_sync_task->name = "_starpu_data_acquire_pre";
		wrapper.pre_sync_task->detach = 0;
		wrapper.pre_sync_task->type = STARPU_TASK_TYPE_DATA_ACQUIRE;

		wrapper.post_sync_task = starpu_task_create();
		wrapper.post_sync_task->name = "_starpu_data_acquire_post";
		wrapper.post_sync_task->detach = 1;
		wrapper.post_sync_task->type = STARPU_TASK_TYPE_DATA_ACQUIRE;

		new_task = _starpu_detect_implicit_data_deps_with_handle(wrapper.pre_sync_task, wrapper.post_sync_task, &_starpu_get_job_associated_to_task(wrapper.post_sync_task)->implicit_dep_slot, handle, mode, sequential_consistency);
		STARPU_PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);
		if (new_task)
		{
			int ret = _starpu_task_submit_internally(new_task);
			STARPU_ASSERT(!ret);
		}

		/* TODO detect if this is superflous */
		wrapper.pre_sync_task->synchronous = 1;
		int ret = _starpu_task_submit_internally(wrapper.pre_sync_task);
		STARPU_ASSERT(!ret);
	}
	else
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&handle->sequential_consistency_mutex);
	}

	/* we try to get the data, if we do not succeed immediately, we set a
 	* callback function that will be executed automatically when the data is
 	* available again, otherwise we fetch the data directly */
	if (!_starpu_attempt_to_submit_data_request_from_apps(handle, mode, _starpu_data_acquire_continuation, &wrapper))
	{
		/* no one has locked this data yet, so we proceed immediately */
		_starpu_data_acquire_launch_fetch(&wrapper, 0, NULL, NULL);
		_starpu_data_acquire_fetch_done(&wrapper);
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

	/* we try to get the data, if we do not succeed immediately, we set a
 	* callback function that will be executed automatically when the data is
 	* available again, otherwise we fetch the data directly */
	if (!_starpu_attempt_to_submit_data_request_from_apps(handle, mode, _starpu_data_acquire_continuation, &wrapper))
	{
		/* no one has locked this data yet, so we proceed immediately */
		_starpu_data_acquire_launch_fetch(&wrapper, 0, NULL, NULL);
		_starpu_data_acquire_fetch_done(&wrapper);
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
void starpu_data_release_on_node(starpu_data_handle_t handle, int node)
{
	STARPU_ASSERT(handle);

	/* In case there are some implicit dependencies, unlock the "post sync" tasks */
	_starpu_unlock_post_sync_tasks(handle);

	/* The application can now release the rw-lock */
	if (node >= 0)
		_starpu_release_data_on_node(handle, 0, &handle->per_node[node]);
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
		if (!_starpu_notify_data_dependencies(handle))
			_starpu_spin_unlock(&handle->header_lock);
	}
}

void starpu_data_release(starpu_data_handle_t handle)
{
	int home_node = handle->home_node;
	if (home_node < 0)
		home_node = STARPU_MAIN_RAM;
	starpu_data_release_on_node(handle, home_node);
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
	if (!_starpu_notify_data_dependencies(handle))
		_starpu_spin_unlock(&handle->header_lock);
}

static
int _starpu_prefetch_data_on_node_with_mode(starpu_data_handle_t handle, unsigned node, unsigned async, enum starpu_data_access_mode mode, enum _starpu_is_prefetch prefetch, int prio)
{
	STARPU_ASSERT(handle);

	/* it is forbidden to call this function from a callback or a codelet */
	STARPU_ASSERT_MSG(async || _starpu_worker_may_perform_blocking_calls(), "Synchronous prefetch is not possible from a task or a callback");

	/* Check that previous tasks have set a value if needed */
	_starpu_data_check_initialized(handle, mode);

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
		if (!_starpu_notify_data_dependencies(handle))
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

int starpu_data_fetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async)
{
	return _starpu_prefetch_data_on_node_with_mode(handle, node, async, STARPU_R, STARPU_FETCH, 0);
}

int starpu_data_prefetch_on_node_prio(starpu_data_handle_t handle, unsigned node, unsigned async, int prio)
{
	return _starpu_prefetch_data_on_node_with_mode(handle, node, async, STARPU_R, STARPU_PREFETCH, prio);
}

int starpu_data_prefetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async)
{
	return starpu_data_prefetch_on_node_prio(handle, node, async, 0);
}

int starpu_data_idle_prefetch_on_node_prio(starpu_data_handle_t handle, unsigned node, unsigned async, int prio)
{
	return _starpu_prefetch_data_on_node_with_mode(handle, node, async, STARPU_R, STARPU_IDLEFETCH, prio);
}

int starpu_data_idle_prefetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async)
{
	return starpu_data_idle_prefetch_on_node_prio(handle, node, async, 0);
}

static void _starpu_data_wont_use(void *data)
{
	unsigned node;
	starpu_data_handle_t handle = data;

	_STARPU_TRACE_DATA_DOING_WONT_USE(handle);

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
		starpu_data_idle_prefetch_on_node(handle, handle->home_node, 1);
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
					starpu_data_idle_prefetch_on_node(handle, i, 1);
			}
		}
	}
}

void starpu_data_wont_use(starpu_data_handle_t handle)
{
	if (!handle->initialized)
		/* No value atm actually */
		return;
	_STARPU_TRACE_DATA_WONT_USE(handle);
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
void starpu_data_query_status(starpu_data_handle_t handle, int memory_node, int *is_allocated, int *is_valid, int *is_requested)
{
// XXX : this is just a hint, so we don't take the lock ...
//	_starpu_spin_lock(&handle->header_lock);

	if (is_allocated)
		*is_allocated = handle->per_node[memory_node].allocated;

	if (is_valid)
		*is_valid = (handle->per_node[memory_node].state != STARPU_INVALID);

	if (is_requested)
	{
		int requested = 0;

		unsigned node;
		for (node = 0; node < STARPU_MAXNODES; node++)
		{
			if (handle->per_node[memory_node].requested & (1UL << node))
			{
				requested = 1;
				break;
			}
		}

		*is_requested = requested;
	}

//	_starpu_spin_unlock(&handle->header_lock);
}
