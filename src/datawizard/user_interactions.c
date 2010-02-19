/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <common/config.h>
#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>
#include <datawizard/write_back.h>
#include <core/dependencies/data-concurrency.h>

int starpu_request_data_allocation(starpu_data_handle handle, uint32_t node)
{
	starpu_data_request_t r;

	STARPU_ASSERT(handle);

	r = _starpu_create_data_request(handle, 0, node, node, 0, 0, 1);

	/* we do not increase the refcnt associated to the request since we are
	 * not waiting for its termination */

	_starpu_post_data_request(r, node);

	return 0;
}

struct state_and_node {
	starpu_data_handle state;
	starpu_access_mode mode;
	unsigned node;
	pthread_cond_t cond;
	pthread_mutex_t lock;
	unsigned finished;
	unsigned async;
	unsigned non_blocking;
	void (*callback)(void *);
	void *callback_arg;
};

/* put the current value of the data into STARPU_RAM */
static inline void _starpu_sync_data_with_mem_continuation(void *arg)
{
	int ret;
	struct state_and_node *statenode = arg;

	starpu_data_handle handle = statenode->state;

	STARPU_ASSERT(handle);

	unsigned r = (statenode->mode != STARPU_W);
	unsigned w = (statenode->mode != STARPU_R);

	ret = _starpu_fetch_data_on_node(handle, 0, r, w, 0);
	STARPU_ASSERT(!ret);
	
	if (statenode->non_blocking)
	{
		/* continuation of starpu_sync_data_with_mem_non_blocking: we
		 * execute the callback if any  */
		if (statenode->callback)
			statenode->callback(statenode->callback_arg);

		free(statenode);
	}
	else {
		/* continuation of starpu_sync_data_with_mem */
		pthread_mutex_lock(&statenode->lock);
		statenode->finished = 1;
		pthread_cond_signal(&statenode->cond);
		pthread_mutex_unlock(&statenode->lock);
	}
}

/* The data must be released by calling starpu_release_data_from_mem later on */
int starpu_sync_data_with_mem(starpu_data_handle handle, starpu_access_mode mode)
{
	STARPU_ASSERT(handle);

	/* it is forbidden to call this function from a callback or a codelet */
	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
		return -EDEADLK;

	struct state_and_node statenode =
	{
		.state = handle,
		.mode = mode,
		.node = 0, // unused
		.non_blocking = 0,
		.cond = PTHREAD_COND_INITIALIZER,
		.lock = PTHREAD_MUTEX_INITIALIZER,
		.finished = 0
	};

	/* we try to get the data, if we do not succeed immediately, we set a
 	* callback function that will be executed automatically when the data is
 	* available again, otherwise we fetch the data directly */
	if (!attempt_to_submit_data_request_from_apps(handle, mode,
			_starpu_sync_data_with_mem_continuation, &statenode))
	{
		/* no one has locked this data yet, so we proceed immediately */
		_starpu_sync_data_with_mem_continuation(&statenode);
	}
	else {
		pthread_mutex_lock(&statenode.lock);
		if (!statenode.finished)
			pthread_cond_wait(&statenode.cond, &statenode.lock);
		pthread_mutex_unlock(&statenode.lock);
	}

	return 0;
}

/* The data must be released by calling starpu_release_data_from_mem later on */
int starpu_sync_data_with_mem_non_blocking(starpu_data_handle handle,
		starpu_access_mode mode, void (*callback)(void *), void *arg)
{
	STARPU_ASSERT(handle);

	struct state_and_node *statenode = malloc(sizeof(struct state_and_node));
	STARPU_ASSERT(statenode);

	statenode->state = handle;
	statenode->mode = mode;
	statenode->non_blocking = 1;
	statenode->callback = callback;
	statenode->callback_arg = arg;
	pthread_cond_init(&statenode->cond, NULL);
	pthread_mutex_init(&statenode->lock, NULL);
	statenode->finished = 0;

	/* we try to get the data, if we do not succeed immediately, we set a
 	* callback function that will be executed automatically when the data is
 	* available again, otherwise we fetch the data directly */
	if (!attempt_to_submit_data_request_from_apps(handle, mode,
			_starpu_sync_data_with_mem_continuation, statenode))
	{
		/* no one has locked this data yet, so we proceed immediately */
		_starpu_sync_data_with_mem_continuation(statenode);
	}

	return 0;
}

/* This function must be called after starpu_sync_data_with_mem so that the
 * application release the data */
void starpu_release_data_from_mem(starpu_data_handle handle)
{
	STARPU_ASSERT(handle);

	/* The application can now release the rw-lock */
	_starpu_release_data_on_node(handle, 0, 0);
}



static void _prefetch_data_on_node(void *arg)
{
	struct state_and_node *statenode = arg;

	_starpu_fetch_data_on_node(statenode->state, statenode->node, 1, 0, statenode->async);

	pthread_mutex_lock(&statenode->lock);
	statenode->finished = 1;
	pthread_cond_signal(&statenode->cond);
	pthread_mutex_unlock(&statenode->lock);

	if (!statenode->async)
	{
		starpu_spin_lock(&statenode->state->header_lock);
		notify_data_dependencies(statenode->state);
		starpu_spin_unlock(&statenode->state->header_lock);
	}

}

int _starpu_prefetch_data_on_node_with_mode(starpu_data_handle handle, unsigned node, unsigned async, starpu_access_mode mode)
{
	STARPU_ASSERT(handle);

	/* it is forbidden to call this function from a callback or a codelet */
	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
		return -EDEADLK;

	struct state_and_node statenode =
	{
		.state = handle,
		.node = node,
		.async = async,
		.cond = PTHREAD_COND_INITIALIZER,
		.lock = PTHREAD_MUTEX_INITIALIZER,
		.finished = 0
	};

	if (!attempt_to_submit_data_request_from_apps(handle, mode, _prefetch_data_on_node, &statenode))
	{
		/* we can immediately proceed */
		uint8_t read = (mode != STARPU_W);
		uint8_t write = (mode != STARPU_R);
		_starpu_fetch_data_on_node(handle, node, read, write, async);

		/* remove the "lock"/reference */
		if (!async)
		{
			starpu_spin_lock(&handle->header_lock);
			notify_data_dependencies(handle);
			starpu_spin_unlock(&handle->header_lock);
		}
	}
	else {
		pthread_mutex_lock(&statenode.lock);
		if (!statenode.finished)
			pthread_cond_wait(&statenode.cond, &statenode.lock);
		pthread_mutex_unlock(&statenode.lock);
	}

	return 0;
}

int starpu_prefetch_data_on_node(starpu_data_handle handle, unsigned node, unsigned async)
{
	return _starpu_prefetch_data_on_node_with_mode(handle, node, async, STARPU_R);
}
