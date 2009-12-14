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

int starpu_request_data_allocation(data_state *state, uint32_t node)
{
	data_request_t r;

	r = create_data_request(state, 0, node, node, 0, 0, 1);

	/* we do not increase the refcnt associated to the request since we are
	 * not waiting for its termination */

	post_data_request(r, node);

	return 0;
}


struct state_and_node {
	data_state *state;
	unsigned node;
	pthread_cond_t cond;
	pthread_mutex_t lock;
	unsigned finished;
	unsigned async;
};

/* put the current value of the data into RAM */
static inline void _starpu_sync_data_with_mem_continuation(void *arg)
{
	int ret;
	struct state_and_node *statenode = arg;

	data_state *state = statenode->state;

	ret = fetch_data_on_node(state, 0, 1, 0, 0);
	
	STARPU_ASSERT(!ret);
	
	/* the application does not need to "lock" the data anymore */
	release_data_on_node(state, 0, 0);

	pthread_mutex_lock(&statenode->lock);
	statenode->finished = 1;
	pthread_cond_signal(&statenode->cond);
	pthread_mutex_unlock(&statenode->lock);
}

int starpu_sync_data_with_mem(data_state *state)
{
	/* it is forbidden to call this function from a callback or a codelet */
	if (STARPU_UNLIKELY(!worker_may_perform_blocking_calls()))
		return -EDEADLK;

	struct state_and_node statenode =
	{
		.state = state,
		.node = 0, /* unused here */
		.cond = PTHREAD_COND_INITIALIZER,
		.lock = PTHREAD_MUTEX_INITIALIZER,
		.finished = 0
	};

	/* we try to get the data, if we do not succeed immediately, we set a
 	* callback function that will be executed automatically when the data is
 	* available again, otherwise we fetch the data directly */
	if (!attempt_to_submit_data_request_from_apps(state, STARPU_R, 
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

static inline void do_notify_data_modification(data_state *state, uint32_t modifying_node)
{
	starpu_spin_lock(&state->header_lock);

	unsigned node = 0;
	for (node = 0; node < MAXNODES; node++)
	{
		state->per_node[node].state =
			(node == modifying_node?OWNER:INVALID);
	}

	starpu_spin_unlock(&state->header_lock);
}

static inline void _notify_data_modification_continuation(void *arg)
{
	struct state_and_node *statenode = arg;

	do_notify_data_modification(statenode->state, statenode->node);

	pthread_mutex_lock(&statenode->lock);
	statenode->finished = 1;
	pthread_cond_signal(&statenode->cond);
	pthread_mutex_unlock(&statenode->lock);
}

/* in case the application did modify the data ... invalidate all other copies  */
int starpu_notify_data_modification(data_state *state, uint32_t modifying_node)
{
	/* It is forbidden to call this function from a callback or a codelet */
	if (STARPU_UNLIKELY(!worker_may_perform_blocking_calls()))
		return -EDEADLK;

	struct state_and_node statenode =
	{
		.state = state,
		.node = modifying_node,
		.cond = PTHREAD_COND_INITIALIZER,
		.lock = PTHREAD_MUTEX_INITIALIZER,
		.finished = 0
	};

	if (!attempt_to_submit_data_request_from_apps(state, STARPU_W, _notify_data_modification_continuation, &statenode))
	{
		/* we can immediately proceed */
		do_notify_data_modification(state, modifying_node);
	}
	else {
		pthread_mutex_lock(&statenode.lock);
		if (!statenode.finished)
			pthread_cond_wait(&statenode.cond, &statenode.lock);
		pthread_mutex_unlock(&statenode.lock);
	}

	/* remove the "lock"/reference */
	notify_data_dependencies(state);

	return 0;
}

static void _prefetch_data_on_node(void *arg)
{
	struct state_and_node *statenode = arg;

	fetch_data_on_node(statenode->state, statenode->node, 1, 0, statenode->async);

	pthread_mutex_lock(&statenode->lock);
	statenode->finished = 1;
	pthread_cond_signal(&statenode->cond);
	pthread_mutex_unlock(&statenode->lock);

	if (!statenode->async)
		notify_data_dependencies(statenode->state);

}

int starpu_prefetch_data_on_node(data_state *state, unsigned node, unsigned async)
{
	/* it is forbidden to call this function from a callback or a codelet */
	if (STARPU_UNLIKELY(!worker_may_perform_blocking_calls()))
		return -EDEADLK;

	struct state_and_node statenode =
	{
		.state = state,
		.node = node,
		.async = async,
		.cond = PTHREAD_COND_INITIALIZER,
		.lock = PTHREAD_MUTEX_INITIALIZER,
		.finished = 0
	};

	if (!attempt_to_submit_data_request_from_apps(state, STARPU_R, _prefetch_data_on_node, &statenode))
	{
		/* we can immediately proceed */
		fetch_data_on_node(state, node, 1, 0, async);

		/* remove the "lock"/reference */
		if (!async)
			notify_data_dependencies(state);
	}
	else {
		pthread_mutex_lock(&statenode.lock);
		if (!statenode.finished)
			pthread_cond_wait(&statenode.cond, &statenode.lock);
		pthread_mutex_unlock(&statenode.lock);
	}

	return 0;
}
