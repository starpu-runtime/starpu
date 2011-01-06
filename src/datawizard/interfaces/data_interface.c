/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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

#include <datawizard/datawizard.h>
#include <core/dependencies/data_concurrency.h>

/* 
 * Start monitoring a piece of data
 */

static void _starpu_register_new_data(starpu_data_handle handle,
					uint32_t home_node, uint32_t wt_mask)
{
	STARPU_ASSERT(handle);

	/* initialize the new lock */
	handle->req_list = starpu_data_requester_list_new();
	handle->refcnt = 0;
	_starpu_spin_init(&handle->header_lock);

	/* first take care to properly lock the data */
	_starpu_spin_lock(&handle->header_lock);

	/* there is no hierarchy yet */
	handle->nchildren = 0;
	handle->root_handle = handle;
	handle->father_handle = NULL;
	handle->sibling_index = 0; /* could be anything for the root */
	handle->depth = 1; /* the tree is just a node yet */

	handle->is_not_important = 0;

	handle->sequential_consistency =
		starpu_data_get_default_sequential_consistency_flag();

	PTHREAD_MUTEX_INIT(&handle->sequential_consistency_mutex, NULL);
	handle->last_submitted_mode = STARPU_R;
	handle->last_submitted_writer = NULL;
	handle->last_submitted_readers = NULL;
	handle->post_sync_tasks = NULL;
	handle->post_sync_tasks_cnt = 0;

	/* By default, there are no methods available to perform a reduction */
	handle->redux_cl = NULL;
	handle->init_cl = NULL;

	handle->reduction_refcnt = 0;
	handle->reduction_req_list = starpu_data_requester_list_new();

#ifdef STARPU_USE_FXT
	handle->last_submitted_ghost_writer_id_is_valid = 0;
	handle->last_submitted_ghost_writer_id = 0;
	handle->last_submitted_ghost_readers_id = NULL;
#endif

	handle->wt_mask = wt_mask;

	/* Store some values directly in the handle not to recompute them all
	 * the time. */
	handle->data_size = handle->ops->get_size(handle);
	handle->footprint = _starpu_compute_data_footprint(handle);

	handle->home_node = home_node;

	/* that new data is invalid from all nodes perpective except for the
	 * home node */
	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_data_replicate_s *replicate;
		replicate = &handle->per_node[node];
		
		replicate->memory_node = node;
		replicate->relaxed_coherency = 0;
		replicate->refcnt = 0;

		if (node == home_node) {
			/* this is the home node with the only valid copy */
			replicate->state = STARPU_OWNER;
			replicate->allocated = 1;
			replicate->automatically_allocated = 0;
		}
		else {
			/* the value is not available here yet */
			replicate->state = STARPU_INVALID;
			replicate->allocated = 0;
		}
	}

	unsigned worker;
	unsigned nworkers = starpu_worker_get_count();
	for (worker = 0; worker < nworkers; worker++)
	{
		struct starpu_data_replicate_s *replicate;
		replicate = &handle->per_worker[worker];
		replicate->allocated = 0;
		replicate->automatically_allocated = 0;
		replicate->state = STARPU_INVALID;
		replicate->refcnt = 0;
		replicate->handle = handle;
		replicate->requested = 0;
		replicate->request = NULL;
		replicate->relaxed_coherency = 1;
		replicate->initialized = 0;
		replicate->memory_node = starpu_worker_get_memory_node(worker);

		/* duplicate  the content of the interface on node 0 */
		memcpy(replicate->interface, handle->per_node[0].interface, handle->ops->interface_size);
	} 

	/* now the data is available ! */
	_starpu_spin_unlock(&handle->header_lock);
}

static starpu_data_handle _starpu_data_handle_allocate(struct starpu_data_interface_ops_t *interface_ops)
{
	starpu_data_handle handle =
		calloc(1, sizeof(struct starpu_data_state_t));

	STARPU_ASSERT(handle);

	handle->ops = interface_ops;

	size_t interfacesize = interface_ops->interface_size;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_data_replicate_s *replicate;
		replicate = &handle->per_node[node];
		/* relaxed_coherency = 0 */

		replicate->handle = handle;

		replicate->interface = calloc(1, interfacesize);
		STARPU_ASSERT(replicate->interface);
	}

	unsigned worker;
	unsigned nworkers = starpu_worker_get_count();
	for (worker = 0; worker < nworkers; worker++)
	{
		struct starpu_data_replicate_s *replicate;
		replicate = &handle->per_worker[worker];

		replicate->handle = handle;

		replicate->interface = calloc(1, interfacesize);
		STARPU_ASSERT(replicate->interface);

	}

	return handle;
}

void starpu_data_register(starpu_data_handle *handleptr, uint32_t home_node,
				void *interface,
				struct starpu_data_interface_ops_t *ops)
{
	starpu_data_handle handle =
		_starpu_data_handle_allocate(ops);

	STARPU_ASSERT(handleptr);
	*handleptr = handle;


	/* fill the interface fields with the appropriate method */
	ops->register_data_handle(handle, home_node, interface);

	_starpu_register_new_data(handle, home_node, 0);
}

/* 
 * Stop monitoring a piece of data
 */

void _starpu_data_free_interfaces(starpu_data_handle handle)
{
	unsigned node;
	unsigned worker;
	unsigned nworkers = starpu_worker_get_count();

	for (node = 0; node < STARPU_MAXNODES; node++)
		free(handle->per_node[node].interface);

	for (worker = 0; worker < nworkers; worker++)
		free(handle->per_worker[worker].interface);
}

struct unregister_callback_arg {
	unsigned memory_node;
	starpu_data_handle handle;
	unsigned terminated;
	pthread_mutex_t mutex;
	pthread_cond_t cond;
}; 

static void _starpu_data_unregister_fetch_data_callback(void *_arg)
{
	int ret;
	struct unregister_callback_arg *arg = _arg;

	starpu_data_handle handle = arg->handle;

	STARPU_ASSERT(handle);

	struct starpu_data_replicate_s *replicate = &handle->per_node[arg->memory_node];

	ret = _starpu_fetch_data_on_node(handle, replicate, STARPU_R, 0, NULL, NULL);
	STARPU_ASSERT(!ret);
	
	/* unlock the caller */
	PTHREAD_MUTEX_LOCK(&arg->mutex);
	arg->terminated = 1;
	PTHREAD_COND_SIGNAL(&arg->cond);
	PTHREAD_MUTEX_UNLOCK(&arg->mutex);
}

/* Unregister the data handle, perhaps we don't need to update the home_node
 * (in that case coherent is set to 0) */
static void _starpu_data_unregister(starpu_data_handle handle, unsigned coherent)
{
	STARPU_ASSERT(handle);

	if (coherent)
	{
		/* If sequential consistency is enabled, wait until data is available */
		_starpu_data_wait_until_available(handle, STARPU_RW);

		/* Fetch data in the home of the data to ensure we have a valid copy
		 * where we registered it */
		int home_node = handle->home_node; 
		if (home_node >= 0)
		{
			struct unregister_callback_arg arg;
			arg.handle = handle;
			arg.memory_node = (unsigned)home_node;
			arg.terminated = 0;
			PTHREAD_MUTEX_INIT(&arg.mutex, NULL);
			PTHREAD_COND_INIT(&arg.cond, NULL);
	
			if (!_starpu_attempt_to_submit_data_request_from_apps(handle, STARPU_R,
					_starpu_data_unregister_fetch_data_callback, &arg))
			{
				/* no one has locked this data yet, so we proceed immediately */
				struct starpu_data_replicate_s *home_replicate = &handle->per_node[home_node];
				int ret = _starpu_fetch_data_on_node(handle, home_replicate, STARPU_R, 0, NULL, NULL);
				STARPU_ASSERT(!ret);
			}
			else {
				PTHREAD_MUTEX_LOCK(&arg.mutex);
				while (!arg.terminated)
					PTHREAD_COND_WAIT(&arg.cond, &arg.mutex);
				PTHREAD_MUTEX_UNLOCK(&arg.mutex);
			}
		}
	}
	else {
		/* Should we postpone the unregister operation ? */
		if ((handle->refcnt > 0) && handle->lazy_unregister)
			return;
	}

	_starpu_data_free_interfaces(handle);

	/* Destroy the data now */
	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_data_replicate_s *local = &handle->per_node[node];

		if (local->allocated && local->automatically_allocated){
			/* free the data copy in a lazy fashion */
			_starpu_request_mem_chunk_removal(handle, node);
		}
	}

	starpu_data_requester_list_delete(handle->req_list);
	starpu_data_requester_list_delete(handle->reduction_req_list);

	free(handle);
}

void starpu_data_unregister(starpu_data_handle handle)
{
	_starpu_data_unregister(handle, 1);
}

void starpu_data_unregister_no_coherency(starpu_data_handle handle)
{
	_starpu_data_unregister(handle, 0);
}

void starpu_data_invalidate(starpu_data_handle handle)
{
	STARPU_ASSERT(handle);

	starpu_data_acquire(handle, STARPU_W);

	_starpu_spin_lock(&handle->header_lock);

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_data_replicate_s *local = &handle->per_node[node];

		if (local->allocated && local->automatically_allocated){
			/* free the data copy in a lazy fashion */
			_starpu_request_mem_chunk_removal(handle, node);
		}

		local->state = STARPU_INVALID; 
	}

	_starpu_spin_unlock(&handle->header_lock);

	starpu_data_release(handle);
}

unsigned starpu_get_handle_interface_id(starpu_data_handle handle)
{
	return handle->ops->interfaceid;
}

void *starpu_data_get_interface_on_node(starpu_data_handle handle, unsigned memory_node)
{
	return handle->per_node[memory_node].interface;
}
