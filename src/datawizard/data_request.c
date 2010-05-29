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

#include <starpu.h>
#include <common/config.h>
#include <datawizard/datawizard.h>

/* requests that have not been treated at all */
static starpu_data_request_list_t data_requests[STARPU_MAXNODES];
static pthread_cond_t data_requests_list_cond[STARPU_MAXNODES];
static pthread_mutex_t data_requests_list_mutex[STARPU_MAXNODES];

/* requests that are not terminated (eg. async transfers) */
static starpu_data_request_list_t data_requests_pending[STARPU_MAXNODES];
static pthread_cond_t data_requests_pending_list_cond[STARPU_MAXNODES];
static pthread_mutex_t data_requests_pending_list_mutex[STARPU_MAXNODES];

void _starpu_init_data_request_lists(void)
{
	unsigned i;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		data_requests[i] = starpu_data_request_list_new();
		PTHREAD_MUTEX_INIT(&data_requests_list_mutex[i], NULL);
		PTHREAD_COND_INIT(&data_requests_list_cond[i], NULL);

		data_requests_pending[i] = starpu_data_request_list_new();
		PTHREAD_MUTEX_INIT(&data_requests_pending_list_mutex[i], NULL);
		PTHREAD_COND_INIT(&data_requests_pending_list_cond[i], NULL);
	}
}

void _starpu_deinit_data_request_lists(void)
{
	unsigned i;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		PTHREAD_COND_DESTROY(&data_requests_pending_list_cond[i]);
		PTHREAD_MUTEX_DESTROY(&data_requests_pending_list_mutex[i]);
		starpu_data_request_list_delete(data_requests_pending[i]);

		PTHREAD_COND_DESTROY(&data_requests_list_cond[i]);
		PTHREAD_MUTEX_DESTROY(&data_requests_list_mutex[i]);
		starpu_data_request_list_delete(data_requests[i]);
	}
}

/* this should be called with the lock r->handle->header_lock taken */
static void starpu_data_request_destroy(starpu_data_request_t r)
{
	r->handle->per_node[r->dst_node].request = NULL;

	starpu_data_request_delete(r);
}

/* handle->lock should already be taken !  */
starpu_data_request_t _starpu_create_data_request(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node, uint32_t handling_node, starpu_access_mode mode, unsigned is_prefetch)
{
	starpu_data_request_t r = starpu_data_request_new();

	_starpu_spin_init(&r->lock);

	r->handle = handle;
	r->src_node = src_node;
	r->dst_node = dst_node;
	r->mode = mode;

	r->handling_node = handling_node;

	r->completed = 0;
	r->retval = -1;

	r->next_req_count = 0;

	r->callbacks = NULL;

	r->is_a_prefetch_request = is_prefetch;

	/* associate that request with the handle so that further similar
	 * requests will reuse that one  */

	_starpu_spin_lock(&r->lock);

	handle->per_node[dst_node].request = r;

	handle->per_node[dst_node].refcnt++;

	if (mode & STARPU_R)
		handle->per_node[src_node].refcnt++;

	r->refcnt = 1;

	_starpu_spin_unlock(&r->lock);

	return r;
}

/* handle->lock should be taken */
starpu_data_request_t _starpu_search_existing_data_request(starpu_data_handle handle, uint32_t dst_node, starpu_access_mode mode)
{
	starpu_data_request_t r = handle->per_node[dst_node].request;

	if (r)
	{
		_starpu_spin_lock(&r->lock);

		/* perhaps we need to "upgrade" the request */
		if (mode & STARPU_R)
		{
			/* in case the exisiting request did not imply a memory
			 * transfer yet, we have to increment the refcnt now
			 * (so that the source remains valid) */
			if (!(r->mode & STARPU_R))
				handle->per_node[dst_node].refcnt++;

			r->mode |= STARPU_R;
		}

		if (mode & STARPU_W)
			r->mode |= STARPU_W;
	}

	return r;
}

int _starpu_wait_data_request_completion(starpu_data_request_t r, unsigned may_alloc)
{
	int retval;
	int do_delete = 0;

	uint32_t local_node = _starpu_get_local_memory_node();

	do {
		_starpu_spin_lock(&r->lock);

		if (r->completed)
			break;

		_starpu_spin_unlock(&r->lock);

		_starpu_wake_all_blocked_workers_on_node(r->handling_node);

		_starpu_datawizard_progress(local_node, may_alloc);

	} while (1);


	retval = r->retval;
	if (retval)
		fprintf(stderr, "REQUEST %p COMPLETED (retval %d) !\n", r, r->retval);
		

	r->refcnt--;

	/* if nobody is waiting on that request, we can get rid of it */
	if (r->refcnt == 0)
		do_delete = 1;

	_starpu_spin_unlock(&r->lock);
	
	if (do_delete)
		starpu_data_request_destroy(r);
	
	return retval;
}

/* this is non blocking */
void _starpu_post_data_request(starpu_data_request_t r, uint32_t handling_node)
{
//	fprintf(stderr, "POST REQUEST\n");

	if (r->mode & STARPU_R)
	{
		STARPU_ASSERT(r->handle->per_node[r->src_node].allocated);
		STARPU_ASSERT(r->handle->per_node[r->src_node].refcnt);
	}

	/* insert the request in the proper list */
	PTHREAD_MUTEX_LOCK(&data_requests_list_mutex[handling_node]);

	starpu_data_request_list_push_front(data_requests[handling_node], r);

	PTHREAD_MUTEX_UNLOCK(&data_requests_list_mutex[handling_node]);

	_starpu_wake_all_blocked_workers_on_node(handling_node);
}

/* We assume that r->lock is taken by the caller */
void _starpu_data_request_append_callback(starpu_data_request_t r, void (*callback_func)(void *), void *callback_arg)
{
	STARPU_ASSERT(r);

	if (callback_func)
	{
		struct callback_list *link = malloc(sizeof(struct callback_list));
		STARPU_ASSERT(link);

		link->callback_func = callback_func;
		link->callback_arg = callback_arg;
		link->next = r->callbacks;
		r->callbacks = link;
	}
}

static void starpu_handle_data_request_completion(starpu_data_request_t r)
{
	unsigned do_delete = 0;
	starpu_data_handle handle = r->handle;

	uint32_t src_node = r->src_node;
	uint32_t dst_node = r->dst_node;

	_starpu_update_data_state(handle, dst_node, r->mode);

#ifdef STARPU_USE_FXT
	size_t size = handle->ops->get_size(handle);
	STARPU_TRACE_END_DRIVER_COPY(src_node, dst_node, size, r->com_id);
#endif

	unsigned chained_req;
	for (chained_req = 0; chained_req < r->next_req_count; chained_req++)
	{
		_starpu_post_data_request(r->next_req[chained_req], r->next_req[chained_req]->handling_node);
	}

	r->completed = 1;
	
	handle->per_node[dst_node].refcnt--;

	if (r->mode & STARPU_R)
		handle->per_node[src_node].refcnt--;

	r->refcnt--;

	/* if nobody is waiting on that request, we can get rid of it */
	if (r->refcnt == 0)
		do_delete = 1;
	
	r->retval = 0;

	/* In case there are one or multiple callbacks, we execute them now. */
	struct callback_list *callbacks = r->callbacks;
	
	_starpu_spin_unlock(&r->lock);

	if (do_delete)
		starpu_data_request_destroy(r);

	_starpu_spin_unlock(&handle->header_lock);

	/* We do the callback once the lock is released so that they can do
	 * blocking operations with the handle (eg. release it) */
	while (callbacks)
	{
		callbacks->callback_func(callbacks->callback_arg);

		struct callback_list *next = callbacks->next;
		free(callbacks);
		callbacks = next;
	}
}

/* TODO : accounting to see how much time was spent working for other people ... */
static int starpu_handle_data_request(starpu_data_request_t r, unsigned may_alloc)
{
	starpu_data_handle handle = r->handle;

	_starpu_spin_lock(&handle->header_lock);

	_starpu_spin_lock(&r->lock);

	if (r->mode & STARPU_R)
	{
		STARPU_ASSERT(handle->per_node[r->src_node].allocated);
		STARPU_ASSERT(handle->per_node[r->src_node].refcnt);
	}

	/* perform the transfer */
	/* the header of the data must be locked by the worker that submitted the request */
	r->retval = _starpu_driver_copy_data_1_to_1(handle, r->src_node, r->dst_node, !(r->mode & STARPU_R), r, may_alloc);

	if (r->retval == ENOMEM)
	{
		_starpu_spin_unlock(&r->lock);
		_starpu_spin_unlock(&handle->header_lock);

		return ENOMEM;
	}

	if (r->retval == EAGAIN)
	{
		_starpu_spin_unlock(&r->lock);
		_starpu_spin_unlock(&handle->header_lock);

		/* the request is pending and we put it in the corresponding queue  */
		PTHREAD_MUTEX_LOCK(&data_requests_pending_list_mutex[r->handling_node]);
		starpu_data_request_list_push_front(data_requests_pending[r->handling_node], r);
		PTHREAD_MUTEX_UNLOCK(&data_requests_pending_list_mutex[r->handling_node]);

		return EAGAIN;
	}

	/* the request has been handled */
	starpu_handle_data_request_completion(r);

	return 0;
}

void _starpu_handle_node_data_requests(uint32_t src_node, unsigned may_alloc)
{
	/* for all entries of the list */
	starpu_data_request_t r;

	/* take all the entries from the request list */
        PTHREAD_MUTEX_LOCK(&data_requests_list_mutex[src_node]);

	starpu_data_request_list_t local_list = data_requests[src_node];

	if (starpu_data_request_list_empty(local_list))
	{
		/* there is no request */
                PTHREAD_MUTEX_UNLOCK(&data_requests_list_mutex[src_node]);

		return;
	}

	data_requests[src_node] = starpu_data_request_list_new();

	PTHREAD_MUTEX_UNLOCK(&data_requests_list_mutex[src_node]);

	while (!starpu_data_request_list_empty(local_list))
	{
                int res;

		r = starpu_data_request_list_pop_back(local_list);

		res = starpu_handle_data_request(r, may_alloc);
		if (res == ENOMEM)
		{
                        PTHREAD_MUTEX_LOCK(&data_requests_list_mutex[src_node]);

			starpu_data_request_list_push_front(data_requests[src_node], r);

			PTHREAD_MUTEX_UNLOCK(&data_requests_list_mutex[src_node]);
		}

		/* wake the requesting worker up */
		// if we do not progress ..
		// pthread_cond_broadcast(&data_requests_list_cond[src_node]);
	}

	starpu_data_request_list_delete(local_list);
}

static void _handle_pending_node_data_requests(uint32_t src_node, unsigned force)
{
//	fprintf(stderr, "_starpu_handle_pending_node_data_requests ...\n");

	PTHREAD_MUTEX_LOCK(&data_requests_pending_list_mutex[src_node]);

	/* for all entries of the list */
	starpu_data_request_list_t local_list = data_requests_pending[src_node];
	data_requests_pending[src_node] = starpu_data_request_list_new();

	PTHREAD_MUTEX_UNLOCK(&data_requests_pending_list_mutex[src_node]);

	while (!starpu_data_request_list_empty(local_list))
	{
		starpu_data_request_t r;
		r = starpu_data_request_list_pop_back(local_list);

		starpu_data_handle handle = r->handle;
		
		_starpu_spin_lock(&handle->header_lock);
	
		_starpu_spin_lock(&r->lock);
	
		/* wait until the transfer is terminated */
		if (force)
		{
			_starpu_driver_wait_request_completion(&r->async_channel, src_node);
			starpu_handle_data_request_completion(r);
		}
		else {
			if (_starpu_driver_test_request_completion(&r->async_channel, src_node))
			{
				
				starpu_handle_data_request_completion(r);
			}
			else {
				_starpu_spin_unlock(&r->lock);
				_starpu_spin_unlock(&handle->header_lock);

				/* wake the requesting worker up */
				PTHREAD_MUTEX_LOCK(&data_requests_pending_list_mutex[src_node]);
				starpu_data_request_list_push_front(data_requests_pending[src_node], r);
				PTHREAD_MUTEX_UNLOCK(&data_requests_pending_list_mutex[src_node]);
			}
		}
	}

	starpu_data_request_list_delete(local_list);
}

void _starpu_handle_pending_node_data_requests(uint32_t src_node)
{
	_handle_pending_node_data_requests(src_node, 0);
}

void _starpu_handle_all_pending_node_data_requests(uint32_t src_node)
{
	_handle_pending_node_data_requests(src_node, 1);
}

int _starpu_check_that_no_data_request_exists(uint32_t node)
{
	/* XXX lock that !!! that's a quick'n'dirty test */
	int no_request = starpu_data_request_list_empty(data_requests[node]);
	int no_pending = starpu_data_request_list_empty(data_requests_pending[node]);

	return (no_request && no_pending);
}
