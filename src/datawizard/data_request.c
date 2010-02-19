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
#include <datawizard/data_request.h>
#include <pthread.h>

/* requests that have not been treated at all */
static starpu_data_request_list_t data_requests[STARPU_MAXNODES];
static pthread_cond_t data_requests_list_cond[STARPU_MAXNODES];
static pthread_mutex_t data_requests_list_mutex[STARPU_MAXNODES];

/* requests that are not terminated (eg. async transfers) */
static starpu_data_request_list_t data_requests_pending[STARPU_MAXNODES];
static pthread_cond_t data_requests_pending_list_cond[STARPU_MAXNODES];
static pthread_mutex_t data_requests_pending_list_mutex[STARPU_MAXNODES];

void starpu_init_data_request_lists(void)
{
	unsigned i;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		data_requests[i] = starpu_data_request_list_new();
		pthread_mutex_init(&data_requests_list_mutex[i], NULL);
		pthread_cond_init(&data_requests_list_cond[i], NULL);

		data_requests_pending[i] = starpu_data_request_list_new();
		pthread_mutex_init(&data_requests_pending_list_mutex[i], NULL);
		pthread_cond_init(&data_requests_pending_list_cond[i], NULL);
	}
}

void starpu_deinit_data_request_lists(void)
{
	unsigned i;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		pthread_cond_destroy(&data_requests_pending_list_cond[i]);
		pthread_mutex_destroy(&data_requests_pending_list_mutex[i]);
		starpu_data_request_list_delete(data_requests_pending[i]);

		pthread_cond_destroy(&data_requests_list_cond[i]);
		pthread_mutex_destroy(&data_requests_list_mutex[i]);
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
starpu_data_request_t starpu_create_data_request(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node, uint32_t handling_node, uint8_t read, uint8_t write, unsigned is_prefetch)
{
	starpu_data_request_t r = starpu_data_request_new();

	starpu_spin_init(&r->lock);

	r->handle = handle;
	r->src_node = src_node;
	r->dst_node = dst_node;
	r->read = read;
	r->write = write;

	r->handling_node = handling_node;

	r->completed = 0;
	r->retval = -1;

	r->next_req_count = 0;

	r->strictness = 1;
	r->is_a_prefetch_request = is_prefetch;

	/* associate that request with the handle so that further similar
	 * requests will reuse that one  */

	starpu_spin_lock(&r->lock);

	handle->per_node[dst_node].request = r;

	handle->per_node[dst_node].refcnt++;

	if (read)
		handle->per_node[src_node].refcnt++;

	r->refcnt = 1;

	starpu_spin_unlock(&r->lock);

	return r;
}

/* handle->lock should be taken */
starpu_data_request_t starpu_search_existing_data_request(starpu_data_handle handle, uint32_t dst_node, uint8_t read, uint8_t write)
{
	starpu_data_request_t r = handle->per_node[dst_node].request;

	if (r)
	{
		/* perhaps we need to "upgrade" the request */
		if (read)
		{
			/* in case the exisiting request did not imply a memory
			 * transfer yet, we have to increment the refcnt now
			 * (so that the source remains valid) */
			if (!r->read)
				handle->per_node[dst_node].refcnt++;

			r->read = 1;
		}

		if (write)
			r->write = 1;

		starpu_spin_lock(&r->lock);
	}

	return r;
}

int starpu_wait_data_request_completion(starpu_data_request_t r, unsigned may_alloc)
{
	int retval;
	int do_delete = 0;

	uint32_t local_node = get_local_memory_node();

	do {
		starpu_spin_lock(&r->lock);

		if (r->completed)
			break;

		starpu_spin_unlock(&r->lock);

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

	starpu_spin_unlock(&r->lock);
	
	if (do_delete)
		starpu_data_request_destroy(r);
	
	return retval;
}

/* this is non blocking */
void starpu_post_data_request(starpu_data_request_t r, uint32_t handling_node)
{
	int res;
//	fprintf(stderr, "POST REQUEST\n");

	if (r->read)
	{
		STARPU_ASSERT(r->handle->per_node[r->src_node].allocated);
		STARPU_ASSERT(r->handle->per_node[r->src_node].refcnt);
	}

	/* insert the request in the proper list */
	res = pthread_mutex_lock(&data_requests_list_mutex[handling_node]);
	STARPU_ASSERT(!res);

	starpu_data_request_list_push_front(data_requests[handling_node], r);

	res = pthread_mutex_unlock(&data_requests_list_mutex[handling_node]);
	STARPU_ASSERT(!res);

	_starpu_wake_all_blocked_workers_on_node(handling_node);
}

static void starpu_handle_data_request_completion(starpu_data_request_t r)
{
	unsigned do_delete = 0;
	starpu_data_handle handle = r->handle;

	uint32_t src_node = r->src_node;
	uint32_t dst_node = r->dst_node;

	starpu_update_data_state(handle, dst_node, r->write);

#ifdef STARPU_USE_FXT
	size_t size = handle->ops->get_size(handle);
	TRACE_END_DRIVER_COPY(src_node, dst_node, size, r->com_id);
#endif

	unsigned chained_req;
	for (chained_req = 0; chained_req < r->next_req_count; chained_req++)
	{
		starpu_post_data_request(r->next_req[chained_req], r->next_req[chained_req]->handling_node);
	}

	r->completed = 1;
	
	handle->per_node[dst_node].refcnt--;

	if (r->read)
		handle->per_node[src_node].refcnt--;

	r->refcnt--;

	/* if nobody is waiting on that request, we can get rid of it */
	if (r->refcnt == 0)
		do_delete = 1;
	
	r->retval = 0;

	starpu_spin_unlock(&r->lock);

	if (do_delete)
		starpu_data_request_destroy(r);

	starpu_spin_unlock(&handle->header_lock);
}

/* TODO : accounting to see how much time was spent working for other people ... */
static int starpu_handle_data_request(starpu_data_request_t r, unsigned may_alloc)
{
	starpu_data_handle handle = r->handle;

	starpu_spin_lock(&handle->header_lock);

	starpu_spin_lock(&r->lock);

	if (r->read)
	{
		STARPU_ASSERT(handle->per_node[r->src_node].allocated);
		STARPU_ASSERT(handle->per_node[r->src_node].refcnt);
	}

	/* perform the transfer */
	/* the header of the data must be locked by the worker that submitted the request */
	r->retval = starpu_driver_copy_data_1_to_1(handle, r->src_node, r->dst_node, !r->read, r, may_alloc);

	if (r->retval == ENOMEM)
	{
		starpu_spin_unlock(&r->lock);
		starpu_spin_unlock(&handle->header_lock);

		return ENOMEM;
	}

	if (r->retval == EAGAIN)
	{
		starpu_spin_unlock(&r->lock);
		starpu_spin_unlock(&handle->header_lock);

		/* the request is pending and we put it in the corresponding queue  */
		pthread_mutex_lock(&data_requests_pending_list_mutex[r->handling_node]);
		starpu_data_request_list_push_front(data_requests_pending[r->handling_node], r);
		pthread_mutex_unlock(&data_requests_pending_list_mutex[r->handling_node]);

		return EAGAIN;
	}

	/* the request has been handled */
	starpu_handle_data_request_completion(r);

	return 0;
}

void starpu_handle_node_data_requests(uint32_t src_node, unsigned may_alloc)
{
	int res;

	/* for all entries of the list */
	starpu_data_request_t r;

	/* take all the entries from the request list */
	res = pthread_mutex_lock(&data_requests_list_mutex[src_node]);
	STARPU_ASSERT(!res);

	starpu_data_request_list_t local_list = data_requests[src_node];

	if (starpu_data_request_list_empty(local_list))
	{
		/* there is no request */
		res = pthread_mutex_unlock(&data_requests_list_mutex[src_node]);
		STARPU_ASSERT(!res);

		return;
	}

	data_requests[src_node] = starpu_data_request_list_new();

	res = pthread_mutex_unlock(&data_requests_list_mutex[src_node]);
	STARPU_ASSERT(!res);

	while (!starpu_data_request_list_empty(local_list))
	{
		r = starpu_data_request_list_pop_back(local_list);

		res = starpu_handle_data_request(r, may_alloc);
		if (res == ENOMEM)
		{
			res = pthread_mutex_lock(&data_requests_list_mutex[src_node]);
			STARPU_ASSERT(!res);

			starpu_data_request_list_push_front(data_requests[src_node], r);

			res = pthread_mutex_unlock(&data_requests_list_mutex[src_node]);
			STARPU_ASSERT(!res);
		}

		/* wake the requesting worker up */
		// if we do not progress ..
		// pthread_cond_broadcast(&data_requests_list_cond[src_node]);
	}

	starpu_data_request_list_delete(local_list);
}

static void _handle_pending_node_data_requests(uint32_t src_node, unsigned force)
{
	int res;
//	fprintf(stderr, "starpu_handle_pending_node_data_requests ...\n");

	res = pthread_mutex_lock(&data_requests_pending_list_mutex[src_node]);
	STARPU_ASSERT(!res);

	/* for all entries of the list */
	starpu_data_request_list_t local_list = data_requests_pending[src_node];
	data_requests_pending[src_node] = starpu_data_request_list_new();

	res = pthread_mutex_unlock(&data_requests_pending_list_mutex[src_node]);
	STARPU_ASSERT(!res);

	while (!starpu_data_request_list_empty(local_list))
	{
		starpu_data_request_t r;
		r = starpu_data_request_list_pop_back(local_list);

		starpu_data_handle handle = r->handle;
		
		starpu_spin_lock(&handle->header_lock);
	
		starpu_spin_lock(&r->lock);
	
		/* wait until the transfer is terminated */
		if (force)
		{
			starpu_driver_wait_request_completion(&r->async_channel, src_node);
			starpu_handle_data_request_completion(r);
		}
		else {
			if (starpu_driver_test_request_completion(&r->async_channel, src_node))
			{
				
				starpu_handle_data_request_completion(r);
			}
			else {
				starpu_spin_unlock(&r->lock);
				starpu_spin_unlock(&handle->header_lock);

				/* wake the requesting worker up */
				pthread_mutex_lock(&data_requests_pending_list_mutex[src_node]);
				starpu_data_request_list_push_front(data_requests_pending[src_node], r);
				pthread_mutex_unlock(&data_requests_pending_list_mutex[src_node]);
			}
		}
	}

	starpu_data_request_list_delete(local_list);
}

void starpu_handle_pending_node_data_requests(uint32_t src_node)
{
	_handle_pending_node_data_requests(src_node, 0);
}

void starpu_handle_all_pending_node_data_requests(uint32_t src_node)
{
	_handle_pending_node_data_requests(src_node, 1);
}

int starpu_check_that_no_data_request_exists(uint32_t node)
{
	/* XXX lock that !!! that's a quick'n'dirty test */
	int no_request = starpu_data_request_list_empty(data_requests[node]);
	int no_pending = starpu_data_request_list_empty(data_requests_pending[node]);

	return (no_request && no_pending);
}
