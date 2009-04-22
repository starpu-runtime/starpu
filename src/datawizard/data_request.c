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

#include <datawizard/data_request.h>
#include <pthread.h>

static data_request_list_t data_requests[MAXNODES];
static pthread_cond_t data_requests_list_cond[MAXNODES];
static pthread_mutex_t data_requests_list_mutex[MAXNODES];

void init_data_request_lists(void)
{
	unsigned i;
	for (i = 0; i < MAXNODES; i++)
	{
		data_requests[i] = data_request_list_new();
		pthread_mutex_init(&data_requests_list_mutex[i], NULL);
		pthread_cond_init(&data_requests_list_cond[i], NULL);
	}
}

void deinit_data_request_lists(void)
{
	unsigned i;
	for (i = 0; i < MAXNODES; i++)
	{
		pthread_cond_destroy(&data_requests_list_cond[i]);
		pthread_mutex_destroy(&data_requests_list_mutex[i]);
		data_request_list_delete(data_requests[i]);
	}
}

int post_data_request(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	int retvalue;

	data_request_t r = data_request_new();

	r->state = state;
	r->src_node = src_node;
	r->dst_node = dst_node;

	r->completed = 0;

	/* insert the request in the proper list */
	pthread_mutex_lock(&data_requests_list_mutex[src_node]);
	data_request_list_push_front(data_requests[src_node], r);
	pthread_mutex_unlock(&data_requests_list_mutex[src_node]);

	/* wake the threads that could perform that operation */
	wake_all_blocked_workers_on_node(src_node);

#ifdef NO_DATA_RW_LOCK
	/* XXX: since there is no concurrency on this data (we don't use the
	 * rw-lock) we can assume that the data on the source node should not
	 * be invalidated.
	 * TODO: handle the situation of a possible invalidation caused by
	 * memory eviction mechanism. This could be done by the means of a
	 * specific state (or flag) in the MSI protocol. */
	pthread_spin_unlock(&state->header_lock);
#endif

//	/* wait for the request to be performed */
//	while(sem_trywait(&r->sem) == -1)
//	{
//		wake_all_blocked_workers_on_node(src_node);
//		datawizard_progress(dst_node);
//	}

	/* wait for the request to be performed */
	pthread_mutex_lock(&data_requests_list_mutex[src_node]);
	while(!r->completed)
	{
		// if we do not progress
		// pthread_cond_wait(&data_requests_list_cond[src_node], &data_requests_list_mutex[src_node]);
		pthread_mutex_unlock(&data_requests_list_mutex[src_node]);

		wake_all_blocked_workers_on_node(src_node);
		datawizard_progress(dst_node);

		pthread_mutex_lock(&data_requests_list_mutex[src_node]);
	}
	pthread_mutex_unlock(&data_requests_list_mutex[src_node]);

#ifdef NO_DATA_RW_LOCK
	pthread_spin_lock(&state->header_lock);
#endif

	retvalue = r->retval;
	
	/* the request is useless now */
	data_request_delete(r);

	return retvalue;	
}

void handle_node_data_requests(uint32_t src_node)
{
	pthread_mutex_lock(&data_requests_list_mutex[src_node]);

	/* for all entries of the list */
	data_request_list_t l = data_requests[src_node];
	data_request_t r;

	while (!data_request_list_empty(l))
	{
		r = data_request_list_pop_back(l);		
		pthread_mutex_unlock(&data_requests_list_mutex[src_node]);

		/* TODO : accounting to see how much time was spent working for other people ... */

		/* perform the transfer */
		/* the header of the data must be locked by the worker that submitted the request */
		r->retval = driver_copy_data_1_to_1(r->state, r->src_node, r->dst_node, 0);
		
		/* wake the requesting worker up */
		pthread_mutex_lock(&data_requests_list_mutex[src_node]);
		r->completed = 1;

		// if we do not progress ..
		// pthread_cond_broadcast(&data_requests_list_cond[src_node]);

	}

	pthread_mutex_unlock(&data_requests_list_mutex[src_node]);
}
