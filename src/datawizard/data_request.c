/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
 * Copyright (C) 2018,2021  Federal University of Rio Grande do Sul (UFRGS)
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
#include <common/config.h>
#include <common/utils.h>
#include <datawizard/datawizard.h>
#include <datawizard/memory_nodes.h>
#include <core/disk.h>
#include <core/simgrid.h>

void _starpu_init_data_request_lists(void)
{
	unsigned i, j;
	enum _starpu_data_request_inout k;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		struct _starpu_node *node = _starpu_get_node_struct(i);
		for (j = 0; j < STARPU_MAXNODES; j++)
		{
			for (k = _STARPU_DATA_REQUEST_IN; k <= _STARPU_DATA_REQUEST_OUT; k++)
			{
				_starpu_data_request_prio_list_init(&node->data_requests[j][k]);
				_starpu_data_request_prio_list_init(&node->prefetch_requests[j][k]);
				_starpu_data_request_prio_list_init(&node->idle_requests[j][k]);

#ifndef STARPU_DEBUG
				/* Tell helgrind that we are fine with checking for list_empty
				 * in _starpu_handle_node_data_requests, we will call it
				 * periodically anyway */
				STARPU_HG_DISABLE_CHECKING(node->data_requests[j][k].tree.root);
				STARPU_HG_DISABLE_CHECKING(node->prefetch_requests[j][k].tree.root);
				STARPU_HG_DISABLE_CHECKING(node->idle_requests[j][k].tree.root);
#endif
				_starpu_data_request_prio_list_init(&node->data_requests_pending[j][k]);
				node->data_requests_npending[j][k] = 0;

				STARPU_PTHREAD_MUTEX_INIT(&node->data_requests_list_mutex[j][k], NULL);
				STARPU_PTHREAD_MUTEX_INIT(&node->data_requests_pending_list_mutex[j][k], NULL);
			}
		}
		STARPU_HG_DISABLE_CHECKING(node->data_requests_npending);
	}
}

void _starpu_deinit_data_request_lists(void)
{
	unsigned i, j;
	enum _starpu_data_request_inout k;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		struct _starpu_node *node = _starpu_get_node_struct(i);
		for (j = 0; j < STARPU_MAXNODES; j++)
		{
			for (k = _STARPU_DATA_REQUEST_IN; k <= _STARPU_DATA_REQUEST_OUT; k++)
			{
				_starpu_data_request_prio_list_deinit(&node->data_requests[j][k]);
				_starpu_data_request_prio_list_deinit(&node->prefetch_requests[j][k]);
				_starpu_data_request_prio_list_deinit(&node->idle_requests[j][k]);
				_starpu_data_request_prio_list_deinit(&node->data_requests_pending[j][k]);
				STARPU_PTHREAD_MUTEX_DESTROY(&node->data_requests_pending_list_mutex[j][k]);
				STARPU_PTHREAD_MUTEX_DESTROY(&node->data_requests_list_mutex[j][k]);
			}
		}
	}
}

/* Unlink the request from the handle. New requests can then be made. */
/* this should be called with the lock r->handle->header_lock taken */
static void _starpu_data_request_unlink(struct _starpu_data_request *r)
{
	_starpu_spin_checklocked(&r->handle->header_lock);

	/* If this is a write invalidation request, we store it in the handle
	 */
	if (r->handle->write_invalidation_req == r)
	{
		STARPU_ASSERT(r->mode == STARPU_W);
		r->handle->write_invalidation_req = NULL;
	}
	else
	{
		unsigned node;
		struct _starpu_data_request **prevp, *prev;

		if (r->mode & STARPU_R)
			/* If this is a read request, we store the pending requests
			 * between src and dst. */
			node = r->src_replicate->memory_node;
		else
			/* If this is a write only request, then there is no source and
			 * we use the destination node to cache the request. */
			node = r->dst_replicate->memory_node;

		/* Look for ourself in the list, we should be not very far. */
		for (prevp = &r->dst_replicate->request[node], prev = NULL;
		     *prevp && *prevp != r;
		     prev = *prevp, prevp = &prev->next_same_req)
			;

		STARPU_ASSERT(*prevp == r);
		*prevp = r->next_same_req;

		if (!r->next_same_req)
		{
			/* I was last */
			STARPU_ASSERT(r->dst_replicate->last_request[node] == r);
			if (prev)
				r->dst_replicate->last_request[node] = prev;
			else
				r->dst_replicate->last_request[node] = NULL;
		}
	}
}

static void _starpu_data_request_destroy(struct _starpu_data_request *r)
{
	//fprintf(stderr, "DESTROY REQ %p (%d) refcnt %d\n", r, node, r->refcnt);
	_starpu_data_request_delete(r);
}

/* handle->lock should already be taken !  */
struct _starpu_data_request *_starpu_create_data_request(starpu_data_handle_t handle,
							 struct _starpu_data_replicate *src_replicate,
							 struct _starpu_data_replicate *dst_replicate,
							 int handling_node,
							 enum starpu_data_access_mode mode,
							 unsigned ndeps,
							 struct starpu_task *task,
							 enum starpu_is_prefetch is_prefetch,
							 int prio,
							 unsigned is_write_invalidation,
							 const char *origin)
{
	struct _starpu_data_request *r = _starpu_data_request_new();

	_starpu_spin_checklocked(&handle->header_lock);

	_starpu_spin_init(&r->lock);

	_STARPU_TRACE_DATA_REQUEST_CREATED(handle, src_replicate?src_replicate->memory_node:-1, dst_replicate?dst_replicate->memory_node:-1, prio, is_prefetch, r);

	r->origin = origin;
	r->handle = handle;
	r->src_replicate = src_replicate;
	r->dst_replicate = dst_replicate;
	r->mode = mode;
	r->async_channel.node_ops = NULL;
	r->async_channel.starpu_mp_common_finished_sender = 0;
	r->async_channel.starpu_mp_common_finished_receiver = 0;
	r->async_channel.polling_node_sender = NULL;
	r->async_channel.polling_node_receiver = NULL;
	memset(&r->async_channel.event, 0, sizeof(r->async_channel.event));
	if (handling_node == -1)
		handling_node = STARPU_MAIN_RAM;
	r->handling_node = handling_node;
	if (is_write_invalidation)
	{
		r->peer_node = handling_node;
		r->inout = _STARPU_DATA_REQUEST_IN;
	}
	else if (dst_replicate->memory_node == handling_node)
	{
		if (src_replicate)
			r->peer_node = src_replicate->memory_node;
		else
			r->peer_node = handling_node;
		r->inout = _STARPU_DATA_REQUEST_IN;
	}
	else
	{
		r->peer_node = dst_replicate->memory_node;
		r->inout = _STARPU_DATA_REQUEST_OUT;
	}
	STARPU_ASSERT(starpu_node_get_kind(handling_node) == STARPU_CPU_RAM || _starpu_memory_node_get_nworkers(handling_node));
	r->completed = 0;
	r->added_ref = 0;
	r->canceled = 0;
	r->prefetch = is_prefetch;
	r->task = task;
	r->nb_tasks_prefetch = 0;
	r->prio = prio;
	r->retval = -1;
	r->ndeps = ndeps;
	r->next_same_req = NULL;
	r->next_req_count = 0;
	r->callbacks = NULL;
	r->com_id = 0;

	_starpu_spin_lock(&r->lock);

	/* For a fetch, take a reference as soon as now on the target, to avoid
	 * replicate eviction */
	if (is_prefetch == STARPU_FETCH && dst_replicate)
	{
		r->added_ref = 1;
		dst_replicate->refcnt++;
	}
	handle->busy_count++;

	if (is_write_invalidation)
	{
		STARPU_ASSERT(!handle->write_invalidation_req);
		handle->write_invalidation_req = r;
	}
	else
	{
		unsigned node;

		if (mode & STARPU_R)
			node = src_replicate->memory_node;
		else
			node = dst_replicate->memory_node;

		if (!dst_replicate->request[node])
			dst_replicate->request[node] = r;
		else
			dst_replicate->last_request[node]->next_same_req = r;
		dst_replicate->last_request[node] = r;

		if (mode & STARPU_R)
		{
			/* Take a reference on the source for the request to be
			 * able to read it */
			src_replicate->refcnt++;
			handle->busy_count++;
		}
	}

	r->refcnt = 1;

	_starpu_spin_unlock(&r->lock);

	return r;
}

int _starpu_wait_data_request_completion(struct _starpu_data_request *r, enum _starpu_may_alloc may_alloc)
{
	int retval;
	int do_delete = 0;
	int completed;

#ifdef STARPU_SIMGRID
	unsigned local_node = starpu_worker_get_local_memory_node();

	starpu_pthread_wait_t wait;

	starpu_pthread_wait_init(&wait);
	/* We need to get woken both when requests finish on our node, and on
	 * the target node of the request we are waiting for */
	starpu_pthread_queue_register(&wait, &_starpu_simgrid_transfer_queue[local_node]);
	starpu_pthread_queue_register(&wait, &_starpu_simgrid_transfer_queue[(unsigned) r->dst_replicate->memory_node]);
#endif

	struct _starpu_worker *worker = _starpu_get_local_worker_key();
	enum _starpu_worker_status old_status = STATUS_UNKNOWN;

	if (worker)
	{
		old_status = worker->status;
		if (!(old_status & STATUS_WAITING))
			_starpu_add_worker_status(worker, STATUS_INDEX_WAITING, NULL);
	}

	do
	{
#ifdef STARPU_SIMGRID
		starpu_pthread_wait_reset(&wait);
#endif

		STARPU_SYNCHRONIZE();
		if (STARPU_RUNNING_ON_VALGRIND)
			completed = 1;
		else
			completed = r->completed;
		if (completed)
		{
			_starpu_spin_lock(&r->lock);
			if (r->completed)
				break;
			_starpu_spin_unlock(&r->lock);
		}

#ifndef STARPU_SIMGRID
#ifndef STARPU_NON_BLOCKING_DRIVERS
		/* XXX: shouldn't be needed, and doesn't work with chained requests anyway */
		_starpu_wake_all_blocked_workers_on_node(r->handling_node);
#endif
#endif

		_starpu_datawizard_progress(may_alloc);

#ifdef STARPU_SIMGRID
		starpu_pthread_wait_wait(&wait);
#endif
	}
	while (1);

	if (worker)
	{
		if (!(old_status & STATUS_WAITING))
			_starpu_clear_worker_status(worker, STATUS_INDEX_WAITING, NULL);
	}

#ifdef STARPU_SIMGRID
	starpu_pthread_queue_unregister(&wait, &_starpu_simgrid_transfer_queue[local_node]);
	starpu_pthread_queue_unregister(&wait, &_starpu_simgrid_transfer_queue[(unsigned) r->dst_replicate->memory_node]);
	starpu_pthread_wait_destroy(&wait);
#endif


	retval = r->retval;
	if (retval)
		_STARPU_DISP("REQUEST %p completed with retval %d!\n", r, r->retval);


	r->refcnt--;

	/* if nobody is waiting on that request, we can get rid of it */
	if (r->refcnt == 0)
		do_delete = 1;

	_starpu_spin_unlock(&r->lock);

	if (do_delete)
		_starpu_data_request_destroy(r);

	return retval;
}

/* this is non blocking */
void _starpu_post_data_request(struct _starpu_data_request *r)
{
	unsigned handling_node = r->handling_node;
	STARPU_ASSERT(starpu_node_get_kind(handling_node) == STARPU_CPU_RAM || _starpu_memory_node_get_nworkers(handling_node));

//	_STARPU_DEBUG("POST REQUEST\n");

	/* If some dependencies are not fulfilled yet, we don't actually post the request */
	if (r->ndeps > 0)
		return;

	struct _starpu_node *node_struct = _starpu_get_node_struct(handling_node);

	if (r->mode & STARPU_R)
	{
		STARPU_ASSERT(r->src_replicate->allocated || r->src_replicate->mapped != STARPU_UNMAPPED);
		STARPU_ASSERT(r->src_replicate->refcnt);
	}

	/* insert the request in the proper list */
	STARPU_PTHREAD_MUTEX_LOCK(&node_struct->data_requests_list_mutex[r->peer_node][r->inout]);
	if (r->prefetch >= STARPU_IDLEFETCH)
		_starpu_data_request_prio_list_push_back(&node_struct->idle_requests[r->peer_node][r->inout], r);
	else if (r->prefetch > STARPU_FETCH)
		_starpu_data_request_prio_list_push_back(&node_struct->prefetch_requests[r->peer_node][r->inout], r);
	else
		_starpu_data_request_prio_list_push_back(&node_struct->data_requests[r->peer_node][r->inout], r);
	STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->data_requests_list_mutex[r->peer_node][r->inout]);

#ifndef STARPU_NON_BLOCKING_DRIVERS
	_starpu_wake_all_blocked_workers_on_node(handling_node);
#endif
}

/* We assume that r->lock is taken by the caller */
void _starpu_data_request_append_callback(struct _starpu_data_request *r, void (*callback_func)(void *), void *callback_arg)
{
	STARPU_ASSERT(r);

	if (callback_func)
	{
		struct _starpu_callback_list *link;
		_STARPU_MALLOC(link, sizeof(struct _starpu_callback_list));

		link->callback_func = callback_func;
		link->callback_arg = callback_arg;
		link->next = r->callbacks;
		r->callbacks = link;
	}
}

/* This method is called with handle's header_lock taken, and unlocks it */
static void starpu_handle_data_request_completion(struct _starpu_data_request *r)
{
	unsigned do_delete = 0;
	starpu_data_handle_t handle = r->handle;
	enum starpu_data_access_mode mode = r->mode;

	struct _starpu_data_replicate *src_replicate = r->src_replicate;
	struct _starpu_data_replicate *dst_replicate = r->dst_replicate;


	if (r->canceled < 2 && dst_replicate)
	{
#ifdef STARPU_MEMORY_STATS
		enum _starpu_cache_state old_src_replicate_state = src_replicate->state;
#endif

		_starpu_spin_checklocked(&handle->header_lock);
		_starpu_update_data_state(handle, r->dst_replicate, mode);
		dst_replicate->load_request = NULL;

#ifdef STARPU_MEMORY_STATS
		if (src_replicate->state == STARPU_INVALID)
		{
			if (old_src_replicate_state == STARPU_OWNER)
				_starpu_memory_handle_stats_invalidated(handle, src_replicate->memory_node);
			else
			{
				/* XXX Currently only ex-OWNER are tagged as invalidated */
				/* XXX Have to check all old state of every node in case a SHARED data become OWNED by the dst_replicate */
			}

		}
		if (dst_replicate->state == STARPU_SHARED)
			_starpu_memory_handle_stats_loaded_shared(handle, dst_replicate->memory_node);
		else if (dst_replicate->state == STARPU_OWNER)
		{
			_starpu_memory_handle_stats_loaded_owner(handle, dst_replicate->memory_node);
		}
#endif
	}

#ifdef STARPU_USE_FXT
	if (fut_active && r->canceled < 2 && r->com_id > 0)
	{
		unsigned src_node = src_replicate->memory_node;
		unsigned dst_node = dst_replicate->memory_node;
		size_t size = _starpu_data_get_size(handle);
		_STARPU_TRACE_END_DRIVER_COPY(src_node, dst_node, size, r->com_id, r->prefetch);
	}
#endif

	/* Once the request has been fulfilled, we may submit the requests that
	 * were chained to that request. */
	unsigned chained_req;
	for (chained_req = 0; chained_req < r->next_req_count; chained_req++)
	{
		struct _starpu_data_request *next_req = r->next_req[chained_req];
		STARPU_ASSERT(next_req->ndeps > 0);
		next_req->ndeps--;
		_starpu_post_data_request(next_req);
	}

	r->completed = 1;

#ifdef STARPU_SIMGRID
	/* Wake potential worker which was waiting for it */
	if (dst_replicate)
		_starpu_wake_all_blocked_workers_on_node(dst_replicate->memory_node);
#endif

	/* Remove a reference on the destination replicate for the request */
	if (dst_replicate)
	{
		if (r->canceled < 2 && dst_replicate->mc)
			/* Make sure it stays there for the task.  */
			dst_replicate->nb_tasks_prefetch += r->nb_tasks_prefetch;

		if (r->added_ref)
		{
			STARPU_ASSERT(dst_replicate->refcnt > 0);
			dst_replicate->refcnt--;
		}
	}
	STARPU_ASSERT(handle->busy_count > 0);
	handle->busy_count--;

	/* In case the source was "locked" by the request too */
	if (mode & STARPU_R)
	{
		STARPU_ASSERT(src_replicate->refcnt > 0);
		src_replicate->refcnt--;
		STARPU_ASSERT(handle->busy_count > 0);
		handle->busy_count--;
	}
	_starpu_data_request_unlink(r);

	unsigned destroyed = _starpu_data_check_not_busy(handle);

	r->refcnt--;

	/* if nobody is waiting on that request, we can get rid of it */
	if (r->refcnt == 0)
		do_delete = 1;

	r->retval = 0;

	/* In case there are one or multiple callbacks, we execute them now. */
	struct _starpu_callback_list *callbacks = r->callbacks;

	_starpu_spin_unlock(&r->lock);

	if (do_delete)
		_starpu_data_request_destroy(r);

	if (!destroyed)
		_starpu_spin_unlock(&handle->header_lock);

	/* We do the callback once the lock is released so that they can do
	 * blocking operations with the handle (eg. release it) */
	while (callbacks)
	{
		callbacks->callback_func(callbacks->callback_arg);

		struct _starpu_callback_list *next = callbacks->next;
		free(callbacks);
		callbacks = next;
	}
}

void _starpu_data_request_complete_wait(void *arg)
{
	struct _starpu_data_request *r = arg;
	_starpu_spin_lock(&r->handle->header_lock);
	_starpu_spin_lock(&r->lock);
	starpu_handle_data_request_completion(r);
}

/* TODO : accounting to see how much time was spent working for other people ... */
static int starpu_handle_data_request(struct _starpu_data_request *r, enum _starpu_may_alloc may_alloc)
{
	starpu_data_handle_t handle = r->handle;

#ifndef STARPU_SIMGRID
	if (_starpu_spin_trylock(&handle->header_lock))
		return -EBUSY;
	if (_starpu_spin_trylock(&r->lock))
	{
		_starpu_spin_unlock(&handle->header_lock);
		return -EBUSY;
	}
#else
	/* Have to wait for the handle, whatever it takes, in simgrid,
	 * since we can not afford going to sleep, since nobody would wake us
	 * up. */
	_starpu_spin_lock(&handle->header_lock);
	_starpu_spin_lock(&r->lock);
#endif

	struct _starpu_data_replicate *src_replicate = r->src_replicate;
	struct _starpu_data_replicate *dst_replicate = r->dst_replicate;

	if (r->canceled)
	{
		/* Ok, canceled before starting copies etc. */
		r->canceled = 2;
		/* Nothing left to do */
		starpu_handle_data_request_completion(r);
		return 0;
	}

	if (dst_replicate)
	{
		struct _starpu_data_request *r2 = dst_replicate->load_request;
		if (r2 && r2 != r)
		{
			/* Oh, some other transfer is already loading the value. Just wait for it */
			r->canceled = 2;
			_starpu_spin_unlock(&r->lock);
			_starpu_spin_lock(&r2->lock);
			if (r->prefetch < r2->prefetch)
				/* Upgrade the existing request */
				_starpu_update_prefetch_status(r2, r->prefetch);
			_starpu_data_request_append_callback(r2, _starpu_data_request_complete_wait, r);
			_starpu_spin_unlock(&r2->lock);
			_starpu_spin_unlock(&handle->header_lock);
			return 0;
		}

		/* We are loading this replicate.
		 * Note: we might fail to allocate memory, but we will keep on and others will wait for us. */
		dst_replicate->load_request = r;
	}

	enum starpu_data_access_mode r_mode = r->mode;

	STARPU_ASSERT(!(r_mode & STARPU_R) || src_replicate);
	STARPU_ASSERT(!(r_mode & STARPU_R) || src_replicate->allocated || src_replicate->mapped != STARPU_UNMAPPED);
	STARPU_ASSERT(!(r_mode & STARPU_R) || src_replicate->refcnt);

	/* For prefetches, we take a reference on the destination only now that
	 * we will really try to fetch the data (instead of in
	 * _starpu_create_data_request) */
	if (dst_replicate && r->prefetch > STARPU_FETCH)
	{
		r->added_ref = 1;	/* Note: we might get upgraded while trying to allocate */
		dst_replicate->refcnt++;
	}

	_starpu_spin_unlock(&r->lock);

	if (r_mode == STARPU_UNMAP)
	{
		/* Unmap request, simply do it */
		STARPU_ASSERT(dst_replicate->mapped == src_replicate->memory_node);
		STARPU_ASSERT(handle->ops->unmap_data);
		handle->ops->unmap_data(src_replicate->data_interface, src_replicate->memory_node,
					dst_replicate->data_interface, dst_replicate->memory_node);
		dst_replicate->mapped = STARPU_UNMAPPED;
		r->retval = 0;
	}
	/* FIXME: the request may get upgraded from here to freeing it... */

	/* perform the transfer */
	/* the header of the data must be locked by the worker that submitted the request */


	if (dst_replicate && dst_replicate->state == STARPU_INVALID)
		r->retval = _starpu_driver_copy_data_1_to_1(handle, src_replicate,
						    dst_replicate, !(r_mode & STARPU_R), r, may_alloc, r->prefetch);
	else
		/* Already valid actually, no need to transfer anything */
		r->retval = 0;

	if (r->retval == -ENOMEM)
	{
		/* If there was not enough memory, we will try to redo the
		 * request later. */

		if (r->prefetch > STARPU_FETCH)
		{
			STARPU_ASSERT(r->added_ref);
			/* Drop ref until next try */
			r->added_ref = 0;
			dst_replicate->refcnt--;
		}

		_starpu_spin_unlock(&handle->header_lock);
		return -ENOMEM;
	}

	if (r->retval == -EAGAIN)
	{
		/* The request was successful, but could not be terminated
		 * immediately. We will handle the completion of the request
		 * asynchronously. The request is put in the list of "pending"
		 * requests in the meantime. */
		_starpu_spin_unlock(&handle->header_lock);
		struct _starpu_node *node_struct = _starpu_get_node_struct(r->handling_node);

		STARPU_PTHREAD_MUTEX_LOCK(&node_struct->data_requests_pending_list_mutex[r->peer_node][r->inout]);
		_starpu_data_request_prio_list_push_back(&node_struct->data_requests_pending[r->peer_node][r->inout], r);
		node_struct->data_requests_npending[r->peer_node][r->inout]++;
		STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->data_requests_pending_list_mutex[r->peer_node][r->inout]);

		return -EAGAIN;
	}

	/* the request has been handled */
	_starpu_spin_lock(&r->lock);
	starpu_handle_data_request_completion(r);

	return 0;
}

static int __starpu_handle_node_data_requests(struct _starpu_data_request_prio_list reqlist[STARPU_MAXNODES][2], unsigned handling_node, unsigned peer_node, enum _starpu_data_request_inout inout, enum _starpu_may_alloc may_alloc, unsigned n, unsigned *pushed, enum starpu_is_prefetch prefetch)
{
	struct _starpu_data_request *r;
	unsigned i;
	int ret = 0;

	*pushed = 0;

#ifdef STARPU_NON_BLOCKING_DRIVERS
	/* This is racy, but not posing problems actually, since we know we
	 * will come back here to probe again regularly anyway.
	 * Thus, do not expose this optimization to helgrind */
	if (!STARPU_RUNNING_ON_VALGRIND && _starpu_data_request_prio_list_empty(&reqlist[peer_node][inout]))
		return 0;
#endif

	struct _starpu_node *node_struct = _starpu_get_node_struct(handling_node);
	/* We create a new list to pickup some requests from the main list, and
	 * we handle the request(s) one by one from it, without concurrency issues.
	 */
	struct _starpu_data_request_list local_list, remain_list;
	_starpu_data_request_list_init(&local_list);

#ifdef STARPU_NON_BLOCKING_DRIVERS
	/* take all the entries from the request list */
	if (STARPU_PTHREAD_MUTEX_TRYLOCK(&node_struct->data_requests_list_mutex[peer_node][inout]))
	{
		/* List is busy, do not bother with it */
		return -EBUSY;
	}
#else
	STARPU_PTHREAD_MUTEX_LOCK(&node_struct->data_requests_list_mutex[peer_node][inout]);
#endif

	for (i = node_struct->data_requests_npending[peer_node][inout];
		i < n && ! _starpu_data_request_prio_list_empty(&reqlist[peer_node][inout]);
		i++)
	{
		r = _starpu_data_request_prio_list_pop_front_highest(&reqlist[peer_node][inout]);
		_starpu_data_request_list_push_back(&local_list, r);
	}

	if (!_starpu_data_request_prio_list_empty(&reqlist[peer_node][inout]))
		/* We have left some requests */
		ret = -EBUSY;

	STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->data_requests_list_mutex[peer_node][inout]);

	if (_starpu_data_request_list_empty(&local_list))
		/* there is no request */
		return 0;

	/* This will contain the remaining requests */
	_starpu_data_request_list_init(&remain_list);

	double start = starpu_timing_now();
	/* for all entries of the list */
	while (!_starpu_data_request_list_empty(&local_list))
	{
		int res;

		if (node_struct->data_requests_npending[peer_node][inout] >= n)
		{
			/* Too many requests at the same time, skip pushing
			 * more for now */
			ret = -EBUSY;
			break;
		}

		r = _starpu_data_request_list_pop_front(&local_list);

		res = starpu_handle_data_request(r, may_alloc);
		if (res != 0 && res != -EAGAIN)
		{
			/* handle is busy, or not enough memory, postpone for now */
			ret = res;
			/* Prefetch requests might have gotten promoted while in tmp list */
			_starpu_data_request_list_push_back(&remain_list, r);
			if (prefetch > STARPU_FETCH)
				/* Prefetching more there would make the situation even worse */
				break;
		}
		else
			(*pushed)++;

		if (starpu_timing_now() - start >= MAX_PUSH_TIME)
		{
			/* We have spent a lot of time doing requests, skip pushing more for now */
			ret = -EBUSY;
			break;
		}
	}

	/* Gather remainder */
	_starpu_data_request_list_push_list_back(&remain_list, &local_list);

	if (!_starpu_data_request_list_empty(&remain_list))
	{
		STARPU_PTHREAD_MUTEX_LOCK(&node_struct->data_requests_list_mutex[peer_node][inout]);
		while (!_starpu_data_request_list_empty(&remain_list))
		{
			r = _starpu_data_request_list_pop_back(&remain_list);
			if (r->prefetch >= STARPU_IDLEFETCH)
				_starpu_data_request_prio_list_push_front(&node_struct->idle_requests[r->peer_node][r->inout], r);
			else if (r->prefetch > STARPU_FETCH)
				_starpu_data_request_prio_list_push_front(&node_struct->prefetch_requests[r->peer_node][r->inout], r);
			else
				_starpu_data_request_prio_list_push_front(&node_struct->data_requests[r->peer_node][r->inout], r);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->data_requests_list_mutex[peer_node][inout]);

#ifdef STARPU_SIMGRID
		if (*pushed)
		{
			/* We couldn't process the request due to missing
			 * space. Advance the clock a bit to let eviction have
			 * the time to make some room for us. Ideally we should
			 * rather have the caller block, and explicitly wait
			 * for eviction to happen.
			 */
			starpu_sleep(0.000001);
			_starpu_wake_all_blocked_workers_on_node(handling_node);
		}
#elif !defined(STARPU_NON_BLOCKING_DRIVERS)
		_starpu_wake_all_blocked_workers_on_node(handling_node);
#endif
	}

	return ret;
}

int _starpu_handle_node_data_requests(unsigned handling_node, unsigned peer_node, enum _starpu_data_request_inout inout, enum _starpu_may_alloc may_alloc, unsigned *pushed)
{
	return __starpu_handle_node_data_requests(_starpu_get_node_struct(handling_node)->data_requests, handling_node, peer_node, inout, may_alloc, MAX_PENDING_REQUESTS_PER_NODE, pushed, STARPU_FETCH);
}

int _starpu_handle_node_prefetch_requests(unsigned handling_node, unsigned peer_node, enum _starpu_data_request_inout inout, enum _starpu_may_alloc may_alloc, unsigned *pushed)
{
	return __starpu_handle_node_data_requests(_starpu_get_node_struct(handling_node)->prefetch_requests, handling_node, peer_node, inout, may_alloc, MAX_PENDING_PREFETCH_REQUESTS_PER_NODE, pushed, STARPU_PREFETCH);
}

int _starpu_handle_node_idle_requests(unsigned handling_node, unsigned peer_node, enum _starpu_data_request_inout inout, enum _starpu_may_alloc may_alloc, unsigned *pushed)
{
	return __starpu_handle_node_data_requests(_starpu_get_node_struct(handling_node)->idle_requests, handling_node, peer_node, inout, may_alloc, MAX_PENDING_IDLE_REQUESTS_PER_NODE, pushed, STARPU_IDLEFETCH);
}

static int _handle_pending_node_data_requests(unsigned handling_node, unsigned peer_node, enum _starpu_data_request_inout inout, unsigned force)
{
//	_STARPU_DEBUG("_starpu_handle_pending_node_data_requests ...\n");
//
	struct _starpu_data_request_prio_list new_data_requests_pending;
	unsigned taken, kept;
	struct _starpu_node *node_struct = _starpu_get_node_struct(handling_node);

#ifdef STARPU_NON_BLOCKING_DRIVERS
	/* Here helgrind would should that this is an un protected access.
	 * We however don't care about missing an entry, we will get called
	 * again sooner or later. */
	if (!STARPU_RUNNING_ON_VALGRIND && _starpu_data_request_prio_list_empty(&node_struct->data_requests_pending[peer_node][inout]))
		return 0;
#endif

#ifdef STARPU_NON_BLOCKING_DRIVERS
	if (!force)
	{
		if (STARPU_PTHREAD_MUTEX_TRYLOCK(&node_struct->data_requests_pending_list_mutex[peer_node][inout]))
		{
			/* List is busy, do not bother with it */
			return 0;
		}
	}
	else
#endif
		/* We really want to handle requests */
		STARPU_PTHREAD_MUTEX_LOCK(&node_struct->data_requests_pending_list_mutex[peer_node][inout]);

	if (_starpu_data_request_prio_list_empty(&node_struct->data_requests_pending[peer_node][inout]))
	{
		/* there is no request */
		STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->data_requests_pending_list_mutex[peer_node][inout]);
		return 0;
	}
	/* for all entries of the list */
	struct _starpu_data_request_prio_list local_list = node_struct->data_requests_pending[peer_node][inout];
	_starpu_data_request_prio_list_init(&node_struct->data_requests_pending[peer_node][inout]);

	STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->data_requests_pending_list_mutex[peer_node][inout]);

	_starpu_data_request_prio_list_init(&new_data_requests_pending);
	taken = 0;
	kept = 0;

	while (!_starpu_data_request_prio_list_empty(&local_list))
	{
		struct _starpu_data_request *r;
		r = _starpu_data_request_prio_list_pop_front_highest(&local_list);
		taken++;

		starpu_data_handle_t handle = r->handle;

#ifndef STARPU_SIMGRID
		if (force)
			/* Have to wait for the handle, whatever it takes */
#endif
			/* Or when running in simgrid, in which case we can not
			 * afford going to sleep, since nobody would wake us
			 * up. */
			_starpu_spin_lock(&handle->header_lock);
#ifndef STARPU_SIMGRID
		else
			if (_starpu_spin_trylock(&handle->header_lock))
			{
				/* Handle is busy, retry this later */
				_starpu_data_request_prio_list_push_back(&new_data_requests_pending, r);
				kept++;
				continue;
			}
#endif

		/* This shouldn't be too hard to acquire */
		_starpu_spin_lock(&r->lock);

		/* wait until the transfer is terminated */
		if (force)
		{
			_starpu_driver_wait_request_completion(&r->async_channel);
			starpu_handle_data_request_completion(r);
		}
		else
		{
			if (_starpu_driver_test_request_completion(&r->async_channel))
			{
				/* The request was completed */
				starpu_handle_data_request_completion(r);
			}
			else
			{
				/* The request was not completed, so we put it
				 * back again on the list of pending requests
				 * so that it can be handled later on. */
				_starpu_spin_unlock(&r->lock);
				_starpu_spin_unlock(&handle->header_lock);

				_starpu_data_request_prio_list_push_back(&new_data_requests_pending, r);
				kept++;
			}
		}
	}
	_starpu_data_request_prio_list_deinit(&local_list);
	STARPU_PTHREAD_MUTEX_LOCK(&node_struct->data_requests_pending_list_mutex[peer_node][inout]);
	node_struct->data_requests_npending[peer_node][inout] -= taken - kept;
	if (kept)
		_starpu_data_request_prio_list_push_prio_list_back(&node_struct->data_requests_pending[peer_node][inout], &new_data_requests_pending);
	STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->data_requests_pending_list_mutex[peer_node][inout]);

	return taken - kept;
}

int _starpu_handle_pending_node_data_requests(unsigned handling_node, unsigned peer_node, enum _starpu_data_request_inout inout)
{
	return _handle_pending_node_data_requests(handling_node, peer_node, inout, 0);
}

int _starpu_handle_all_pending_node_data_requests(unsigned handling_node, unsigned peer_node, enum _starpu_data_request_inout inout)
{
	return _handle_pending_node_data_requests(handling_node, peer_node, inout, 1);
}

/* Note: the returned value will be outdated since the locks are not taken at
 * entry/exit */
static int __starpu_check_that_no_data_request_exists(unsigned node, unsigned peer_node, enum _starpu_data_request_inout inout)
{
	int no_request;
	int no_pending;
	struct _starpu_node *node_struct = _starpu_get_node_struct(node);

	STARPU_PTHREAD_MUTEX_LOCK(&node_struct->data_requests_list_mutex[peer_node][inout]);
	no_request = _starpu_data_request_prio_list_empty(&node_struct->data_requests[peer_node][inout])
	          && _starpu_data_request_prio_list_empty(&node_struct->prefetch_requests[peer_node][inout])
		  && _starpu_data_request_prio_list_empty(&node_struct->idle_requests[peer_node][inout]);
	STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->data_requests_list_mutex[peer_node][inout]);
	STARPU_PTHREAD_MUTEX_LOCK(&node_struct->data_requests_pending_list_mutex[peer_node][inout]);
	no_pending = !node_struct->data_requests_npending[peer_node][inout];
	STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->data_requests_pending_list_mutex[peer_node][inout]);

	return no_request && no_pending;
}

int _starpu_check_that_no_data_request_exists(unsigned node)
{
	unsigned peer_node, nnodes = starpu_memory_nodes_get_count();

	for (peer_node = 0; peer_node < nnodes; peer_node++)
		if (!__starpu_check_that_no_data_request_exists(node, peer_node, _STARPU_DATA_REQUEST_IN)
		 || !__starpu_check_that_no_data_request_exists(node, peer_node, _STARPU_DATA_REQUEST_OUT))
		 return 0;
	 return 1;
}

/* Note: the returned value will be outdated since the locks are not taken at
 * entry/exit */
int _starpu_check_that_no_data_request_is_pending(unsigned node, unsigned peer_node, enum _starpu_data_request_inout inout)
{
	return !_starpu_get_node_struct(node)->data_requests_npending[peer_node][inout];
}


void _starpu_update_prefetch_status(struct _starpu_data_request *r, enum starpu_is_prefetch prefetch)
{
	struct _starpu_node *node_struct = _starpu_get_node_struct(r->handling_node);
	_starpu_spin_checklocked(&r->handle->header_lock);
	STARPU_ASSERT(r->prefetch > prefetch);

	if (prefetch == STARPU_FETCH && !r->added_ref)
	{
		/* That would have been done by _starpu_create_data_request */
		r->added_ref = 1;
		r->dst_replicate->refcnt++;
	}

	r->prefetch=prefetch;

	if (prefetch >= STARPU_IDLEFETCH)
		/* No possible actual change */
		return;

	/* We have to promote chained_request too! */
	unsigned chained_req;
	for (chained_req = 0; chained_req < r->next_req_count; chained_req++)
	{
		struct _starpu_data_request *next_req = r->next_req[chained_req];
		if (next_req->prefetch > prefetch)
			_starpu_update_prefetch_status(next_req, prefetch);
	}

	STARPU_PTHREAD_MUTEX_LOCK(&node_struct->data_requests_list_mutex[r->peer_node][r->inout]);

	int found = 1;

	/* The request can be in a different list (handling request or the temp list)
	 * we have to check that it is really in the prefetch or idle list. */
	if (_starpu_data_request_prio_list_ismember(&node_struct->prefetch_requests[r->peer_node][r->inout], r))
		_starpu_data_request_prio_list_erase(&node_struct->prefetch_requests[r->peer_node][r->inout], r);
	else if (_starpu_data_request_prio_list_ismember(&node_struct->idle_requests[r->peer_node][r->inout], r))
		_starpu_data_request_prio_list_erase(&node_struct->idle_requests[r->peer_node][r->inout], r);
	else
		found = 0;

	if (found)
	{
		if (prefetch > STARPU_FETCH)
			_starpu_data_request_prio_list_push_back(&node_struct->prefetch_requests[r->peer_node][r->inout],r);
		else
			_starpu_data_request_prio_list_push_back(&node_struct->data_requests[r->peer_node][r->inout],r);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->data_requests_list_mutex[r->peer_node][r->inout]);

#ifndef STARPU_NON_BLOCKING_DRIVERS
	_starpu_wake_all_blocked_workers_on_node(r->handling_node);
#endif
}
