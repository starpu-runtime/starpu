/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
 * Copyright (C) 2018       Federal University of Rio Grande do Sul (UFRGS)
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

/* requests that have not been treated at all */
static struct _starpu_data_request_prio_list data_requests[STARPU_MAXNODES];
static struct _starpu_data_request_prio_list prefetch_requests[STARPU_MAXNODES];
static struct _starpu_data_request_prio_list idle_requests[STARPU_MAXNODES];
static starpu_pthread_mutex_t data_requests_list_mutex[STARPU_MAXNODES];

/* requests that are not terminated (eg. async transfers) */
static struct _starpu_data_request_prio_list data_requests_pending[STARPU_MAXNODES];
static unsigned data_requests_npending[STARPU_MAXNODES];
static starpu_pthread_mutex_t data_requests_pending_list_mutex[STARPU_MAXNODES];

void _starpu_init_data_request_lists(void)
{
	unsigned i;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		_starpu_data_request_prio_list_init(&data_requests[i]);
		_starpu_data_request_prio_list_init(&prefetch_requests[i]);
		_starpu_data_request_prio_list_init(&idle_requests[i]);

#ifndef STARPU_DEBUG
		/* Tell helgrind that we are fine with checking for list_empty
		 * in _starpu_handle_node_data_requests, we will call it
		 * periodically anyway */
		STARPU_HG_DISABLE_CHECKING(data_requests[i].tree.root);
		STARPU_HG_DISABLE_CHECKING(prefetch_requests[i].tree.root);
		STARPU_HG_DISABLE_CHECKING(idle_requests[i].tree.root);
#endif

		STARPU_PTHREAD_MUTEX_INIT(&data_requests_list_mutex[i], NULL);

		_starpu_data_request_prio_list_init(&data_requests_pending[i]);
		data_requests_npending[i] = 0;
		STARPU_PTHREAD_MUTEX_INIT(&data_requests_pending_list_mutex[i], NULL);
	}
	STARPU_HG_DISABLE_CHECKING(data_requests_npending);
}

void _starpu_deinit_data_request_lists(void)
{
	unsigned i;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		_starpu_data_request_prio_list_deinit(&data_requests[i]);
		_starpu_data_request_prio_list_deinit(&prefetch_requests[i]);
		_starpu_data_request_prio_list_deinit(&idle_requests[i]);
		STARPU_PTHREAD_MUTEX_DESTROY(&data_requests_pending_list_mutex[i]);
		_starpu_data_request_prio_list_deinit(&data_requests_pending[i]);
		STARPU_PTHREAD_MUTEX_DESTROY(&data_requests_list_mutex[i]);
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
	else if (r->mode & STARPU_R)
	{
		/* If this is a read request, we store the pending requests
		 * between src and dst. */
		unsigned node = r->src_replicate->memory_node;
		STARPU_ASSERT(r->dst_replicate->request[node] == r);
		r->dst_replicate->request[node] = NULL;
	}
	else
	{
		/* If this is a write only request, then there is no source and
		 * we use the destination node to cache the request. */
		unsigned node = r->dst_replicate->memory_node;
		STARPU_ASSERT(r->dst_replicate->request[node] == r);
		r->dst_replicate->request[node] = NULL;
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
							 enum _starpu_is_prefetch is_prefetch,
							 int prio,
							 unsigned is_write_invalidation,
							 const char *origin)
{
	struct _starpu_data_request *r = _starpu_data_request_new();

	_starpu_spin_checklocked(&handle->header_lock);

	_starpu_spin_init(&r->lock);

	_STARPU_TRACE_DATA_REQUEST_CREATED(handle, src_replicate?src_replicate->memory_node:-1, dst_replicate?dst_replicate->memory_node:-1, prio, is_prefetch);

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
#ifdef STARPU_USE_MPI_MASTER_SLAVE
        r->async_channel.event.mpi_ms_event.requests = NULL;
#endif
	if (handling_node == -1)
		handling_node = STARPU_MAIN_RAM;
	r->handling_node = handling_node;
	STARPU_ASSERT(starpu_node_get_kind(handling_node) == STARPU_CPU_RAM || _starpu_memory_node_get_nworkers(handling_node));
	r->completed = 0;
	r->prefetch = is_prefetch;
	r->prio = prio;
	r->retval = -1;
	r->ndeps = ndeps;
	r->next_req_count = 0;
	r->callbacks = NULL;
	r->com_id = 0;

	_starpu_spin_lock(&r->lock);

	/* Take a reference on the target for the request to be able to write it */
	if (dst_replicate)
		dst_replicate->refcnt++;
	handle->busy_count++;

	if (is_write_invalidation)
	{
		STARPU_ASSERT(!handle->write_invalidation_req);
		handle->write_invalidation_req = r;
	}
	else if (mode & STARPU_R)
	{
		unsigned src_node = src_replicate->memory_node;
		STARPU_ASSERT(!dst_replicate->request[src_node]);
		dst_replicate->request[src_node] = r;
		/* Take a reference on the source for the request to be able to read it */
		src_replicate->refcnt++;
		handle->busy_count++;
	}
	else
	{
		unsigned dst_node = dst_replicate->memory_node;
		STARPU_ASSERT(!dst_replicate->request[dst_node]);
		dst_replicate->request[dst_node] = r;
	}

	r->refcnt = 1;

	_starpu_spin_unlock(&r->lock);

	return r;
}

int _starpu_wait_data_request_completion(struct _starpu_data_request *r, unsigned may_alloc)
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
		old_status = worker->status ;
		_starpu_set_worker_status(worker, STATUS_WAITING);
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
		_starpu_set_worker_status(worker, old_status);
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

	if (r->mode & STARPU_R)
	{
		STARPU_ASSERT(r->src_replicate->allocated);
		STARPU_ASSERT(r->src_replicate->refcnt);
	}

	/* insert the request in the proper list */
	STARPU_PTHREAD_MUTEX_LOCK(&data_requests_list_mutex[handling_node]);
	if (r->prefetch == STARPU_IDLEFETCH)
		_starpu_data_request_prio_list_push_back(&idle_requests[handling_node], r);
	else if (r->prefetch > STARPU_FETCH)
		_starpu_data_request_prio_list_push_back(&prefetch_requests[handling_node], r);
	else
		_starpu_data_request_prio_list_push_back(&data_requests[handling_node], r);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data_requests_list_mutex[handling_node]);

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


	if (dst_replicate)
	{
#ifdef STARPU_MEMORY_STATS
		enum _starpu_cache_state old_src_replicate_state = src_replicate->state;
#endif

		_starpu_spin_checklocked(&handle->header_lock);
		_starpu_update_data_state(handle, r->dst_replicate, mode);

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

	if (r->com_id > 0)
	{
#ifdef STARPU_USE_FXT
		unsigned src_node = src_replicate->memory_node;
		unsigned dst_node = dst_replicate->memory_node;
		size_t size = _starpu_data_get_size(handle);
		_STARPU_TRACE_END_DRIVER_COPY(src_node, dst_node, size, r->com_id, r->prefetch);
#endif
	}

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
		STARPU_ASSERT(dst_replicate->refcnt > 0);
		dst_replicate->refcnt--;
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

/* TODO : accounting to see how much time was spent working for other people ... */
static int starpu_handle_data_request(struct _starpu_data_request *r, unsigned may_alloc, enum _starpu_is_prefetch prefetch)
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

	enum starpu_data_access_mode r_mode = r->mode;

	STARPU_ASSERT(!(r_mode & STARPU_R) || src_replicate);
	STARPU_ASSERT(!(r_mode & STARPU_R) || src_replicate->allocated);
	STARPU_ASSERT(!(r_mode & STARPU_R) || src_replicate->refcnt);

	_starpu_spin_unlock(&r->lock);

	/* FIXME: the request may get upgraded from here to freeing it... */

	/* perform the transfer */
	/* the header of the data must be locked by the worker that submitted the request */


	if (dst_replicate && dst_replicate->state == STARPU_INVALID)
		r->retval = _starpu_driver_copy_data_1_to_1(handle, src_replicate,
						    dst_replicate, !(r_mode & STARPU_R), r, may_alloc, prefetch);
	else
		/* Already valid actually, no need to transfer anything */
		r->retval = 0;

	if (r->retval == -ENOMEM)
	{
		/* If there was not enough memory, we will try to redo the
		 * request later. */
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

		STARPU_PTHREAD_MUTEX_LOCK(&data_requests_pending_list_mutex[r->handling_node]);
		_starpu_data_request_prio_list_push_back(&data_requests_pending[r->handling_node], r);
		data_requests_npending[r->handling_node]++;
		STARPU_PTHREAD_MUTEX_UNLOCK(&data_requests_pending_list_mutex[r->handling_node]);

		return -EAGAIN;
	}

	/* the request has been handled */
	_starpu_spin_lock(&r->lock);
	starpu_handle_data_request_completion(r);

	return 0;
}

static int __starpu_handle_node_data_requests(struct _starpu_data_request_prio_list *reqlist, unsigned src_node, unsigned may_alloc, unsigned n, unsigned *pushed, enum _starpu_is_prefetch prefetch)
{
	struct _starpu_data_request *r;
	struct _starpu_data_request_prio_list new_data_requests[prefetch + 1]; /* Indexed by prefetch level */
	unsigned i;
	int ret = 0;

	*pushed = 0;

#ifdef STARPU_NON_BLOCKING_DRIVERS
	/* This is racy, but not posing problems actually, since we know we
	 * will come back here to probe again regularly anyway.
	 * Thus, do not expose this optimization to helgrind */
	if (!STARPU_RUNNING_ON_VALGRIND && _starpu_data_request_prio_list_empty(&reqlist[src_node]))
		return 0;
#endif

	/* TODO optimize */

#ifdef STARPU_NON_BLOCKING_DRIVERS
	/* take all the entries from the request list */
	if (STARPU_PTHREAD_MUTEX_TRYLOCK(&data_requests_list_mutex[src_node]))
	{
		/* List is busy, do not bother with it */
		return -EBUSY;
	}
#else
	STARPU_PTHREAD_MUTEX_LOCK(&data_requests_list_mutex[src_node]);
#endif

	if (_starpu_data_request_prio_list_empty(&reqlist[src_node]))
	{
		/* there is no request */
                STARPU_PTHREAD_MUTEX_UNLOCK(&data_requests_list_mutex[src_node]);
		return 0;
	}

	/* There is an entry: we create a new empty list to replace the list of
	 * requests, and we handle the request(s) one by one in the former
	 * list, without concurrency issues.*/
	struct _starpu_data_request_prio_list local_list = reqlist[src_node];
	_starpu_data_request_prio_list_init(&reqlist[src_node]);

	STARPU_PTHREAD_MUTEX_UNLOCK(&data_requests_list_mutex[src_node]);

	for (i = 0; i <= prefetch; i++)
		_starpu_data_request_prio_list_init(&new_data_requests[i]);

	double start = starpu_timing_now();
	/* for all entries of the list */
	while (!_starpu_data_request_prio_list_empty(&local_list))
	{
                int res;

		if (data_requests_npending[src_node] >= n)
		{
			/* Too many requests at the same time, skip pushing
			 * more for now */
			ret = -EBUSY;
			break;
		}

		r = _starpu_data_request_prio_list_pop_front_highest(&local_list);

		res = starpu_handle_data_request(r, may_alloc, prefetch);
		if (res != 0 && res != -EAGAIN)
		{
			/* handle is busy, or not enough memory, postpone for now */
			ret = res;
			/* Prefetch requests might have gotten promoted while in tmp list */
			_starpu_data_request_prio_list_push_back(&new_data_requests[r->prefetch], r);
			if (prefetch > STARPU_FETCH)
				/* Prefetching more there would make the situation even worse */
				break;
		}

		(*pushed)++;
		if (starpu_timing_now() - start >= MAX_PUSH_TIME)
		{
			/* We have spent a lot of time doing requests, skip pushing more for now */
			ret = -EBUSY;
			break;
		}
	}

	/* Push back requests we didn't handle on the proper list */
	while (!_starpu_data_request_prio_list_empty(&local_list))
	{
		r = _starpu_data_request_prio_list_pop_front_highest(&local_list);
		/* Prefetch requests might have gotten promoted while in tmp list */
		_starpu_data_request_prio_list_push_back(&new_data_requests[r->prefetch], r);
	}
	_starpu_data_request_prio_list_deinit(&local_list);

	for (i = 0; i <= prefetch; i++)
		if (!_starpu_data_request_prio_list_empty(&new_data_requests[i]))
			break;

	if (i <= prefetch)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&data_requests_list_mutex[src_node]);
		if (!(_starpu_data_request_prio_list_empty(&new_data_requests[STARPU_FETCH])))
		{
			_starpu_data_request_prio_list_push_prio_list_back(&new_data_requests[STARPU_FETCH], &data_requests[src_node]);
			data_requests[src_node] = new_data_requests[STARPU_FETCH];
		}
		if (prefetch >= STARPU_PREFETCH && !(_starpu_data_request_prio_list_empty(&new_data_requests[STARPU_PREFETCH])))
		{
			_starpu_data_request_prio_list_push_prio_list_back(&new_data_requests[STARPU_PREFETCH], &prefetch_requests[src_node]);
			prefetch_requests[src_node] = new_data_requests[STARPU_PREFETCH];
		}
		if (prefetch >= STARPU_IDLEFETCH && !(_starpu_data_request_prio_list_empty(&new_data_requests[STARPU_IDLEFETCH])))
		{
			_starpu_data_request_prio_list_push_prio_list_back(&new_data_requests[STARPU_IDLEFETCH], &idle_requests[src_node]);
			idle_requests[src_node] = new_data_requests[STARPU_IDLEFETCH];
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&data_requests_list_mutex[src_node]);

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
			_starpu_wake_all_blocked_workers_on_node(src_node);
		}
#elif !defined(STARPU_NON_BLOCKING_DRIVERS)
		_starpu_wake_all_blocked_workers_on_node(src_node);
#endif
	}

	return ret;
}

int _starpu_handle_node_data_requests(unsigned src_node, unsigned may_alloc, unsigned *pushed)
{
	return __starpu_handle_node_data_requests(data_requests, src_node, may_alloc, MAX_PENDING_REQUESTS_PER_NODE, pushed, STARPU_FETCH);
}

int _starpu_handle_node_prefetch_requests(unsigned src_node, unsigned may_alloc, unsigned *pushed)
{
	return __starpu_handle_node_data_requests(prefetch_requests, src_node, may_alloc, MAX_PENDING_PREFETCH_REQUESTS_PER_NODE, pushed, STARPU_PREFETCH);
}

int _starpu_handle_node_idle_requests(unsigned src_node, unsigned may_alloc, unsigned *pushed)
{
	return __starpu_handle_node_data_requests(idle_requests, src_node, may_alloc, MAX_PENDING_IDLE_REQUESTS_PER_NODE, pushed, STARPU_IDLEFETCH);
}

static int _handle_pending_node_data_requests(unsigned src_node, unsigned force)
{
//	_STARPU_DEBUG("_starpu_handle_pending_node_data_requests ...\n");
//
	struct _starpu_data_request_prio_list new_data_requests_pending;
	unsigned taken, kept;

#ifdef STARPU_NON_BLOCKING_DRIVERS
	/* Here helgrind would should that this is an un protected access.
	 * We however don't care about missing an entry, we will get called
	 * again sooner or later. */
	if (!STARPU_RUNNING_ON_VALGRIND && _starpu_data_request_prio_list_empty(&data_requests_pending[src_node]))
		return 0;
#endif

#ifdef STARPU_NON_BLOCKING_DRIVERS
	if (!force)
	{
		if (STARPU_PTHREAD_MUTEX_TRYLOCK(&data_requests_pending_list_mutex[src_node]))
		{
			/* List is busy, do not bother with it */
			return 0;
		}
	}
	else
#endif
		/* We really want to handle requests */
		STARPU_PTHREAD_MUTEX_LOCK(&data_requests_pending_list_mutex[src_node]);

	if (_starpu_data_request_prio_list_empty(&data_requests_pending[src_node]))
	{
		/* there is no request */
		STARPU_PTHREAD_MUTEX_UNLOCK(&data_requests_pending_list_mutex[src_node]);
		return 0;
	}
	/* for all entries of the list */
	struct _starpu_data_request_prio_list local_list = data_requests_pending[src_node];
	_starpu_data_request_prio_list_init(&data_requests_pending[src_node]);

	STARPU_PTHREAD_MUTEX_UNLOCK(&data_requests_pending_list_mutex[src_node]);

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
	STARPU_PTHREAD_MUTEX_LOCK(&data_requests_pending_list_mutex[src_node]);
	data_requests_npending[src_node] -= taken - kept;
	if (kept)
		_starpu_data_request_prio_list_push_prio_list_back(&data_requests_pending[src_node], &new_data_requests_pending);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data_requests_pending_list_mutex[src_node]);

	return taken - kept;
}

int _starpu_handle_pending_node_data_requests(unsigned src_node)
{
	return _handle_pending_node_data_requests(src_node, 0);
}

int _starpu_handle_all_pending_node_data_requests(unsigned src_node)
{
	return _handle_pending_node_data_requests(src_node, 1);
}

/* Note: the returned value will be outdated since the locks are not taken at
 * entry/exit */
int _starpu_check_that_no_data_request_exists(unsigned node)
{
	int no_request;
	int no_pending;

	STARPU_PTHREAD_MUTEX_LOCK(&data_requests_list_mutex[node]);
	no_request = _starpu_data_request_prio_list_empty(&data_requests[node])
	          && _starpu_data_request_prio_list_empty(&prefetch_requests[node])
		  && _starpu_data_request_prio_list_empty(&idle_requests[node]);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data_requests_list_mutex[node]);
	STARPU_PTHREAD_MUTEX_LOCK(&data_requests_pending_list_mutex[node]);
	no_pending = !data_requests_npending[node];
	STARPU_PTHREAD_MUTEX_UNLOCK(&data_requests_pending_list_mutex[node]);

	return no_request && no_pending;
}

/* Note: the returned value will be outdated since the locks are not taken at
 * entry/exit */
int _starpu_check_that_no_data_request_is_pending(unsigned node)
{
	return !data_requests_npending[node];
}


void _starpu_update_prefetch_status(struct _starpu_data_request *r, enum _starpu_is_prefetch prefetch)
{
	STARPU_ASSERT(r->prefetch > prefetch);
	r->prefetch=prefetch;

	/* We have to promote chained_request too! */
	unsigned chained_req;
	for (chained_req = 0; chained_req < r->next_req_count; chained_req++)
	{
		struct _starpu_data_request *next_req = r->next_req[chained_req];
		if (next_req->prefetch > prefetch)
			_starpu_update_prefetch_status(next_req, prefetch);
	}

	STARPU_PTHREAD_MUTEX_LOCK(&data_requests_list_mutex[r->handling_node]);

	/* The request can be in a different list (handling request or the temp list)
	 * we have to check that it is really in the prefetch list. */
	if (_starpu_data_request_prio_list_ismember(&prefetch_requests[r->handling_node], r))
	{
		_starpu_data_request_prio_list_erase(&prefetch_requests[r->handling_node],r);
		_starpu_data_request_prio_list_push_back(&data_requests[r->handling_node],r);
	}
	/* The request can be in a different list (handling request or the temp list)
	 * we have to check that it is really in the idle list. */
	else if (_starpu_data_request_prio_list_ismember(&idle_requests[r->handling_node], r))
	{
		_starpu_data_request_prio_list_erase(&idle_requests[r->handling_node],r);
		if (prefetch >= STARPU_PREFETCH)
			_starpu_data_request_prio_list_push_back(&prefetch_requests[r->handling_node],r);
		else
			_starpu_data_request_prio_list_push_back(&data_requests[r->handling_node],r);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&data_requests_list_mutex[r->handling_node]);

#ifndef STARPU_NON_BLOCKING_DRIVERS
	_starpu_wake_all_blocked_workers_on_node(r->handling_node);
#endif
}
