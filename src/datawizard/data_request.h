/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2021       Federal University of Rio Grande do Sul (UFRGS)
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

/** @file */

/* This one includes us, so make sure to include it first */
#include <datawizard/coherency.h>

#ifndef __DATA_REQUEST_H__
#define __DATA_REQUEST_H__

#include <semaphore.h>
#include <datawizard/copy_driver.h>
#include <common/list.h>
#include <common/prio_list.h>
#include <common/starpu_spinlock.h>

#pragma GCC visibility push(hidden)

/* TODO: This should be tuned according to driver capabilities
 * Data interfaces should also have to declare how many asynchronous requests
 * they have actually started (think of e.g. csr).
 */
#define MAX_PENDING_REQUESTS_PER_NODE 5
#define MAX_PENDING_PREFETCH_REQUESTS_PER_NODE 2
#define MAX_PENDING_IDLE_REQUESTS_PER_NODE 1
/** Maximum time in us that we can afford pushing requests before going back to the driver loop, e.g. for checking GPU task termination */
#define MAX_PUSH_TIME 1000

struct _starpu_data_replicate;

struct _starpu_callback_list
{
	void (*callback_func)(void *);
	void *callback_arg;
	struct _starpu_callback_list *next;
};

enum _starpu_data_request_inout
{
	_STARPU_DATA_REQUEST_IN, _STARPU_DATA_REQUEST_OUT
};

/** This represents a data request, i.e. we want some data to get transferred
 * from a source to a destination. */
LIST_TYPE(_starpu_data_request,
	struct _starpu_spinlock lock;
	unsigned refcnt;
	const char *origin; /** Name of the function that triggered the request */

	starpu_data_handle_t handle;
	struct _starpu_data_replicate *src_replicate;
	struct _starpu_data_replicate *dst_replicate;

	/** Which memory node will actually perform the transfer.
	 * This is important in the CUDA/OpenCL case, where only the worker for
	 * the node can make the CUDA/OpenCL calls.
	 */
	unsigned handling_node;
	unsigned peer_node;
	enum _starpu_data_request_inout inout;

	/*
	 * What the destination node wants to do with the data: write to it,
	 * read it, or read and write to it. Only in the two latter cases we
	 * need an actual transfer, the first only needs an allocation.
	 *
	 * With mapped buffers, an additional case is mode = 0, which means
	 * unmapping the buffer.
	 */
	enum starpu_data_access_mode mode;

	/** Elements needed to make the transfer asynchronous */
	struct _starpu_async_channel async_channel;

	/** Whether the transfer is completed. */
	unsigned completed:1;

	/** Whether we have already added our reference to the dst replicate. */
	unsigned added_ref:1;

	/** Whether the request was canceled before being handled (because the transfer already happened another way). */
	unsigned canceled:2;

	/** Whether this is just a prefetch request */
	enum starpu_is_prefetch prefetch:3;

	/** Task this request is for */
	struct starpu_task *task;

	/** Number of tasks which used this as a prefetch */
	unsigned nb_tasks_prefetch;

	/** Priority of the request. Default is 0 */
	int prio;

	/** The value returned by the transfer function */
	int retval;

	/** The request will not actually be submitted until there remains
	 * dependencies. */
	unsigned ndeps;

	/** Some further tasks may have requested prefetches for the same data
	 * much later on, link with them */
	struct _starpu_data_request *next_same_req;

	/** in case we have a chain of request (eg. for nvidia multi-GPU), this
	 * is the list of requests which are waiting for this one. */
	struct _starpu_data_request *next_req[STARPU_MAXNODES+1];
	/** The number of requests in next_req */
	unsigned next_req_count;

	struct _starpu_callback_list *callbacks;

	unsigned long com_id;
)
PRIO_LIST_TYPE(_starpu_data_request, prio)

/** Everyone that wants to access some piece of data will post a request.
 * Not only StarPU internals, but also the application may put such requests */
LIST_TYPE(_starpu_data_requester,
	/** what kind of access is requested ? */
	enum starpu_data_access_mode mode;

	/** applications may also directly manipulate data */
	unsigned is_requested_by_codelet;

	/** in case this is a codelet that will do the access */
	struct _starpu_job *j;
	unsigned buffer_index;

	int prio;

	/** if this is more complicated ... (eg. application request)
	 * NB: this callback is not called with the lock taken !
	 */
	void (*ready_data_callback)(void *argcb);
	void *argcb;
)
PRIO_LIST_TYPE(_starpu_data_requester, prio)

void _starpu_init_data_request_lists(void);
void _starpu_deinit_data_request_lists(void);
void _starpu_post_data_request(struct _starpu_data_request *r);
/** returns 0 if we have pushed all requests, -EBUSY or -ENOMEM otherwise */
int _starpu_handle_node_data_requests(unsigned handling_node, unsigned peer_node, enum _starpu_data_request_inout inout, enum _starpu_may_alloc may_alloc, unsigned *pushed);
int _starpu_handle_node_prefetch_requests(unsigned handling_node, unsigned peer_node, enum _starpu_data_request_inout inout, enum _starpu_may_alloc may_alloc, unsigned *pushed);
int _starpu_handle_node_idle_requests(unsigned handling_node, unsigned peer_node, enum _starpu_data_request_inout inout, enum _starpu_may_alloc may_alloc, unsigned *pushed);

int _starpu_handle_pending_node_data_requests(unsigned handling_node, unsigned peer_node, enum _starpu_data_request_inout inout);
int _starpu_handle_all_pending_node_data_requests(unsigned handling_node, unsigned peer_node, enum _starpu_data_request_inout inout);

int _starpu_check_that_no_data_request_exists(unsigned handling_node);
int _starpu_check_that_no_data_request_is_pending(unsigned handling_node, unsigned peer_node, enum _starpu_data_request_inout inout);

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
							 const char *origin) STARPU_ATTRIBUTE_MALLOC;

int _starpu_wait_data_request_completion(struct _starpu_data_request *r, enum _starpu_may_alloc may_alloc);

void _starpu_data_request_append_callback(struct _starpu_data_request *r,
					  void (*callback_func)(void *),
					  void *callback_arg);

void _starpu_update_prefetch_status(struct _starpu_data_request *r, enum starpu_is_prefetch prefetch);

#pragma GCC visibility pop

#endif // __DATA_REQUEST_H__
