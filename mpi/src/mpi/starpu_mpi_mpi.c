/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2017       Guillaume Beauchamp
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

#include <stdlib.h>
#include <limits.h>
#include <starpu_mpi.h>
#include <starpu_mpi_datatype.h>
#include <starpu_mpi_private.h>
#include <starpu_mpi_cache.h>
#include <starpu_profiling.h>
#include <starpu_mpi_stats.h>
#include <starpu_mpi_cache.h>
#include <mpi/starpu_mpi_sync_data.h>
#include <mpi/starpu_mpi_early_data.h>
#include <mpi/starpu_mpi_early_request.h>
#include <starpu_mpi_select_node.h>
#include <mpi/starpu_mpi_tag.h>
#include <mpi/starpu_mpi_comm.h>
#include <starpu_mpi_init.h>
#include <common/config.h>
#include <common/thread.h>
#include <datawizard/interfaces/data_interface.h>
#include <datawizard/coherency.h>
#include <core/simgrid.h>
#include <core/task.h>
#include <core/topology.h>
#include <core/workers.h>

#ifdef STARPU_USE_MPI_MPI

/* Number of ready requests to process before polling for completed requests */
static unsigned nready_process;

/* Number of send requests to submit to MPI at the same time */
static unsigned ndetached_send;

static void _starpu_mpi_add_sync_point_in_fxt(void);
static void _starpu_mpi_handle_ready_request(struct _starpu_mpi_req *req);
static void _starpu_mpi_handle_request_termination(struct _starpu_mpi_req *req);
#ifdef STARPU_MPI_VERBOSE
static char *_starpu_mpi_request_type(enum _starpu_mpi_request_type request_type);
#endif
static void _starpu_mpi_handle_detached_request(struct _starpu_mpi_req *req);
static void _starpu_mpi_early_data_cb(void* arg);

/* The list of ready requests */
static struct _starpu_mpi_req_list ready_recv_requests;
static struct _starpu_mpi_req_prio_list ready_send_requests;

/* The list of detached requests that have already been submitted to MPI */
static struct _starpu_mpi_req_list detached_requests;
static unsigned detached_send_nrequests;
static starpu_pthread_mutex_t detached_requests_mutex;

/* Condition to wake up progression thread */
static starpu_pthread_cond_t progress_cond;
/* Condition to wake up waiting for all current MPI requests to finish */
static starpu_pthread_cond_t barrier_cond;
static starpu_pthread_mutex_t progress_mutex;
#ifndef STARPU_SIMGRID
static starpu_pthread_t progress_thread;
#endif
static int running = 0;

/* Driver taken by StarPU-MPI to process tasks when there is no requests to
 * handle instead of polling endlessly */
static struct starpu_driver *mpi_driver = NULL;
static int mpi_driver_call_freq = 0;
static int mpi_driver_task_freq = 0;

#ifdef STARPU_SIMGRID
static int wait_counter;
static starpu_pthread_cond_t wait_counter_cond;
static starpu_pthread_mutex_t wait_counter_mutex;
starpu_pthread_wait_t _starpu_mpi_thread_wait;
starpu_pthread_queue_t _starpu_mpi_thread_dontsleep;
#endif

/* Count requests posted by the application and not yet submitted to MPI */
static starpu_pthread_mutex_t mutex_posted_requests;
static starpu_pthread_mutex_t mutex_ready_requests;
static int posted_requests = 0, ready_requests = 0, newer_requests, barrier_running = 0;

#define _STARPU_MPI_INC_POSTED_REQUESTS(value) { STARPU_PTHREAD_MUTEX_LOCK(&mutex_posted_requests); posted_requests += value; STARPU_PTHREAD_MUTEX_UNLOCK(&mutex_posted_requests); }
#define _STARPU_MPI_INC_READY_REQUESTS(value) { STARPU_PTHREAD_MUTEX_LOCK(&mutex_ready_requests); ready_requests += value; STARPU_PTHREAD_MUTEX_UNLOCK(&mutex_ready_requests); }

extern struct _starpu_mpi_req *_starpu_mpi_irecv_common(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, unsigned detached, unsigned sync, void (*callback)(void *), void *arg, int sequential_consistency, int is_internal_req, starpu_ssize_t count);

#ifdef STARPU_SIMGRID
#pragma weak smpi_simulated_main_
extern int smpi_simulated_main_(int argc, char *argv[]);

#pragma weak smpi_process_set_user_data
#if !HAVE_DECL_SMPI_PROCESS_SET_USER_DATA && !defined(smpi_process_set_user_data)
extern void smpi_process_set_user_data(void *);
#endif
#endif

#ifdef STARPU_USE_FXT
static int trace_loop = 0;
#endif

 /********************************************************/
 /*                                                      */
 /*  Send/Receive functionalities                        */
 /*                                                      */
 /********************************************************/

struct _starpu_mpi_early_data_cb_args
{
	starpu_data_handle_t data_handle;
	starpu_data_handle_t early_handle;
	struct _starpu_mpi_req *req;
	void *buffer;
	size_t size;
};

void _starpu_mpi_submit_ready_request_inc(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_INC_POSTED_REQUESTS(1);
	_starpu_mpi_submit_ready_request(req);
}

void _starpu_mpi_coop_sends_build_tree(struct _starpu_mpi_coop_sends *coop_sends)
{
	(void)coop_sends;
	/* TODO: turn them into redirects & forwards */
}

void _starpu_mpi_submit_coop_sends(struct _starpu_mpi_coop_sends *coop_sends, int submit_control, int submit_data)
{
	(void)submit_control;
	unsigned i, n = coop_sends->n;

	/* Note: coop_sends might disappear very very soon after last request is submitted */
	for (i = 0; i < n; i++)
	{
		if (coop_sends->reqs_array[i]->request_type == SEND_REQ && submit_data)
		{
			_STARPU_MPI_DEBUG(0, "cooperative sends %p sending to %d\n", coop_sends, coop_sends->reqs_array[i]->node_tag.node.rank);
			_starpu_mpi_submit_ready_request(coop_sends->reqs_array[i]);
		}
		/* TODO: handle redirect requests */
	}
}

void _starpu_mpi_submit_ready_request(void *arg)
{
	_STARPU_MPI_LOG_IN();
	struct _starpu_mpi_req *req = arg;

	_STARPU_MPI_INC_POSTED_REQUESTS(-1);

	_STARPU_MPI_DEBUG(0, "new req %p srcdst %d tag %"PRIi64" and type %s %d\n", req, req->node_tag.node.rank, req->node_tag.data_tag, _starpu_mpi_request_type(req->request_type), req->backend->is_internal_req);

	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);

	if (req->request_type == RECV_REQ)
	{
		/* Case : the request is the internal receive request submitted
		 * by StarPU-MPI to receive incoming data without a matching
		 * early_request from the application. We immediately allocate the
		 * pointer associated to the data_handle, and push it into the
		 * ready_requests list, so as the real MPI request can be submitted
		 * before the next submission of the envelope-catching request. */
		if (req->backend->is_internal_req)
		{
			_starpu_mpi_datatype_allocate(req->data_handle, req);
			if (req->registered_datatype == 1)
			{
				req->count = 1;
				req->ptr = starpu_data_handle_to_pointer(req->data_handle, STARPU_MAIN_RAM);
			}
			else
			{
				STARPU_ASSERT(req->count);
				req->ptr = (void *)starpu_malloc_on_node_flags(STARPU_MAIN_RAM, req->count, 0);
			}

			_STARPU_MPI_DEBUG(3, "Pushing internal starpu_mpi_irecv request %p type %s tag %"PRIi64" src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n",
					  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, req->ptr,
					  req->datatype_name, (int)req->count, req->registered_datatype);
			_starpu_mpi_req_list_push_front(&ready_recv_requests, req);
			_STARPU_MPI_INC_READY_REQUESTS(+1);

			/* inform the starpu mpi thread that the request has been pushed in the ready_requests list */
			STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
			STARPU_PTHREAD_MUTEX_LOCK(&req->backend->posted_mutex);
			req->posted = 1;
			STARPU_PTHREAD_COND_BROADCAST(&req->backend->posted_cond);
			STARPU_PTHREAD_MUTEX_UNLOCK(&req->backend->posted_mutex);
			STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
		}
		else
		{
			/* test whether some data with the given tag and source have already been received by StarPU-MPI*/
			struct _starpu_mpi_early_data_handle *early_data_handle = _starpu_mpi_early_data_find(&req->node_tag);

			if (early_data_handle)
			{
				/* Case: a receive request for a data with the given tag and source has already been
				 * posted to MPI by StarPU. Asynchronously requests a Read permission over the temporary handle ,
				 * so as when the internal receive is completed, the _starpu_mpi_early_data_cb function
				 * will be called to bring the data back to the original data handle associated to the request.*/
				STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
				STARPU_PTHREAD_MUTEX_LOCK(&(early_data_handle->req_mutex));
				while (!(early_data_handle->req_ready))
					STARPU_PTHREAD_COND_WAIT(&(early_data_handle->req_cond), &(early_data_handle->req_mutex));
				STARPU_PTHREAD_MUTEX_UNLOCK(&(early_data_handle->req_mutex));
				STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);

				_STARPU_MPI_DEBUG(3, "The RECV request %p with tag %"PRIi64" has already been received, copying previously received data into handle's pointer..\n", req, req->node_tag.data_tag);
				STARPU_ASSERT(req->data_handle != early_data_handle->handle);

				req->backend->internal_req = early_data_handle->req;
				req->backend->early_data_handle = early_data_handle;

				struct _starpu_mpi_early_data_cb_args *cb_args;
				_STARPU_MPI_MALLOC(cb_args, sizeof(struct _starpu_mpi_early_data_cb_args));
				cb_args->data_handle = req->data_handle;
				cb_args->early_handle = early_data_handle->handle;
				cb_args->buffer = early_data_handle->buffer;
				cb_args->size = early_data_handle->size;
				cb_args->req = req;

				_STARPU_MPI_DEBUG(3, "Calling data_acquire_cb on starpu_mpi_copy_cb..\n");
				STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
				starpu_data_acquire_on_node_cb(early_data_handle->handle,STARPU_MAIN_RAM,STARPU_R,_starpu_mpi_early_data_cb,(void*) cb_args);
				STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
			}
			else
			{
				struct _starpu_mpi_req *sync_req = _starpu_mpi_sync_data_find(req->node_tag.data_tag, req->node_tag.node.rank, req->node_tag.node.comm);
				_STARPU_MPI_DEBUG(3, "----------> Looking for sync data for tag %"PRIi64" and src %d = %p\n", req->node_tag.data_tag, req->node_tag.node.rank, sync_req);
				if (sync_req)
				{
					/* Case: we already received the send envelope, we can proceed with the receive */
					req->sync = 1;
					_starpu_mpi_datatype_allocate(req->data_handle, req);
					if (req->registered_datatype == 1)
					{
						req->count = 1;
						req->ptr = starpu_data_handle_to_pointer(req->data_handle, STARPU_MAIN_RAM);
					}
					else
					{
						req->count = sync_req->count;
						STARPU_ASSERT(req->count);
						req->ptr = (void *)starpu_malloc_on_node_flags(STARPU_MAIN_RAM, req->count, 0);
					}
					_starpu_mpi_req_list_push_front(&ready_recv_requests, req);
					_STARPU_MPI_INC_READY_REQUESTS(+1);
					/* Throw away the dumb request that was only used to know that we got the envelope */
					_starpu_mpi_request_destroy(sync_req);
				}
				else
				{
					/* Case: no matching data has been received. Store the receive request as an early_request. */
					_STARPU_MPI_DEBUG(3, "Adding the pending receive request %p (srcdst %d tag %"PRIi64") into the request hashmap\n", req, req->node_tag.node.rank, req->node_tag.data_tag);
					_starpu_mpi_early_request_enqueue(req);
				}
			}
		}
	}
	else
	{
		if (req->request_type == SEND_REQ)
			_starpu_mpi_req_prio_list_push_front(&ready_send_requests, req);
		else
			_starpu_mpi_req_list_push_front(&ready_recv_requests, req);
		_STARPU_MPI_INC_READY_REQUESTS(+1);
		_STARPU_MPI_DEBUG(3, "Pushing new request %p type %s tag %"PRIi64" src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n",
				  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, req->ptr,
				  req->datatype_name, (int)req->count, req->registered_datatype);
	}

	newer_requests = 1;
	STARPU_PTHREAD_COND_BROADCAST(&progress_cond);
#ifdef STARPU_SIMGRID
	starpu_pthread_queue_signal(&_starpu_mpi_thread_dontsleep);
#endif
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
	_STARPU_MPI_LOG_OUT();
}

void _starpu_mpi_req_willpost(struct _starpu_mpi_req *req STARPU_ATTRIBUTE_UNUSED)
{
	_STARPU_MPI_INC_POSTED_REQUESTS(1);
}

#ifdef STARPU_SIMGRID
int _starpu_mpi_simgrid_mpi_test(unsigned *done, int *flag)
{
	*flag = 0;
	if (*done)
	{
		starpu_pthread_queue_signal(&_starpu_mpi_thread_dontsleep);
		*flag = 1;
	}
	return MPI_SUCCESS;
}

static void _starpu_mpi_simgrid_wait_req_func(void* arg)
{
	struct _starpu_simgrid_mpi_req *sim_req = arg;
	int ret;
	STARPU_PTHREAD_MUTEX_LOCK(&wait_counter_mutex);
	wait_counter++;
	STARPU_PTHREAD_MUTEX_UNLOCK(&wait_counter_mutex);

	ret = MPI_Wait(sim_req->request, sim_req->status);

	STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Wait returning %s", _starpu_mpi_get_mpi_error_code(ret));

	*(sim_req->done) = 1;
	starpu_pthread_queue_broadcast(sim_req->queue);

	free(sim_req);

	STARPU_PTHREAD_MUTEX_LOCK(&wait_counter_mutex);
	if (--wait_counter == 0)
		STARPU_PTHREAD_COND_SIGNAL(&wait_counter_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&wait_counter_mutex);
}

void _starpu_mpi_simgrid_wait_req(MPI_Request *request, MPI_Status *status, starpu_pthread_queue_t *queue, unsigned *done)
{
	struct _starpu_simgrid_mpi_req *sim_req;
	_STARPU_MPI_CALLOC(sim_req, 1, sizeof(struct _starpu_simgrid_mpi_req));
	sim_req->request = request;
	sim_req->status = status;
	sim_req->queue = queue;
	sim_req->done = done;
	*done = 0;

	_starpu_simgrid_xbt_thread_create("wait for mpi transfer", _starpu_mpi_simgrid_wait_req_func, sim_req);
}
#endif

 /********************************************************/
 /*                                                      */
 /*  Send functionalities                                */
 /*                                                      */
 /********************************************************/

static void _starpu_mpi_isend_data_func(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	_STARPU_MPI_DEBUG(0, "post MPI isend request %p type %s tag %"PRIi64" dst %d data %p datasize %ld ptr %p datatype '%s' count %d registered_datatype %d sync %d\n", req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, starpu_data_get_size(req->data_handle), req->ptr, req->datatype_name, (int)req->count, req->registered_datatype, req->sync);

	_starpu_mpi_comm_amounts_inc(req->node_tag.node.comm, req->node_tag.node.rank, req->datatype, req->count);

	_STARPU_MPI_TRACE_ISEND_SUBMIT_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag, 0);

	if (req->sync == 0)
	{
		_STARPU_MPI_COMM_TO_DEBUG(req, req->count, req->datatype, req->node_tag.node.rank, _STARPU_MPI_TAG_DATA, req->node_tag.data_tag, req->node_tag.node.comm);
		req->ret = MPI_Isend(req->ptr, req->count, req->datatype, req->node_tag.node.rank, _STARPU_MPI_TAG_DATA, req->node_tag.node.comm, &req->backend->data_request);
		STARPU_MPI_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Isend returning %s", _starpu_mpi_get_mpi_error_code(req->ret));
	}
	else
	{
		_STARPU_MPI_COMM_TO_DEBUG(req, req->count, req->datatype, req->node_tag.node.rank, _STARPU_MPI_TAG_SYNC_DATA, req->node_tag.data_tag, req->node_tag.node.comm);
		req->ret = MPI_Issend(req->ptr, req->count, req->datatype, req->node_tag.node.rank, _STARPU_MPI_TAG_SYNC_DATA, req->node_tag.node.comm, &req->backend->data_request);
		STARPU_MPI_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Issend returning %s", _starpu_mpi_get_mpi_error_code(req->ret));
	}

#ifdef STARPU_SIMGRID
	_starpu_mpi_simgrid_wait_req(&req->backend->data_request, &req->status_store, &req->queue, &req->done);
#endif

	_STARPU_MPI_TRACE_ISEND_SUBMIT_END(req->node_tag.node.rank, req->node_tag.data_tag, starpu_data_get_size(req->data_handle), req->pre_sync_jobid);

	/* somebody is perhaps waiting for the MPI request to be posted */
	STARPU_PTHREAD_MUTEX_LOCK(&req->backend->req_mutex);
	req->submitted = 1;
	STARPU_PTHREAD_COND_BROADCAST(&req->backend->req_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&req->backend->req_mutex);

	_starpu_mpi_handle_detached_request(req);

	_STARPU_MPI_LOG_OUT();
}

void _starpu_mpi_isend_size_func(struct _starpu_mpi_req *req)
{
	_starpu_mpi_datatype_allocate(req->data_handle, req);

	_STARPU_MPI_CALLOC(req->backend->envelope, 1,sizeof(struct _starpu_mpi_envelope));
	req->backend->envelope->mode = _STARPU_MPI_ENVELOPE_DATA;
	req->backend->envelope->data_tag = req->node_tag.data_tag;
	req->backend->envelope->sync = req->sync;

	if (req->registered_datatype == 1)
	{
		int size, ret;
		req->count = 1;
		req->ptr = starpu_data_handle_to_pointer(req->data_handle, STARPU_MAIN_RAM);

		MPI_Type_size(req->datatype, &size);
		req->backend->envelope->size = (starpu_ssize_t)req->count * size;
		_STARPU_MPI_DEBUG(20, "Post MPI isend count (%ld) datatype_size %ld request to %d\n",req->count,starpu_data_get_size(req->data_handle), req->node_tag.node.rank);
		_STARPU_MPI_COMM_TO_DEBUG(req->backend->envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.node.rank, _STARPU_MPI_TAG_ENVELOPE, req->backend->envelope->data_tag, req->node_tag.node.comm);
		ret = MPI_Isend(req->backend->envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.node.rank, _STARPU_MPI_TAG_ENVELOPE, req->node_tag.node.comm, &req->backend->size_req);
		STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "when sending envelope, MPI_Isend returning %s", _starpu_mpi_get_mpi_error_code(ret));
	}
	else
	{
		int ret;

 		// Do not pack the data, just try to find out the size
		starpu_data_pack(req->data_handle, NULL, &(req->backend->envelope->size));

		if (req->backend->envelope->size != -1)
 		{
 			// We already know the size of the data, let's send it to overlap with the packing of the data
			_STARPU_MPI_DEBUG(20, "Sending size %ld (%ld %s) to node %d (first call to pack)\n", req->backend->envelope->size, sizeof(req->count), "MPI_BYTE", req->node_tag.node.rank);
			req->count = req->backend->envelope->size;
			_STARPU_MPI_COMM_TO_DEBUG(req->backend->envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.node.rank, _STARPU_MPI_TAG_ENVELOPE, req->backend->envelope->data_tag, req->node_tag.node.comm);
			ret = MPI_Isend(req->backend->envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.node.rank, _STARPU_MPI_TAG_ENVELOPE, req->node_tag.node.comm, &req->backend->size_req);
			STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "when sending size, MPI_Isend returning %s", _starpu_mpi_get_mpi_error_code(ret));
 		}

 		// Pack the data
 		starpu_data_pack(req->data_handle, &req->ptr, &req->count);
		if (req->backend->envelope->size == -1)
 		{
 			// We know the size now, let's send it
			_STARPU_MPI_DEBUG(20, "Sending size %ld (%ld %s) to node %d (second call to pack)\n", req->backend->envelope->size, sizeof(req->count), "MPI_BYTE", req->node_tag.node.rank);
			_STARPU_MPI_COMM_TO_DEBUG(req->backend->envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.node.rank, _STARPU_MPI_TAG_ENVELOPE, req->backend->envelope->data_tag, req->node_tag.node.comm);
			ret = MPI_Isend(req->backend->envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.node.rank, _STARPU_MPI_TAG_ENVELOPE, req->node_tag.node.comm, &req->backend->size_req);
			STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "when sending size, MPI_Isend returning %s", _starpu_mpi_get_mpi_error_code(ret));
 		}
 		else
 		{
 			// We check the size returned with the 2 calls to pack is the same
			STARPU_MPI_ASSERT_MSG(req->count == req->backend->envelope->size, "Calls to pack_data returned different sizes %ld != %ld", req->count, req->backend->envelope->size);
 		}
		// We can send the data now
	}

	if (req->sync)
	{
		// If the data is to be sent in synchronous mode, we need to wait for the receiver ready message
		_starpu_mpi_sync_data_add(req);
	}
	else
	{
		// Otherwise we can send the data
		_starpu_mpi_isend_data_func(req);
	}
}

/********************************************************/
/*                                                      */
/*  receive functionalities                             */
/*                                                      */
/********************************************************/

void _starpu_mpi_irecv_size_func(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	_STARPU_MPI_DEBUG(0, "post MPI irecv request %p type %s tag %"PRIi64" src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n", req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);

	_STARPU_MPI_TRACE_IRECV_SUBMIT_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag);

	if (req->sync)
	{
		struct _starpu_mpi_envelope *_envelope;
		_STARPU_MPI_CALLOC(_envelope, 1, sizeof(struct _starpu_mpi_envelope));
		_envelope->mode = _STARPU_MPI_ENVELOPE_SYNC_READY;
		_envelope->data_tag = req->node_tag.data_tag;
		_STARPU_MPI_DEBUG(20, "Telling node %d it can send the data and waiting for the data back ...\n", req->node_tag.node.rank);
		_STARPU_MPI_COMM_TO_DEBUG(_envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.node.rank, _STARPU_MPI_TAG_ENVELOPE, _envelope->data_tag, req->node_tag.node.comm);
		req->ret = MPI_Send(_envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.node.rank, _STARPU_MPI_TAG_ENVELOPE, req->node_tag.node.comm);
		STARPU_MPI_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Send returning %s", _starpu_mpi_get_mpi_error_code(req->ret));
		free(_envelope);
		_envelope = NULL;
	}

	if (req->sync)
	{
		_STARPU_MPI_COMM_FROM_DEBUG(req, req->count, req->datatype, req->node_tag.node.rank, _STARPU_MPI_TAG_SYNC_DATA, req->node_tag.data_tag, req->node_tag.node.comm);
		req->ret = MPI_Irecv(req->ptr, req->count, req->datatype, req->node_tag.node.rank, _STARPU_MPI_TAG_SYNC_DATA, req->node_tag.node.comm, &req->backend->data_request);
	}
	else
	{
		_STARPU_MPI_COMM_FROM_DEBUG(req, req->count, req->datatype, req->node_tag.node.rank, _STARPU_MPI_TAG_DATA, req->node_tag.data_tag, req->node_tag.node.comm);
		req->ret = MPI_Irecv(req->ptr, req->count, req->datatype, req->node_tag.node.rank, _STARPU_MPI_TAG_DATA, req->node_tag.node.comm, &req->backend->data_request);
	}
#ifdef STARPU_SIMGRID
	_starpu_mpi_simgrid_wait_req(&req->backend->data_request, &req->status_store, &req->queue, &req->done);
#endif
	STARPU_MPI_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_IRecv returning %s", _starpu_mpi_get_mpi_error_code(req->ret));

	_STARPU_MPI_TRACE_IRECV_SUBMIT_END(req->node_tag.node.rank, req->node_tag.data_tag);

	/* somebody is perhaps waiting for the MPI request to be posted */
	STARPU_PTHREAD_MUTEX_LOCK(&req->backend->req_mutex);
	req->submitted = 1;
	STARPU_PTHREAD_COND_BROADCAST(&req->backend->req_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&req->backend->req_mutex);

	_starpu_mpi_handle_detached_request(req);

	_STARPU_MPI_LOG_OUT();
}

/********************************************************/
/*                                                      */
/*  Wait functionalities                                */
/*                                                      */
/********************************************************/

#ifndef STARPU_SIMGRID
void _starpu_mpi_wait_func(struct _starpu_mpi_req *waiting_req)
{
	_STARPU_MPI_LOG_IN();
	/* Which is the mpi request we are waiting for ? */
	struct _starpu_mpi_req *req = waiting_req->backend->other_request;

	_STARPU_MPI_TRACE_UWAIT_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag);
	if (req->backend->data_request != MPI_REQUEST_NULL)
	{
		req->ret = MPI_Wait(&req->backend->data_request, waiting_req->status);
		STARPU_MPI_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Wait returning %s", _starpu_mpi_get_mpi_error_code(req->ret));
	}
	_STARPU_MPI_TRACE_UWAIT_END(req->node_tag.node.rank, req->node_tag.data_tag);

	_starpu_mpi_handle_request_termination(req);

	_STARPU_MPI_LOG_OUT();
}
#endif

int _starpu_mpi_wait(starpu_mpi_req *public_req, MPI_Status *status)
{
	int ret;
	struct _starpu_mpi_req *req = *public_req;

	_STARPU_MPI_LOG_IN();

#ifdef STARPU_SIMGRID
	_STARPU_MPI_TRACE_UWAIT_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag);
	starpu_pthread_wait_t wait;
	starpu_pthread_wait_init(&wait);
	starpu_pthread_queue_register(&wait, &req->queue);
	while (1)
	{
		starpu_pthread_wait_reset(&wait);
		if (req->done)
			break;
		starpu_pthread_wait_wait(&wait);
	}
	starpu_pthread_queue_unregister(&wait, &req->queue);
	starpu_pthread_wait_destroy(&wait);
	_STARPU_MPI_TRACE_UWAIT_END(req->node_tag.node.rank, req->node_tag.data_tag);

	if (status)
		*status = req->status_store;
	_starpu_mpi_handle_request_termination(req);
#else
	struct _starpu_mpi_req *waiting_req;
	/* We cannot try to complete a MPI request that was not actually posted
	 * to MPI yet. */
	STARPU_PTHREAD_MUTEX_LOCK(&(req->backend->req_mutex));
	while (!(req->submitted))
		STARPU_PTHREAD_COND_WAIT(&(req->backend->req_cond), &(req->backend->req_mutex));
	STARPU_PTHREAD_MUTEX_UNLOCK(&(req->backend->req_mutex));

	/* Initialize the request structure */
	 _starpu_mpi_request_init(&waiting_req);
	waiting_req->prio = INT_MAX;
	waiting_req->status = status;
	waiting_req->backend->other_request = req;
	waiting_req->func = _starpu_mpi_wait_func;
	waiting_req->request_type = WAIT_REQ;

	_starpu_mpi_submit_ready_request_inc(waiting_req);

	/* We wait for the MPI request to finish */
	STARPU_PTHREAD_MUTEX_LOCK(&req->backend->req_mutex);
	while (!req->completed)
		STARPU_PTHREAD_COND_WAIT(&req->backend->req_cond, &req->backend->req_mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&req->backend->req_mutex);

	/* The internal request structure was automatically allocated */
	_starpu_mpi_request_destroy(waiting_req);
#endif

	*public_req = NULL;
	if (req->backend->internal_req)
	{
		_starpu_mpi_request_destroy(req->backend->internal_req);
	}
	ret = req->ret;
	_starpu_mpi_request_destroy(req);

	_STARPU_MPI_LOG_OUT();
	return ret;
}

/********************************************************/
/*                                                      */
/*  Test functionalities                                */
/*                                                      */
/********************************************************/

#ifndef STARPU_SIMGRID
void _starpu_mpi_test_func(struct _starpu_mpi_req *testing_req)
{
	_STARPU_MPI_LOG_IN();
	/* Which is the mpi request we are testing for ? */
	struct _starpu_mpi_req *req = testing_req->backend->other_request;

	_STARPU_MPI_DEBUG(0, "Test request %p type %s tag %"PRIi64" src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n",
			  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, req->ptr,
			  req->datatype_name, (int)req->count, req->registered_datatype);

	_STARPU_MPI_TRACE_UTESTING_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag);

	req->ret = MPI_Test(&req->backend->data_request, testing_req->flag, testing_req->status);

	STARPU_MPI_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Test returning %s", _starpu_mpi_get_mpi_error_code(req->ret));

	_STARPU_MPI_TRACE_UTESTING_END(req->node_tag.node.rank, req->node_tag.data_tag);

	if (*testing_req->flag)
	{
		testing_req->ret = req->ret;
		_starpu_mpi_handle_request_termination(req);
	}

	STARPU_PTHREAD_MUTEX_LOCK(&testing_req->backend->req_mutex);
	testing_req->completed = 1;
	STARPU_PTHREAD_COND_SIGNAL(&testing_req->backend->req_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&testing_req->backend->req_mutex);
	_STARPU_MPI_LOG_OUT();
}
#endif

int _starpu_mpi_test(starpu_mpi_req *public_req, int *flag, MPI_Status *status)
{
	_STARPU_MPI_LOG_IN();
	int ret = 0;

	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_test needs a valid starpu_mpi_req");

	struct _starpu_mpi_req *req = *public_req;

	STARPU_MPI_ASSERT_MSG(!req->detached, "MPI_Test cannot be called on a detached request");

	STARPU_VALGRIND_YIELD();

#ifdef STARPU_SIMGRID
	ret = req->ret = _starpu_mpi_simgrid_mpi_test(&req->done, flag);
	if (*flag)
	{
		if (status)
			*status = req->status_store;
		_starpu_mpi_handle_request_termination(req);
	}
#else
	STARPU_PTHREAD_MUTEX_LOCK(&req->backend->req_mutex);
	unsigned submitted = req->submitted;
	STARPU_PTHREAD_MUTEX_UNLOCK(&req->backend->req_mutex);

	if (submitted)
	{
		struct _starpu_mpi_req *testing_req;

		/* Initialize the request structure */
		_starpu_mpi_request_init(&testing_req);
		testing_req->prio = INT_MAX;
		testing_req->flag = flag;
		testing_req->status = status;
		testing_req->backend->other_request = req;
		testing_req->func = _starpu_mpi_test_func;
		testing_req->completed = 0;
		testing_req->request_type = TEST_REQ;

		_starpu_mpi_submit_ready_request_inc(testing_req);

		/* We wait for the test request to finish */
		STARPU_PTHREAD_MUTEX_LOCK(&(testing_req->backend->req_mutex));
		while (!(testing_req->completed))
			STARPU_PTHREAD_COND_WAIT(&(testing_req->backend->req_cond), &(testing_req->backend->req_mutex));
		STARPU_PTHREAD_MUTEX_UNLOCK(&(testing_req->backend->req_mutex));

		ret = testing_req->ret;

		_starpu_mpi_request_destroy(testing_req);
	}
	else
	{
		*flag = 0;
	}
#endif

	if (*flag)
	{
		/* The request was completed so we free the internal
		 * request structure which was automatically allocated
		 * */
		*public_req = NULL;
		if (req->backend->internal_req)
		{
			_starpu_mpi_request_destroy(req->backend->internal_req);
		}
		_starpu_mpi_request_destroy(req);
	}

	_STARPU_MPI_LOG_OUT();
	return ret;
}

/********************************************************/
/*                                                      */
/*  Barrier functionalities                             */
/*                                                      */
/********************************************************/

static void _starpu_mpi_barrier_func(struct _starpu_mpi_req *barrier_req)
{
	_STARPU_MPI_LOG_IN();

	barrier_req->ret = MPI_Barrier(barrier_req->node_tag.node.comm);
	STARPU_MPI_ASSERT_MSG(barrier_req->ret == MPI_SUCCESS, "MPI_Barrier returning %s", _starpu_mpi_get_mpi_error_code(barrier_req->ret));

	_starpu_mpi_handle_request_termination(barrier_req);
	_STARPU_MPI_LOG_OUT();
}

int _starpu_mpi_barrier(MPI_Comm comm)
{
	struct _starpu_mpi_req *barrier_req;
	int ret = posted_requests+ready_requests;

	_STARPU_MPI_LOG_IN();

	/* First wait for *both* all tasks and MPI requests to finish, in case
	 * some tasks generate MPI requests, MPI requests generate tasks, etc.
	 */
	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
	STARPU_MPI_ASSERT_MSG(!barrier_running, "Concurrent starpu_mpi_barrier is not implemented, even on different communicators");
	barrier_running = 1;
	do
	{
		while (posted_requests || ready_requests)
			/* Wait for all current MPI requests to finish */
			STARPU_PTHREAD_COND_WAIT(&barrier_cond, &progress_mutex);
		/* No current request, clear flag */
		newer_requests = 0;
		STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
		/* Now wait for all tasks */
		starpu_task_wait_for_all();
		STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
		/* Check newer_requests again, in case some MPI requests
		 * triggered by tasks completed and triggered tasks between
		 * wait_for_all finished and we take the lock */
	} while (posted_requests || ready_requests || newer_requests);
	barrier_running = 0;
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

	/* Initialize the request structure */
	_starpu_mpi_request_init(&barrier_req);
	barrier_req->prio = INT_MAX;
	barrier_req->func = _starpu_mpi_barrier_func;
	barrier_req->request_type = BARRIER_REQ;
	barrier_req->node_tag.node.comm = comm;

	_STARPU_MPI_INC_POSTED_REQUESTS(1);
	_starpu_mpi_submit_ready_request(barrier_req);

	/* We wait for the MPI request to finish */
	STARPU_PTHREAD_MUTEX_LOCK(&barrier_req->backend->req_mutex);
	while (!barrier_req->completed)
		STARPU_PTHREAD_COND_WAIT(&barrier_req->backend->req_cond, &barrier_req->backend->req_mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&barrier_req->backend->req_mutex);

	_starpu_mpi_request_destroy(barrier_req);
	_STARPU_MPI_LOG_OUT();

	return ret;
}

/********************************************************/
/*                                                      */
/*  Progression                                         */
/*                                                      */
/********************************************************/

#ifdef STARPU_MPI_VERBOSE
static char *_starpu_mpi_request_type(enum _starpu_mpi_request_type request_type)
{
	switch (request_type)
	{
		case SEND_REQ: return "SEND_REQ";
		case RECV_REQ: return "RECV_REQ";
		case WAIT_REQ: return "WAIT_REQ";
		case TEST_REQ: return "TEST_REQ";
		case BARRIER_REQ: return "BARRIER_REQ";
		case UNKNOWN_REQ: return "UNSET_REQ";
		default: return "unknown request type";
	}
}
#endif

static void _starpu_mpi_handle_request_termination(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	_STARPU_MPI_DEBUG(2, "complete MPI request %p type %s tag %"PRIi64" src %d data %p ptr %p datatype '%s' count %d registered_datatype %d internal_req %p\n",
			  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, req->ptr,
			  req->datatype_name, (int)req->count, req->registered_datatype, req->backend->internal_req);

	if (req->backend->internal_req)
	{
		free(req->backend->early_data_handle);
		req->backend->early_data_handle = NULL;
	}
	else
	{
		if (req->request_type == RECV_REQ || req->request_type == SEND_REQ)
		{
			if (req->registered_datatype == 0)
			{
				if (req->request_type == SEND_REQ)
				{
					// We need to make sure the communication for sending the size
					// has completed, as MPI can re-order messages, let's call
					// MPI_Wait to make sure data have been sent
					int ret;
					ret = MPI_Wait(&req->backend->size_req, MPI_STATUS_IGNORE);
					STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Wait returning %s", _starpu_mpi_get_mpi_error_code(ret));
					starpu_free_on_node_flags(STARPU_MAIN_RAM, (uintptr_t)req->ptr, req->count, 0);
					req->ptr = NULL;
				}
				else if (req->request_type == RECV_REQ)
				{
					// req->ptr is freed by starpu_data_unpack
					starpu_data_unpack(req->data_handle, req->ptr, req->count);
					starpu_memory_deallocate(STARPU_MAIN_RAM, req->count);
				}
			}
			else
			{
				_starpu_mpi_datatype_free(req->data_handle, &req->datatype);
			}
		}
		_STARPU_MPI_TRACE_TERMINATED(req, req->node_tag.node.rank, req->node_tag.data_tag);
	}

	_starpu_mpi_release_req_data(req);

	if (req->backend->envelope)
	{
		free(req->backend->envelope);
		req->backend->envelope = NULL;
	}

	/* Execute the specified callback, if any */
	if (req->callback)
		req->callback(req->callback_arg);

	/* tell anyone potentially waiting on the request that it is
	 * terminated now */
	STARPU_PTHREAD_MUTEX_LOCK(&req->backend->req_mutex);
	req->completed = 1;
	STARPU_PTHREAD_COND_BROADCAST(&req->backend->req_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&req->backend->req_mutex);
	_STARPU_MPI_LOG_OUT();
}

/* This is called when the data is now received in the early data handle, we can
 * now copy it over to the real handle. */
static void _starpu_mpi_early_data_cb(void* arg)
{
	struct _starpu_mpi_early_data_cb_args *args = arg;

	if (args->buffer)
	{
		/* Data has been received as a raw memory, it has to be unpacked */
		struct starpu_data_interface_ops *itf_src = starpu_data_get_interface_ops(args->early_handle);
		struct starpu_data_interface_ops *itf_dst = starpu_data_get_interface_ops(args->data_handle);
		MPI_Datatype datatype = _starpu_mpi_datatype_get_user_defined_datatype(args->data_handle);

		if (datatype)
		{
			int position=0;
			void *ptr = starpu_data_get_local_ptr(args->data_handle);
			MPI_Unpack(args->buffer, itf_src->get_size(args->early_handle), &position, ptr, 1, datatype, args->req->node_tag.node.comm);
			starpu_free_on_node_flags(STARPU_MAIN_RAM, (uintptr_t) args->buffer, args->size, 0);
			args->buffer = NULL;
			_starpu_mpi_datatype_free(args->data_handle, &datatype);
		}
		else
		{
			STARPU_MPI_ASSERT_MSG(itf_dst->unpack_data, "The data interface does not define an unpack function\n");
			itf_dst->unpack_data(args->data_handle, STARPU_MAIN_RAM, args->buffer, itf_src->get_size(args->early_handle));
			args->buffer = NULL;
		}
	}
	else
	{
		struct starpu_data_interface_ops *itf = starpu_data_get_interface_ops(args->early_handle);
		void* itf_src = starpu_data_get_interface_on_node(args->early_handle, STARPU_MAIN_RAM);
		void* itf_dst = starpu_data_get_interface_on_node(args->data_handle, STARPU_MAIN_RAM);

		if (!itf->copy_methods->ram_to_ram)
		{
			_STARPU_MPI_DEBUG(3, "Initiating any_to_any copy..\n");
			itf->copy_methods->any_to_any(itf_src, STARPU_MAIN_RAM, itf_dst, STARPU_MAIN_RAM, NULL);
		}
		else
		{
			_STARPU_MPI_DEBUG(3, "Initiating ram_to_ram copy..\n");
			itf->copy_methods->ram_to_ram(itf_src, STARPU_MAIN_RAM, itf_dst, STARPU_MAIN_RAM);
		}
	}

	_STARPU_MPI_DEBUG(3, "Done, handling release of early_handle..\n");
	starpu_data_release_on_node(args->early_handle, STARPU_MAIN_RAM);

	_STARPU_MPI_DEBUG(3, "Done, handling unregister of early_handle..\n");
	/* XXX: note that we have already freed the registered buffer above. In
	 * principle that's unsafe. As of now it is fine because StarPU has no
	 reason to access it. */
	starpu_data_unregister_submit(args->early_handle);

	_STARPU_MPI_DEBUG(3, "Done, handling request %p termination of the already received request\n",args->req);
	// If the request is detached, we need to call _starpu_mpi_handle_request_termination
	// as it will not be called automatically as the request is not in the list detached_requests
	if (args->req)
	{
		if (args->req->detached)
		{
			/* have the internal request destroyed now or when completed */
			STARPU_PTHREAD_MUTEX_LOCK(&args->req->backend->internal_req->backend->req_mutex);
			if (args->req->backend->internal_req->backend->to_destroy)
			{
				/* The request completed first, can now destroy it */
				STARPU_PTHREAD_MUTEX_UNLOCK(&args->req->backend->internal_req->backend->req_mutex);
				_starpu_mpi_request_destroy(args->req->backend->internal_req);
			}
			else
			{
				/* The request didn't complete yet, tell it to destroy it when it completes */
				args->req->backend->internal_req->backend->to_destroy = 1;
				STARPU_PTHREAD_MUTEX_UNLOCK(&args->req->backend->internal_req->backend->req_mutex);
			}
			_starpu_mpi_handle_request_termination(args->req);
			_starpu_mpi_request_destroy(args->req);
		}
		else
		{
			// else: If the request is not detached its termination will
			// be handled when calling starpu_mpi_wait
			// We store in the application request the internal MPI
			// request so that it can be used by starpu_mpi_wait
			args->req->backend->data_request = args->req->backend->internal_req->backend->data_request;
			STARPU_PTHREAD_MUTEX_LOCK(&args->req->backend->req_mutex);
			args->req->submitted = 1;
			STARPU_PTHREAD_COND_BROADCAST(&args->req->backend->req_cond);
			STARPU_PTHREAD_MUTEX_UNLOCK(&args->req->backend->req_mutex);
#ifdef STARPU_SIMGRID
			args->req->done = 1;
#endif
		}
	}

	free(args);
	args = NULL;
}

static void _starpu_mpi_test_detached_requests(void)
{
	//_STARPU_MPI_LOG_IN();
	int flag;
	struct _starpu_mpi_req *req;

	STARPU_PTHREAD_MUTEX_LOCK(&detached_requests_mutex);

	if (_starpu_mpi_req_list_empty(&detached_requests))
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&detached_requests_mutex);
		//_STARPU_MPI_LOG_OUT();
		return;
	}

	_STARPU_MPI_TRACE_TESTING_DETACHED_BEGIN();
	req = _starpu_mpi_req_list_begin(&detached_requests);
	while (req != _starpu_mpi_req_list_end(&detached_requests))
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&detached_requests_mutex);

		_STARPU_MPI_TRACE_TEST_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag);
		//_STARPU_MPI_DEBUG(3, "Test detached request %p - mpitag %"PRIi64" - TYPE %s %d\n", &req->backend->data_request, req->node_tag.data_tag, _starpu_mpi_request_type(req->request_type), req->node_tag.node.rank);
#ifdef STARPU_SIMGRID
		req->ret = _starpu_mpi_simgrid_mpi_test(&req->done, &flag);
#else
		STARPU_MPI_ASSERT_MSG(req->backend->data_request != MPI_REQUEST_NULL, "Cannot test completion of the request MPI_REQUEST_NULL");
		req->ret = MPI_Test(&req->backend->data_request, &flag, MPI_STATUS_IGNORE);
#endif

		STARPU_MPI_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Test returning %s", _starpu_mpi_get_mpi_error_code(req->ret));
		_STARPU_MPI_TRACE_TEST_END(req->node_tag.node.rank, req->node_tag.data_tag);

		if (!flag)
		{
			req = _starpu_mpi_req_list_next(req);
		}
		else
		{
			_STARPU_MPI_TRACE_POLLING_END();
		     	struct _starpu_mpi_req *next_req;
			next_req = _starpu_mpi_req_list_next(req);

			_STARPU_MPI_TRACE_COMPLETE_BEGIN(req->request_type, req->node_tag.node.rank, req->node_tag.data_tag);

			STARPU_PTHREAD_MUTEX_LOCK(&detached_requests_mutex);
			if (req->request_type == SEND_REQ)
				detached_send_nrequests--;
			_starpu_mpi_req_list_erase(&detached_requests, req);
			STARPU_PTHREAD_MUTEX_UNLOCK(&detached_requests_mutex);
			_starpu_mpi_handle_request_termination(req);

			_STARPU_MPI_TRACE_COMPLETE_END(req->request_type, req->node_tag.node.rank, req->node_tag.data_tag);

			STARPU_PTHREAD_MUTEX_LOCK(&req->backend->req_mutex);
			/* We don't want to free internal non-detached
			   requests, we need to get their MPI request before
			   destroying them */
			if (req->backend->is_internal_req && !req->backend->to_destroy)
			{
				/* We have completed the request, let the application request destroy it */
				req->backend->to_destroy = 1;
				STARPU_PTHREAD_MUTEX_UNLOCK(&req->backend->req_mutex);
			}
			else
			{
				STARPU_PTHREAD_MUTEX_UNLOCK(&req->backend->req_mutex);
				_starpu_mpi_request_destroy(req);
			}

			req = next_req;
			_STARPU_MPI_TRACE_POLLING_BEGIN();
		}

		STARPU_PTHREAD_MUTEX_LOCK(&detached_requests_mutex);
	}
	_STARPU_MPI_TRACE_TESTING_DETACHED_END();

	STARPU_PTHREAD_MUTEX_UNLOCK(&detached_requests_mutex);
	//_STARPU_MPI_LOG_OUT();
}

static void _starpu_mpi_handle_detached_request(struct _starpu_mpi_req *req)
{
	if (req->detached)
	{
		/* put the submitted request into the list of pending requests
		 * so that it can be handled by the progression mechanisms */
		STARPU_PTHREAD_MUTEX_LOCK(&detached_requests_mutex);
		if (req->request_type == SEND_REQ)
			detached_send_nrequests++;
		_starpu_mpi_req_list_push_back(&detached_requests, req);
		STARPU_PTHREAD_MUTEX_UNLOCK(&detached_requests_mutex);

		STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
		STARPU_PTHREAD_COND_SIGNAL(&progress_cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
	}
}

static void _starpu_mpi_handle_ready_request(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();
	STARPU_MPI_ASSERT_MSG(req, "Invalid request");

	/* submit the request to MPI */
	_STARPU_MPI_DEBUG(2, "Handling new request %p type %s tag %"PRIi64" src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n",
			  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle,
			  req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);
	req->func(req);

	_STARPU_MPI_LOG_OUT();
}

static void _starpu_mpi_receive_early_data(struct _starpu_mpi_envelope *envelope, MPI_Status status, MPI_Comm comm)
{
	_STARPU_MPI_DEBUG(20, "Request with tag %"PRIi64" and source %d not found, creating a early_data_handle to receive incoming data..\n", envelope->data_tag, status.MPI_SOURCE);
	_STARPU_MPI_DEBUG(20, "Request sync %d\n", envelope->sync);

	struct _starpu_mpi_early_data_handle* early_data_handle = _starpu_mpi_early_data_create(envelope, status.MPI_SOURCE, comm);
	_starpu_mpi_early_data_add(early_data_handle);

	starpu_data_handle_t data_handle;
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
	data_handle = _starpu_mpi_tag_get_data_handle_from_tag(envelope->data_tag);
	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);

	if (data_handle && starpu_data_get_interface_id(data_handle) < STARPU_MAX_INTERFACE_ID)
	{
		/* We know which data will receive it and we won't have to unpack, use just the same kind of data.  */
		early_data_handle->buffer = NULL;
		starpu_data_register_same(&early_data_handle->handle, data_handle);
		//_starpu_mpi_early_data_add(early_data_handle);
	}
	else
	{
		/* The application has not registered yet a data with the tag,
		 * we are going to receive the data as a raw memory, and give it
		 * to the application when it post a receive for this tag
		 */
		_STARPU_MPI_DEBUG(3, "Posting a receive for a data of size %d which has not yet been registered\n", (int)envelope->size);
		early_data_handle->buffer = (void *)starpu_malloc_on_node_flags(STARPU_MAIN_RAM, envelope->size, 0);
		early_data_handle->size = envelope->size;
		starpu_variable_data_register(&early_data_handle->handle, STARPU_MAIN_RAM, (uintptr_t) early_data_handle->buffer, envelope->size);
		//_starpu_mpi_early_data_add(early_data_handle);
	}

	_STARPU_MPI_DEBUG(20, "Posting internal detached irecv on early_data_handle with tag %"PRIi64" from comm %ld src %d ..\n",
			  early_data_handle->node_tag.data_tag, (long int)comm, status.MPI_SOURCE);
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
	early_data_handle->req = _starpu_mpi_irecv_common(early_data_handle->handle, status.MPI_SOURCE,
							  early_data_handle->node_tag.data_tag, comm, 1, 0,
							  NULL, NULL, 1, 1, envelope->size);
	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);

	// We wait until the request is pushed in the
	// ready_request list
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
	STARPU_PTHREAD_MUTEX_LOCK(&(early_data_handle->req->backend->posted_mutex));
	while (!(early_data_handle->req->posted))
		STARPU_PTHREAD_COND_WAIT(&(early_data_handle->req->backend->posted_cond), &(early_data_handle->req->backend->posted_mutex));
	STARPU_PTHREAD_MUTEX_UNLOCK(&(early_data_handle->req->backend->posted_mutex));

#ifdef STARPU_DEVEL
#warning check if req_ready is still necessary
#endif
	STARPU_PTHREAD_MUTEX_LOCK(&early_data_handle->req_mutex);
	early_data_handle->req_ready = 1;
	STARPU_PTHREAD_COND_BROADCAST(&early_data_handle->req_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&early_data_handle->req_mutex);
	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);

	// Handle the request immediatly to make sure the mpi_irecv is
	// posted before receiving an other envelope
	_starpu_mpi_req_list_erase(&ready_recv_requests, early_data_handle->req);
	_STARPU_MPI_INC_READY_REQUESTS(-1);
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
	_starpu_mpi_handle_ready_request(early_data_handle->req);
	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
}

static void *_starpu_mpi_progress_thread_func(void *arg)
{
	struct _starpu_mpi_argc_argv *argc_argv = (struct _starpu_mpi_argc_argv *) arg;

	starpu_pthread_setname("MPI");

	_starpu_mpi_env_init();

#ifndef STARPU_SIMGRID
	if (_starpu_mpi_thread_cpuid < 0)
	{
		_starpu_mpi_thread_cpuid = starpu_get_next_bindid(STARPU_THREAD_ACTIVE, NULL, 0);
	}

	if (!_starpu_mpi_nobind && starpu_bind_thread_on(_starpu_mpi_thread_cpuid, STARPU_THREAD_ACTIVE, "MPI") < 0)
	{
		_STARPU_DISP("No core was available for the MPI thread. You should use STARPU_RESERVE_NCPU to leave one core available for MPI, or specify one core less in STARPU_NCPU\n");
	}
	_starpu_mpi_do_initialize(argc_argv);
	if (!_starpu_mpi_nobind && _starpu_mpi_thread_cpuid >= 0)
		/* In case MPI changed the binding */
		starpu_bind_thread_on(_starpu_mpi_thread_cpuid, STARPU_THREAD_ACTIVE, "MPI");
#else
	/* Now that MPI is set up, let the rest of simgrid get initialized */
	char **argv_cpy;
	_STARPU_MPI_MALLOC(argv_cpy, *(argc_argv->argc) * sizeof(char*));
	int i;
	for (i = 0; i < *(argc_argv->argc); i++)
		argv_cpy[i] = strdup((*(argc_argv->argv))[i]);
	void **tsd;
	_STARPU_CALLOC(tsd, MAX_TSD + 1, sizeof(void*));
#ifdef HAVE_SG_ACTOR_DATA
	_starpu_simgrid_actor_create("main", smpi_simulated_main_, _starpu_simgrid_get_host_by_name("MAIN"), *(argc_argv->argc), argv_cpy);
	/* And set TSD for us */
	sg_actor_data_set(sg_actor_self(), tsd);
#else
	MSG_process_create_with_arguments("main", smpi_simulated_main_, NULL, _starpu_simgrid_get_host_by_name("MAIN"), *(argc_argv->argc), argv_cpy);
	/* And set TSD for us */
	if (!smpi_process_set_user_data)
	{
		_STARPU_ERROR("Your version of simgrid does not provide smpi_process_set_user_data, we can not continue without it\n");
	}
	smpi_process_set_user_data(tsd);
#endif
        /* And wait for StarPU to get initialized, to come back to the same
         * situation as native execution where that's always the case. */
	starpu_wait_initialized();
#endif

	_starpu_mpi_comm_amounts_init(argc_argv->comm);
	_starpu_mpi_cache_init(argc_argv->comm);
	_starpu_mpi_select_node_init();
	_starpu_mpi_tag_init();
	_starpu_mpi_comm_init(argc_argv->comm);

	_starpu_mpi_early_request_init();
	_starpu_mpi_early_data_init();
	_starpu_mpi_sync_data_init();
	_starpu_mpi_datatype_init();

	if (mpi_driver)
		starpu_driver_init(mpi_driver);

#ifdef STARPU_SIMGRID
	starpu_pthread_wait_init(&_starpu_mpi_thread_wait);
	starpu_pthread_queue_init(&_starpu_mpi_thread_dontsleep);
	starpu_pthread_queue_register(&_starpu_mpi_thread_wait, &_starpu_mpi_thread_dontsleep);
#endif

#ifdef STARPU_USE_FXT
	if (_starpu_fxt_wait_initialisation())
	{
		/* We need to record our ID in the trace before the main thread makes any MPI call */
		_STARPU_MPI_TRACE_START(argc_argv->rank, argc_argv->world_size);
		starpu_profiling_set_id(argc_argv->rank);
		_starpu_mpi_add_sync_point_in_fxt();
	}
#endif //STARPU_USE_FXT

	/* notify the main thread that the progression thread is ready */
	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
	running = 1;
	STARPU_PTHREAD_COND_SIGNAL(&progress_cond);

 	int envelope_request_submitted = 0;
	int mpi_driver_loop_counter = 0;
	int mpi_driver_task_counter = 0;
	_STARPU_MPI_TRACE_POLLING_BEGIN();

	while (running || posted_requests || !(_starpu_mpi_req_list_empty(&ready_recv_requests)) || !(_starpu_mpi_req_prio_list_empty(&ready_send_requests)) || !(_starpu_mpi_req_list_empty(&detached_requests)))// || !(_starpu_mpi_early_request_count()) || !(_starpu_mpi_sync_data_count()))
	{
#ifdef STARPU_SIMGRID
		starpu_pthread_wait_reset(&_starpu_mpi_thread_wait);
#endif
		/* shall we block ? */
		unsigned block = _starpu_mpi_req_list_empty(&ready_recv_requests) && _starpu_mpi_req_prio_list_empty(&ready_send_requests) && _starpu_mpi_early_request_count() == 0 && _starpu_mpi_sync_data_count() == 0 && _starpu_mpi_req_list_empty(&detached_requests);

		if (block)
		{
			_STARPU_MPI_DEBUG(3, "NO MORE REQUESTS TO HANDLE\n");
			_STARPU_MPI_TRACE_SLEEP_BEGIN();

			if (barrier_running)
				/* Tell mpi_barrier */
				STARPU_PTHREAD_COND_SIGNAL(&barrier_cond);
			STARPU_PTHREAD_COND_WAIT(&progress_cond, &progress_mutex);

			_STARPU_MPI_TRACE_SLEEP_END();
		}

		/* get one recv request */
		unsigned n = 0;
		while (!_starpu_mpi_req_list_empty(&ready_recv_requests))
		{
			_STARPU_MPI_TRACE_POLLING_END();
			struct _starpu_mpi_req *req;

			if (n++ == nready_process)
				/* Already spent some time on submitting ready recv requests, poll before processing more ready recv requests */
				break;

			req = _starpu_mpi_req_list_pop_back(&ready_recv_requests);
			_STARPU_MPI_INC_READY_REQUESTS(-1);

			/* handling a request is likely to block for a while
			 * (on a sync_data_with_mem call), we want to let the
			 * application submit requests in the meantime, so we
			 * release the lock. */
			STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
			_starpu_mpi_handle_ready_request(req);
			STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
		}

		/* get one send request */
		n = 0;
		while (!_starpu_mpi_req_prio_list_empty(&ready_send_requests) && detached_send_nrequests < ndetached_send)
		{
			struct _starpu_mpi_req *req;

			if (n++ == nready_process)
				/* Already spent some time on submitting ready send requests, poll before processing more ready send requests */
				break;

			req = _starpu_mpi_req_prio_list_pop_back_highest(&ready_send_requests);
			_STARPU_MPI_INC_READY_REQUESTS(-1);

			/* handling a request is likely to block for a while
			 * (on a sync_data_with_mem call), we want to let the
			 * application submit requests in the meantime, so we
			 * release the lock. */
			STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
			_starpu_mpi_handle_ready_request(req);
			STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
		}

		_STARPU_MPI_TRACE_POLLING_BEGIN();

		/* If there is no currently submitted envelope_request submitted to
                 * catch envelopes from senders, and there is some pending
                 * receive requests on our side, we resubmit a header request. */
		if (((_starpu_mpi_early_request_count() > 0) || (_starpu_mpi_sync_data_count() > 0)) && (envelope_request_submitted == 0))// && (HASH_COUNT(_starpu_mpi_early_data_handle_hashmap) == 0))
		{
			_starpu_mpi_comm_post_recv();
			envelope_request_submitted = 1;
		}

		/* test whether there are some terminated "detached request" */
		STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
		_starpu_mpi_test_detached_requests();
		STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);

		if (envelope_request_submitted == 1)
		{
			int flag;
			struct _starpu_mpi_envelope *envelope;
			MPI_Status envelope_status;
			MPI_Comm envelope_comm;

			/* test whether an envelope has arrived. */
			flag = _starpu_mpi_comm_test_recv(&envelope_status, &envelope, &envelope_comm);

			if (flag)
			{
				_STARPU_MPI_TRACE_POLLING_END();
				_STARPU_MPI_COMM_FROM_DEBUG(envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, envelope_status.MPI_SOURCE, _STARPU_MPI_TAG_ENVELOPE, envelope->data_tag, envelope_comm);
				_STARPU_MPI_DEBUG(4, "Envelope received with mode %d\n", envelope->mode);
				if (envelope->mode == _STARPU_MPI_ENVELOPE_SYNC_READY)
				{
					struct _starpu_mpi_req *_sync_req = _starpu_mpi_sync_data_find(envelope->data_tag, envelope_status.MPI_SOURCE, envelope_comm);
					_STARPU_MPI_DEBUG(20, "Sending data with tag %"PRIi64" to node %d\n", _sync_req->node_tag.data_tag, envelope_status.MPI_SOURCE);
					STARPU_MPI_ASSERT_MSG(envelope->data_tag == _sync_req->node_tag.data_tag, "Tag mismatch (envelope %"PRIi64" != req %"PRIi64")\n",
							      envelope->data_tag, _sync_req->node_tag.data_tag);
					STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
					_starpu_mpi_isend_data_func(_sync_req);
					STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
				}
				else
				{
					_STARPU_MPI_DEBUG(3, "Searching for application request with tag %"PRIi64" and source %d (size %ld)\n", envelope->data_tag, envelope_status.MPI_SOURCE, envelope->size);

					struct _starpu_mpi_req *early_request = _starpu_mpi_early_request_dequeue(envelope->data_tag, envelope_status.MPI_SOURCE, envelope_comm);

					/* Case: a data will arrive before a matching receive is
					 * posted by the application. Create a temporary handle to
					 * store the incoming data, submit a starpu_mpi_irecv_detached
					 * on this handle, and store it as an early_data
					 */
					if (early_request == NULL)
					{
						if (envelope->sync)
						{
							_STARPU_MPI_DEBUG(2000, "-------------------------> adding request for tag %"PRIi64"\n", envelope->data_tag);
							struct _starpu_mpi_req *new_req;
#ifdef STARPU_DEVEL
#warning creating a request is not really useful.
#endif
							/* Initialize the request structure */
							_starpu_mpi_request_init(&new_req);
							new_req->request_type = RECV_REQ;
							new_req->data_handle = NULL;
							new_req->node_tag.node.rank = envelope_status.MPI_SOURCE;
							new_req->node_tag.data_tag = envelope->data_tag;
							new_req->node_tag.node.comm = envelope_comm;
							new_req->detached = 1;
							new_req->sync = 1;
							new_req->callback = NULL;
							new_req->callback_arg = NULL;
							new_req->func = _starpu_mpi_irecv_size_func;
							new_req->sequential_consistency = 1;
							new_req->backend->is_internal_req = 0; // ????
							new_req->count = envelope->size;
							_starpu_mpi_sync_data_add(new_req);
						}
						else
						{
							_starpu_mpi_receive_early_data(envelope, envelope_status, envelope_comm);
						}
					}
					/* Case: a matching application request has been found for
					 * the incoming data, we handle the correct allocation
					 * of the pointer associated to the data handle, then
					 * submit the corresponding receive with
					 * _starpu_mpi_handle_ready_request. */
					else
					{
						_STARPU_MPI_DEBUG(2000, "A matching application request has been found for the incoming data with tag %"PRIi64"\n", envelope->data_tag);
						_STARPU_MPI_DEBUG(2000, "Request sync %d\n", envelope->sync);

						early_request->sync = envelope->sync;
						_starpu_mpi_datatype_allocate(early_request->data_handle, early_request);
						if (early_request->registered_datatype == 1)
						{
							early_request->count = 1;
							early_request->ptr = starpu_data_handle_to_pointer(early_request->data_handle, STARPU_MAIN_RAM);
						}
						else
						{
							early_request->count = envelope->size;
							early_request->ptr = (void *)starpu_malloc_on_node_flags(STARPU_MAIN_RAM, early_request->count, 0);
							starpu_memory_allocate(STARPU_MAIN_RAM, early_request->count, STARPU_MEMORY_OVERFLOW);

							STARPU_MPI_ASSERT_MSG(early_request->ptr, "cannot allocate message of size %ld\n", early_request->count);
						}

						_STARPU_MPI_DEBUG(3, "Handling new request... \n");
						/* handling a request is likely to block for a while
						 * (on a sync_data_with_mem call), we want to let the
						 * application submit requests in the meantime, so we
						 * release the lock. */
						STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
						_starpu_mpi_handle_ready_request(early_request);
						STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
					}
				}
				envelope_request_submitted = 0;
				_STARPU_MPI_TRACE_POLLING_BEGIN();
			}
			else
			{
				/* A call is made to driver_run_once only when
				 * the progression thread have gone through the
				 * communication progression loop
				 * mpi_driver_call_freq times. It is
				 * interesting to tune the
				 * STARPU_MPI_DRIVER_CALL_FREQUENCY
				 * depending on whether the user wants
				 * reactivity or computing power from the MPI
				 * progression thread. */
				if ( mpi_driver && ( ++mpi_driver_loop_counter == mpi_driver_call_freq ))
				{
					mpi_driver_loop_counter = 0;
					mpi_driver_task_counter = 0;
					while (mpi_driver_task_counter++ < mpi_driver_task_freq)
					{
						_STARPU_MPI_TRACE_DRIVER_RUN_BEGIN();
						STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
						_STARPU_MPI_DEBUG(4, "running once mpi driver\n");
						starpu_driver_run_once(mpi_driver);
						STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
						_STARPU_MPI_TRACE_DRIVER_RUN_END();
					}
				}

				//_STARPU_MPI_DEBUG(4, "Nothing received, continue ..\n");
			}
		}
#ifdef STARPU_SIMGRID
		STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
		starpu_pthread_wait_wait(&_starpu_mpi_thread_wait);
		STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
#endif
	}

	_STARPU_MPI_TRACE_POLLING_END();
	if (envelope_request_submitted)
	{
		_starpu_mpi_comm_cancel_recv();
		envelope_request_submitted = 0;
	}


#ifdef STARPU_SIMGRID
	STARPU_PTHREAD_MUTEX_LOCK(&wait_counter_mutex);
	while (wait_counter != 0)
		STARPU_PTHREAD_COND_WAIT(&wait_counter_cond, &wait_counter_mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&wait_counter_mutex);

	STARPU_PTHREAD_MUTEX_DESTROY(&wait_counter_mutex);
	STARPU_PTHREAD_COND_DESTROY(&wait_counter_cond);

	starpu_pthread_queue_unregister(&_starpu_mpi_thread_wait, &_starpu_mpi_thread_dontsleep);
	starpu_pthread_queue_destroy(&_starpu_mpi_thread_dontsleep);
	starpu_pthread_wait_destroy(&_starpu_mpi_thread_wait);
#endif

	STARPU_MPI_ASSERT_MSG(_starpu_mpi_req_list_empty(&detached_requests), "List of detached requests not empty");
	STARPU_MPI_ASSERT_MSG(detached_send_nrequests == 0, "Number of detached send requests not 0");
	STARPU_MPI_ASSERT_MSG(_starpu_mpi_req_list_empty(&ready_recv_requests), "List of ready requests not empty");
	STARPU_MPI_ASSERT_MSG(_starpu_mpi_req_prio_list_empty(&ready_send_requests), "List of ready requests not empty");
	STARPU_MPI_ASSERT_MSG(posted_requests == 0, "Number of posted request is not zero");
	_starpu_mpi_early_request_check_termination();
	_starpu_mpi_early_data_check_termination();
	_starpu_mpi_sync_data_check_termination();

	if (argc_argv->initialize_mpi)
	{
		_STARPU_MPI_DEBUG(0, "Calling MPI_Finalize()\n");
		MPI_Finalize();
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

	_starpu_mpi_sync_data_shutdown();
	_starpu_mpi_early_data_shutdown();
	_starpu_mpi_early_request_shutdown();
	_starpu_mpi_datatype_shutdown();
	free(argc_argv);

	return NULL;
}

static void _starpu_mpi_add_sync_point_in_fxt(void)
{
#ifdef STARPU_USE_FXT
	int rank;
	int worldsize;
	int ret;

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &worldsize);

	ret = MPI_Barrier(MPI_COMM_WORLD);
	STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Barrier returning %s", _starpu_mpi_get_mpi_error_code(ret));

	/* We generate a "unique" key so that we can make sure that different
	 * FxT traces come from the same MPI run. */
	int random_number;

	/* XXX perhaps we don't want to generate a new seed if the application
	 * specified some reproductible behaviour ? */
	if (rank == 0)
	{
		srand(time(NULL));
		random_number = rand();
	}

	ret = MPI_Bcast(&random_number, 1, MPI_INT, 0, MPI_COMM_WORLD);
	STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Bcast returning %s", _starpu_mpi_get_mpi_error_code(ret));

	_STARPU_MPI_TRACE_BARRIER(rank, worldsize, random_number);

	_STARPU_MPI_DEBUG(3, "unique key %x\n", random_number);
#endif
}

int _starpu_mpi_progress_init(struct _starpu_mpi_argc_argv *argc_argv)
{
        STARPU_PTHREAD_MUTEX_INIT(&progress_mutex, NULL);
        STARPU_PTHREAD_COND_INIT(&progress_cond, NULL);
        STARPU_PTHREAD_COND_INIT(&barrier_cond, NULL);
	_starpu_mpi_req_list_init(&ready_recv_requests);
	_starpu_mpi_req_prio_list_init(&ready_send_requests);

        STARPU_PTHREAD_MUTEX_INIT(&detached_requests_mutex, NULL);
	_starpu_mpi_req_list_init(&detached_requests);

        STARPU_PTHREAD_MUTEX_INIT(&mutex_posted_requests, NULL);
        STARPU_PTHREAD_MUTEX_INIT(&mutex_ready_requests, NULL);

	nready_process = starpu_get_env_number_default("STARPU_MPI_NREADY_PROCESS", 10);
	ndetached_send = starpu_get_env_number_default("STARPU_MPI_NDETACHED_SEND", 10);

#ifdef STARPU_SIMGRID
	STARPU_PTHREAD_MUTEX_INIT(&wait_counter_mutex, NULL);
	STARPU_PTHREAD_COND_INIT(&wait_counter_cond, NULL);
#endif

#ifdef STARPU_SIMGRID
        _starpu_mpi_progress_thread_func(argc_argv);
        return 0;
#else
        STARPU_PTHREAD_CREATE(&progress_thread, NULL, _starpu_mpi_progress_thread_func, argc_argv);

        STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
        while (!running)
                STARPU_PTHREAD_COND_WAIT(&progress_cond, &progress_mutex);
        STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

        return 0;
#endif
}

#ifdef STARPU_SIMGRID
void _starpu_mpi_wait_for_initialization()
{
	/* Wait for MPI initialization to finish */
	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
	while (!running)
		STARPU_PTHREAD_COND_WAIT(&progress_cond, &progress_mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
}
#endif

void _starpu_mpi_progress_shutdown(void **value)
{
        STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
        running = 0;
        STARPU_PTHREAD_COND_BROADCAST(&progress_cond);

#ifdef STARPU_SIMGRID
	starpu_pthread_queue_signal(&_starpu_mpi_thread_dontsleep);
#endif
        STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

#ifdef STARPU_SIMGRID
	/* FIXME: should rather properly wait for _starpu_mpi_progress_thread_func to finish */
	(void) value;
	starpu_sleep(1);
#else
	STARPU_PTHREAD_JOIN(progress_thread, value);
#endif

        STARPU_PTHREAD_MUTEX_DESTROY(&mutex_posted_requests);
        STARPU_PTHREAD_MUTEX_DESTROY(&mutex_ready_requests);
        STARPU_PTHREAD_MUTEX_DESTROY(&progress_mutex);
        STARPU_PTHREAD_COND_DESTROY(&barrier_cond);
}

static int64_t _starpu_mpi_tag_max = INT64_MAX;

int starpu_mpi_comm_get_attr(MPI_Comm comm, int keyval, void *attribute_val, int *flag)
{
	(void) comm;
	if (keyval == STARPU_MPI_TAG_UB)
	{
		*flag = 1;
		*(int64_t **)attribute_val = &_starpu_mpi_tag_max;
	}
	else
	{
		*flag = 0;
	}
	return 0;
}

void _starpu_mpi_driver_init(struct starpu_conf *conf)
{
	/* We only initialize the driver if the environment variable
	 * STARPU_MPI_DRIVER_CALL_FREQUENCY is defined by the user. If this environment
	 * variable is not defined or defined at a value lower than or equal to zero,
	 * StarPU-MPI will not use a driver. */
	int driver_env = starpu_get_env_number_default("STARPU_MPI_DRIVER_CALL_FREQUENCY", 0);
	if (driver_env > 0)
	{
#ifdef STARPU_SIMGRID
		_STARPU_DISP("Warning: MPI driver is not supported with simgrid, this will be disabled");
		return;
#endif
		mpi_driver_call_freq = driver_env;

		_STARPU_MALLOC(mpi_driver, sizeof(struct starpu_driver));
		mpi_driver->type = STARPU_CPU_WORKER;
		mpi_driver->id.cpu_id = 0;

		conf->not_launched_drivers = mpi_driver;
		conf->n_not_launched_drivers = 1;

		int tasks_freq_env = starpu_get_env_number_default("STARPU_MPI_DRIVER_TASK_FREQUENCY", 0);
		if (tasks_freq_env > 0)
			mpi_driver_task_freq = tasks_freq_env;
	}
}

void _starpu_mpi_driver_shutdown()
{
	if (mpi_driver)
	{
		starpu_driver_deinit(mpi_driver);
		free(mpi_driver);
		mpi_driver = NULL;
	}
}

#endif /* STARPU_USE_MPI_MPI */
