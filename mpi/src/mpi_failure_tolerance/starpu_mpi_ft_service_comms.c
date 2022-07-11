/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdarg.h>
#include <stdlib.h>
#include <common/utils.h>

#include <mpi_failure_tolerance/starpu_mpi_checkpoint.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint_template.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint_package.h>
#include <sys/param.h>
#include <starpu_mpi_private.h>
#include <mpi/starpu_mpi_mpi_backend.h> // Should be deduced at preprocessing (Nmad vs MPI)
#include <mpi/starpu_mpi_mpi.h>
#include "starpu_mpi_cache.h"

#define SIMULTANEOUS_ACK_MSG_RECV_MAX 2
#define SIMULTANEOUS_CP_INFO_RECV_MAX 2
#define SIMULTANEOUS_PENDING_SEND_MAX 40

static struct _starpu_mpi_req_list detached_ft_service_requests;
static struct _starpu_mpi_req_list ready_send_ft_service_requests;
static unsigned detached_send_n_ft_service_requests;
static starpu_pthread_mutex_t detached_ft_service_requests_mutex;
static starpu_pthread_mutex_t ft_service_requests_mutex;

int ready_ack_msgs_recv;
int pending_ack_msgs_recv;
int ready_cp_info_msgs_recv;
int pending_cp_info_msgs_recv;
int ready_send_ft_service_msg;
int pending_send_ft_service_msg;

typedef void (*cb_fn_type)(void*);
cb_fn_type ack_msg_recv_cb;
cb_fn_type cp_info_recv_cb;

int _starpu_mpi_ft_service_submit_rdy()
{
	int i;
	struct _starpu_mpi_req* req;
	int max_loop;

	STARPU_PTHREAD_MUTEX_LOCK(&ft_service_requests_mutex);
	max_loop = MIN(SIMULTANEOUS_ACK_MSG_RECV_MAX-pending_ack_msgs_recv, ready_ack_msgs_recv);
	for (i=0 ; i<max_loop ; i++)
	{
		struct _starpu_mpi_cp_ack_arg_cb* arg;
		_STARPU_MALLOC(arg, sizeof(struct _starpu_mpi_cp_ack_arg_cb));
		req = _starpu_mpi_request_fill(NULL, MPI_ANY_SOURCE, _STARPU_MPI_TAG_CP_ACK, MPI_COMM_WORLD,
					       1, 0, 0, ack_msg_recv_cb, arg, RECV_REQ, NULL,
					       1, 0, sizeof(arg->msg));
		req->ptr = (void*)&arg->msg;
		req->datatype = MPI_BYTE;
		_STARPU_MALLOC(req->status, sizeof(MPI_Status));

		STARPU_PTHREAD_MUTEX_LOCK(&detached_ft_service_requests_mutex);
		MPI_Irecv(req->ptr, req->count, req->datatype, req->node_tag.node.rank, req->node_tag.data_tag,
		          req->node_tag.node.comm, &req->backend->data_request);
		_STARPU_MPI_DEBUG(5, "Posting MPI_Irecv ft service msg: req %p tag %"PRIi64" src %d comm %ld ptr %p\n", req,  req->node_tag.data_tag, req->node_tag.node.rank, (long int)req->node_tag.node.comm, req->ptr);
		_starpu_mpi_req_list_push_back(&detached_ft_service_requests, req);
		pending_ack_msgs_recv++;
		ready_ack_msgs_recv--;
		req->submitted = 1;
		STARPU_PTHREAD_MUTEX_UNLOCK(&detached_ft_service_requests_mutex);
	}

	max_loop = MIN(SIMULTANEOUS_CP_INFO_RECV_MAX-pending_cp_info_msgs_recv, ready_cp_info_msgs_recv);
	for (i=0 ; i<max_loop ; i++)
	{
		struct _starpu_mpi_cp_discard_arg_cb* arg;
		_STARPU_MALLOC(arg, sizeof(struct _starpu_mpi_cp_discard_arg_cb));
		req = _starpu_mpi_request_fill(NULL, MPI_ANY_SOURCE, _STARPU_MPI_TAG_CP_INFO, MPI_COMM_WORLD,
		                         1, 0, 0, cp_info_recv_cb, arg, RECV_REQ, NULL,
		                         1, 0, sizeof(arg->msg));
		req->ptr = (void*)&arg->msg;
		req->datatype = MPI_BYTE;
		_STARPU_MALLOC(req->status, sizeof(MPI_Status));

		STARPU_PTHREAD_MUTEX_LOCK(&detached_ft_service_requests_mutex);
		MPI_Irecv(req->ptr, req->count, req->datatype, req->node_tag.node.rank, req->node_tag.data_tag,
		          req->node_tag.node.comm, &req->backend->data_request);
		_STARPU_MPI_DEBUG(5, "Posting MPI_Irecv ft service msg: req %p tag %"PRIi64" src %d comm %ld ptr %p\n", req,  req->node_tag.data_tag, req->node_tag.node.rank, (long int)req->node_tag.node.comm, req->ptr);
		_starpu_mpi_req_list_push_back(&detached_ft_service_requests, req);
		pending_cp_info_msgs_recv++;
		ready_cp_info_msgs_recv--;
		req->submitted = 1;
		STARPU_PTHREAD_MUTEX_UNLOCK(&detached_ft_service_requests_mutex);
	}

	max_loop = MIN(SIMULTANEOUS_PENDING_SEND_MAX-pending_send_ft_service_msg, ready_send_ft_service_msg);
	for (i=0 ; i<max_loop ; i++)
	{
		req = _starpu_mpi_req_list_pop_front(&ready_send_ft_service_requests);
		STARPU_PTHREAD_MUTEX_LOCK(&detached_ft_service_requests_mutex);
		MPI_Isend(req->ptr, req->count, req->datatype, req->node_tag.node.rank, req->node_tag.data_tag,
		          req->node_tag.node.comm, &req->backend->data_request);

		_STARPU_MPI_DEBUG(5, "Posting MPI_Isend ft service msg: req %p tag %"PRIi64" src %d comm %ld ptr %p\n", req,  req->node_tag.data_tag, req->node_tag.node.rank, (long int)req->node_tag.node.comm, req->ptr);
		_starpu_mpi_req_list_push_back(&detached_ft_service_requests, req);
		pending_send_ft_service_msg++;
		ready_send_ft_service_msg--;
		req->submitted = 1;
		STARPU_PTHREAD_MUTEX_UNLOCK(&detached_ft_service_requests_mutex);
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&ft_service_requests_mutex);
}

int _starpu_mpi_ft_service_post_special_recv(int tag)
{
	_STARPU_MPI_DEBUG(5, "Pushing ft service msg: %s tag %"PRIi64" ANYSOURCE\n", _starpu_mpi_request_type(RECV_REQ), tag);

	if (tag==_STARPU_MPI_TAG_CP_ACK)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&ft_service_requests_mutex);
		ready_ack_msgs_recv++;
		STARPU_PTHREAD_MUTEX_UNLOCK(&ft_service_requests_mutex);
	}
	else if (tag==_STARPU_MPI_TAG_CP_INFO)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&ft_service_requests_mutex);
		ready_cp_info_msgs_recv++;
		STARPU_PTHREAD_MUTEX_UNLOCK(&ft_service_requests_mutex);
	}
	else
	{
		STARPU_ABORT_MSG("Only _STARPU_MPI_TAG_CP_ACK or _STARPU_MPI_TAG_CP_INFO are service msgs.\n");
	}
	_starpu_mpi_wake_up_progress_thread();
	return 0;
}

int _starpu_mpi_ft_service_post_send(void* msg, int count, int rank, int tag, MPI_Comm comm, void (*callback)(void *), void* arg)
{
	struct _starpu_mpi_req* req;

	/* Check if the tag is a service message */
	STARPU_ASSERT_MSG(tag==_STARPU_MPI_TAG_CP_ACK || tag == _STARPU_MPI_TAG_CP_INFO, "Only _STARPU_MPI_TAG_CP_ACK or _STARPU_MPI_TAG_CP_INFO are service msgs.");

	/* Initialize the request structure */
	req = _starpu_mpi_request_fill(NULL, rank, tag, comm, 1, 0, 0, callback, arg, SEND_REQ, NULL, 1, 0, count);
//	TODO: Check compatibility with prio
	req->ptr = msg;
	req->datatype = MPI_BYTE;
	_STARPU_MALLOC(req->status, sizeof(MPI_Status));

	_STARPU_MPI_DEBUG(5, "Pushing ft service msg: %s req %p tag %"PRIi64" src %d ptr %p\n", _starpu_mpi_request_type(SEND_REQ), req, tag, rank, msg);

	STARPU_PTHREAD_MUTEX_LOCK(&ft_service_requests_mutex);
	ready_send_ft_service_msg++;
	_starpu_mpi_req_list_push_back(&ready_send_ft_service_requests, req);
	STARPU_PTHREAD_MUTEX_UNLOCK(&ft_service_requests_mutex);

	_starpu_mpi_wake_up_progress_thread();

	return 0;
}

static void _starpu_mpi_handle_ft_request_termination(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();
	_STARPU_MPI_DEBUG(2,
	                  "complete MPI request %p type %s tag %"PRIi64" src %d data %p ptr %p datatype '%s' count %d registered_datatype %d internal_req %p\n",
	                  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank,
	                  req->data_handle, req->ptr,
	                  req->datatype_name, (int) req->count, req->registered_datatype, req->backend->internal_req);

	if (req->backend->internal_req)
	{
//		free(req->backend->early_data_handle);
//		req->backend->early_data_handle = NULL;
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
					starpu_free_on_node_flags(STARPU_MAIN_RAM, (uintptr_t) req->ptr, req->count, 0);
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
				//_starpu_mpi_datatype_free(req->data_handle, &req->datatype);
			}
		}
		//_STARPU_MPI_TRACE_TERMINATED(req, req->node_tag.node.rank, req->node_tag.data_tag);
	}

	_starpu_mpi_release_req_data(req);

	if (req->backend->envelope)
	{
		free(req->backend->envelope);
		req->backend->envelope = NULL;
	}

	/* Execute the specified callback, if any */
	if (req->callback)
	{
		if (req->request_type == RECV_REQ)
		{
			if (req->node_tag.data_tag == _STARPU_MPI_TAG_CP_ACK)
			{
				struct _starpu_mpi_cp_ack_arg_cb* tmp = (struct _starpu_mpi_cp_ack_arg_cb *) req->callback_arg;
				tmp->rank = req->status->MPI_SOURCE;
			}
			else if (req->node_tag.data_tag == _STARPU_MPI_TAG_CP_INFO)
			{
				struct _starpu_mpi_cp_discard_arg_cb* tmp = (struct _starpu_mpi_cp_discard_arg_cb *) req->callback_arg;
				tmp->rank = req->status->MPI_SOURCE;
			}
		}
		req->callback(req->callback_arg);
	}
	/* tell anyone potentially waiting on the request that it is
	 * terminated now */
	STARPU_PTHREAD_MUTEX_LOCK(&req->backend->req_mutex);
	req->completed = 1;
	STARPU_PTHREAD_COND_BROADCAST(&req->backend->req_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&req->backend->req_mutex);
	_STARPU_MPI_LOG_OUT();
}

void starpu_mpi_test_ft_detached_service_requests(void)
{
	//_STARPU_MPI_LOG_IN();
	int flag;
	struct _starpu_mpi_req *req;

	STARPU_PTHREAD_MUTEX_LOCK(&detached_ft_service_requests_mutex);

	if (_starpu_mpi_req_list_empty(&detached_ft_service_requests))
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&detached_ft_service_requests_mutex);
		//_STARPU_MPI_LOG_OUT();
		return;
	}

	//_STARPU_MPI_TRACE_TESTING_DETACHED_BEGIN();
	req = _starpu_mpi_req_list_begin(&detached_ft_service_requests);
	while (req != _starpu_mpi_req_list_end(&detached_ft_service_requests))
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&detached_ft_service_requests_mutex);

		//_STARPU_MPI_TRACE_TEST_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag);
		//_STARPU_MPI_DEBUG(3, "Test detached request %p - mpitag %"PRIi64" - TYPE %s %d\n", &req->backend->data_request, req->node_tag.data_tag, _starpu_mpi_request_type(req->request_type), req->node_tag.node.rank);
#ifdef STARPU_SIMGRID
		req->ret = _starpu_mpi_simgrid_mpi_test(&req->done, &flag);
#else
		STARPU_MPI_ASSERT_MSG(req->backend->data_request != MPI_REQUEST_NULL, "Cannot test completion of the request MPI_REQUEST_NULL");
		req->ret = MPI_Test(&req->backend->data_request, &flag, req->status);
#endif

		STARPU_MPI_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Test returning %s", _starpu_mpi_get_mpi_error_code(req->ret));
		//_STARPU_MPI_TRACE_TEST_END(req->node_tag.node.rank, req->node_tag.data_tag);

		if (!flag)
		{
			req = _starpu_mpi_req_list_next(req);
		}
		else
		{
			//_STARPU_MPI_TRACE_POLLING_END();
			struct _starpu_mpi_req *next_req;
			next_req = _starpu_mpi_req_list_next(req);

			//_STARPU_MPI_TRACE_COMPLETE_BEGIN(req->request_type, req->node_tag.node.rank, req->node_tag.data_tag);

			STARPU_PTHREAD_MUTEX_LOCK(&detached_ft_service_requests_mutex);
			STARPU_PTHREAD_MUTEX_LOCK(&ft_service_requests_mutex);
			if (req->request_type == SEND_REQ)
				pending_send_ft_service_msg--;
			if (req->request_type == RECV_REQ)
			{
				if (req->node_tag.data_tag == _STARPU_MPI_TAG_CP_ACK)
					pending_ack_msgs_recv--;
				else if (req->node_tag.data_tag == _STARPU_MPI_TAG_CP_INFO)
					pending_cp_info_msgs_recv--;
			}
			STARPU_PTHREAD_MUTEX_UNLOCK(&ft_service_requests_mutex);
			_starpu_mpi_req_list_erase(&detached_ft_service_requests, req);
			STARPU_PTHREAD_MUTEX_UNLOCK(&detached_ft_service_requests_mutex);
			_starpu_mpi_handle_ft_request_termination(req);

			//_STARPU_MPI_TRACE_COMPLETE_END(req->request_type, req->node_tag.node.rank, req->node_tag.data_tag);

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
			//_STARPU_MPI_TRACE_POLLING_BEGIN();
		}

		STARPU_PTHREAD_MUTEX_LOCK(&detached_ft_service_requests_mutex);
	}
	//_STARPU_MPI_TRACE_TESTING_DETACHED_END();

	STARPU_PTHREAD_MUTEX_UNLOCK(&detached_ft_service_requests_mutex);
	//_STARPU_MPI_LOG_OUT();
}

int starpu_mpi_ft_service_progress()
{
	starpu_mpi_test_ft_detached_service_requests();
	_starpu_mpi_ft_service_submit_rdy();
	return 0;
}

int starpu_mpi_ft_service_lib_init(void(*_ack_msg_recv_cb)(void*), void(*_cp_info_recv_cb)(void*))
{
	_starpu_mpi_req_list_init(&detached_ft_service_requests);
	_starpu_mpi_req_list_init(&ready_send_ft_service_requests);
	STARPU_PTHREAD_MUTEX_INIT(&detached_ft_service_requests_mutex, NULL);
	STARPU_PTHREAD_MUTEX_INIT(&ft_service_requests_mutex, NULL);
	ready_ack_msgs_recv = 0;
	pending_ack_msgs_recv = 0;
	ready_cp_info_msgs_recv = 0;
	pending_cp_info_msgs_recv = 0;
	ready_send_ft_service_msg = 0;
	pending_send_ft_service_msg = 0;

	ack_msg_recv_cb = _ack_msg_recv_cb;
	cp_info_recv_cb = _cp_info_recv_cb;

	return 0;
}

int starpu_mpi_ft_service_lib_busy()
{
	return !_starpu_mpi_req_list_empty(&detached_ft_service_requests);
}
