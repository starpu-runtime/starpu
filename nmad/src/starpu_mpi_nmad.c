/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010-2014, 2017  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012, 2013, 2014, 2015  Centre National de la Recherche Scientifique
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
#include <starpu_mpi_select_node.h>
#include <starpu_mpi_tag.h>
#include <starpu_mpi_comm.h>
#include <starpu_mpi_init.h>
#include <common/config.h>
#include <common/thread.h>
#include <datawizard/coherency.h>
#include <nm_sendrecv_interface.h>
#include <nm_mpi_nmad.h>
#include <core/task.h>
#include <core/topology.h>

static void _starpu_mpi_handle_request_termination(struct _starpu_mpi_req *req,nm_sr_event_t event);
#ifdef STARPU_VERBOSE
static char *_starpu_mpi_request_type(enum _starpu_mpi_request_type request_type);
#endif
static struct _starpu_mpi_req *_starpu_mpi_isend_common(starpu_data_handle_t data_handle,
							int dest, int data_tag, MPI_Comm comm,
							unsigned detached, unsigned sync, int prio, void (*callback)(void *), void *arg);
static struct _starpu_mpi_req *_starpu_mpi_irecv_common(starpu_data_handle_t data_handle,
							int source, int data_tag, MPI_Comm comm,
							unsigned detached, unsigned sync, void (*callback)(void *), void *arg, int sequential_consistency);
static void _starpu_mpi_handle_new_request(void *arg);

static void _starpu_mpi_handle_pending_request(struct _starpu_mpi_req *req);
static void _starpu_mpi_add_sync_point_in_fxt(void);

static int mpi_thread_cpuid = -1;
int _starpu_mpi_fake_world_size = -1;
int _starpu_mpi_fake_world_rank = -1;

/* Condition to wake up waiting for all current MPI requests to finish */
static starpu_pthread_t progress_thread;
static starpu_pthread_cond_t progress_cond;
static starpu_pthread_mutex_t progress_mutex;
static volatile int running = 0;

/* Count requests posted by the application and not yet submitted to MPI, i.e pushed into the new_requests list */

static volatile int pending_request = 0;

#define REQ_FINALIZED 0x1

PUK_LFSTACK_TYPE(callback,	struct _starpu_mpi_req *req;);
static callback_lfstack_t callback_stack = NULL;

static starpu_sem_t callback_sem;

static void _starpu_mpi_request_init(struct _starpu_mpi_req **req)
{
	_STARPU_MPI_CALLOC(*req, 1, sizeof(struct _starpu_mpi_req));

	/* Initialize the request structure */
	(*req)->data_handle = NULL;
	(*req)->prio = 0;
	(*req)->completed = 0;

	(*req)->datatype = 0;
	(*req)->datatype_name = NULL;
	(*req)->ptr = NULL;
	(*req)->count = -1;
	(*req)->registered_datatype = -1;

	(*req)->node_tag.rank = -1;
	(*req)->node_tag.data_tag = -1;
	(*req)->node_tag.comm = 0;

	(*req)->func = NULL;

	(*req)->status = NULL;
	//	(*req)->data_request = 0;
	(*req)->flag = NULL;

	(*req)->ret = -1;
	piom_cond_init(&((*req)->req_cond), 0);
	//STARPU_PTHREAD_MUTEX_INIT(&((*req)->req_mutex), NULL);
	STARPU_PTHREAD_COND_INIT(&((*req)->req_cond), NULL);
	//	STARPU_PTHREAD_MUTEX_INIT(&((*req)->posted_mutex), NULL);
	//STARPU_PTHREAD_COND_INIT(&((*req)->posted_cond), NULL);

	(*req)->request_type = UNKNOWN_REQ;

	(*req)->submitted = 0;
	(*req)->completed = 0;
	(*req)->posted = 0;

	//(*req)->other_request = NULL;

	(*req)->sync = 0;
	(*req)->detached = -1;
	(*req)->callback = NULL;
	(*req)->callback_arg = NULL;

	//	(*req)->size_req = 0;
	//(*req)->internal_req = NULL;
	//(*req)->is_internal_req = 0;
	//(*req)->to_destroy = 1;
	//(*req)->early_data_handle = NULL;
	//(*req)->envelope = NULL;
	(*req)->sequential_consistency = 1;
	(*req)->pre_sync_jobid = -1;
	(*req)->post_sync_jobid = -1;

#ifdef STARPU_SIMGRID
	starpu_pthread_queue_init(&((*req)->queue));
	starpu_pthread_queue_register(&wait, &((*req)->queue));
	(*req)->done = 0;
#endif
}

static void _starpu_mpi_request_destroy(struct _starpu_mpi_req *req)
{
	piom_cond_destroy(&(req->req_cond));
	free(req);
}

/********************************************************/
/*                                                      */
/*  Send/Receive functionalities                        */
/*                                                      */
/********************************************************/

static void nop_acquire_cb(void *arg)
{
	starpu_data_release(arg);
}

static struct _starpu_mpi_req *_starpu_mpi_isend_irecv_common(starpu_data_handle_t data_handle,
							      int srcdst, int data_tag, MPI_Comm comm,
							      unsigned detached, unsigned sync, int prio, void (*callback)(void *), void *arg,
							      enum _starpu_mpi_request_type request_type, void (*func)(struct _starpu_mpi_req *),
							      enum starpu_data_access_mode mode,
							      int sequential_consistency)
{

	struct _starpu_mpi_req *req;

	if (_starpu_mpi_fake_world_size != -1)
	{
		/* Don't actually do the communication */
		starpu_data_acquire_on_node_cb_sequential_consistency(data_handle, STARPU_MAIN_RAM, mode, nop_acquire_cb, data_handle, sequential_consistency);
		return NULL;
	}

	_STARPU_MPI_LOG_IN();
	STARPU_ATOMIC_ADD( &pending_request, 1);

	/* Initialize the request structure */
	_starpu_mpi_request_init(&req);
	req->request_type = request_type;
	/* prio_list is sorted by increasing values */
	req->prio = prio;
	req->data_handle = data_handle;
	req->node_tag.rank = srcdst;
	req->node_tag.data_tag = data_tag;
	req->node_tag.comm = comm;
	req->detached = detached;
	req->sync = sync;
	req->callback = callback;
	req->callback_arg = arg;
	req->func = func;
	req->sequential_consistency = sequential_consistency;
	nm_mpi_nmad_dest(&req->session, &req->gate, comm, req->node_tag.rank);

	/* Asynchronously request StarPU to fetch the data in main memory: when
	 * it is available in main memory, _starpu_mpi_submit_new_mpi_request(req) is called and
	 * the request is actually submitted */
	starpu_data_acquire_on_node_cb_sequential_consistency_sync_jobids(data_handle, STARPU_MAIN_RAM, mode, _starpu_mpi_handle_new_request, (void *)req, sequential_consistency, &req->pre_sync_jobid, &req->post_sync_jobid);

	_STARPU_MPI_LOG_OUT();
	return req;
}

/********************************************************/
/*                                                      */
/*  Send functionalities                                */
/*                                                      */
/********************************************************/

static void _starpu_mpi_isend_data_func(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	_STARPU_MPI_DEBUG(30, "post MPI isend request %p type %s tag %d src %d data %p datasize %ld ptr %p datatype '%s' count %d registered_datatype %d sync %d\n", req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.rank, req->data_handle, starpu_data_get_size(req->data_handle), req->ptr, req->datatype_name, (int)req->count, req->registered_datatype, req->sync);

	_starpu_mpi_comm_amounts_inc(req->node_tag.comm, req->node_tag.rank, req->datatype, req->count);

	_STARPU_MPI_TRACE_ISEND_SUBMIT_BEGIN(req->node_tag.rank, req->node_tag.data_tag, 0);

	struct nm_data_s data;
	nm_mpi_nmad_data(&data, (void*)req->ptr, req->datatype, req->count);
	nm_sr_send_init(req->session, &(req->request));
	nm_sr_send_pack_data(req->session, &(req->request), &data);
	nm_sr_send_set_priority(req->session, &req->request, req->prio);

	if (req->sync == 0)
	{
		req->ret = nm_sr_send_isend(req->session, &(req->request), req->gate, req->node_tag.data_tag);
		STARPU_ASSERT_MSG(req->ret == NM_ESUCCESS, "MPI_Isend returning %d", req->ret);
	}
	else
	{
		req->ret = nm_sr_send_issend(req->session, &(req->request), req->gate, req->node_tag.data_tag);
		STARPU_ASSERT_MSG(req->ret == NM_ESUCCESS, "MPI_Issend returning %d", req->ret);
	}

	_STARPU_MPI_TRACE_ISEND_SUBMIT_END(req->node_tag.rank, req->node_tag.data_tag, starpu_data_get_size(req->data_handle), req->pre_sync_jobid);

	_starpu_mpi_handle_pending_request(req);

	_STARPU_MPI_LOG_OUT();
}

static void _starpu_mpi_isend_size_func(struct _starpu_mpi_req *req)
{
	_starpu_mpi_datatype_allocate(req->data_handle, req);

	if (req->registered_datatype == 1)
	{
		req->waited = 1;
		req->count = 1;
		req->ptr = starpu_data_get_local_ptr(req->data_handle);
	}
	else
	{
		starpu_ssize_t psize = -1;
		int ret;
		req->waited =2;

		// Do not pack the data, just try to find out the size
		starpu_data_pack(req->data_handle, NULL, &psize);

		if (psize != -1)
		{
			// We already know the size of the data, let's send it to overlap with the packing of the data
			_STARPU_MPI_DEBUG(20, "Sending size %ld (%ld %s) to node %d (first call to pack)\n", psize, sizeof(req->count), "MPI_BYTE", req->node_tag.rank);
			req->count = psize;
			//ret = nm_sr_isend(nm_mpi_communicator_get_session(p_req->p_comm),nm_mpi_communicator_get_gate(p_comm,req->srcdst), req->mpi_tag,&req->count, sizeof(req->count), &req->size_req);
			ret = nm_sr_isend(req->session,req->gate, req->node_tag.data_tag,&req->count, sizeof(req->count), &req->size_req);

			//	ret = MPI_Isend(&req->count, sizeof(req->count), MPI_BYTE, req->srcdst, req->mpi_tag, req->comm, &req->size_req);
			STARPU_ASSERT_MSG(ret == NM_ESUCCESS, "when sending size, nm_sr_isend returning %d", ret);
		}

		// Pack the data
		starpu_data_pack(req->data_handle, &req->ptr, &req->count);
		if (psize == -1)
		{
			// We know the size now, let's send it
			_STARPU_MPI_DEBUG(1, "Sending size %ld (%ld %s) with tag %d to node %d (second call to pack)\n", req->count, sizeof(req->count), "MPI_BYTE", req->node_tag.data_tag, req->node_tag.rank);
			ret = nm_sr_isend(req->session,req->gate, req->node_tag.data_tag,&req->count, sizeof(req->count), &req->size_req);
			STARPU_ASSERT_MSG(ret == NM_ESUCCESS, "when sending size, nm_sr_isend returning %d", ret);
		}
		else
		{
			// We check the size returned with the 2 calls to pack is the same
			STARPU_ASSERT_MSG(req->count == psize, "Calls to pack_data returned different sizes %ld != %ld", req->count, psize);
		}

		// We can send the data now
	}
	_starpu_mpi_isend_data_func(req);
}

static struct _starpu_mpi_req *_starpu_mpi_isend_common(starpu_data_handle_t data_handle,
							int dest, int data_tag, MPI_Comm comm,
							 unsigned detached, unsigned sync, int prio, void (*callback)(void *), void *arg)
{
	return _starpu_mpi_isend_irecv_common(data_handle, dest, data_tag, comm, detached, sync, prio, callback, arg, SEND_REQ, _starpu_mpi_isend_size_func, STARPU_R,1);
}

int starpu_mpi_isend_prio(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int dest, int data_tag, int prio, MPI_Comm comm)
{
	_STARPU_MPI_LOG_IN();
	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_isend needs a valid starpu_mpi_req");

	struct _starpu_mpi_req *req;
	_STARPU_MPI_TRACE_ISEND_COMPLETE_BEGIN(dest, data_tag, 0);
	req = _starpu_mpi_isend_common(data_handle, dest, data_tag, comm, 0, 0, prio, NULL, NULL);
	_STARPU_MPI_TRACE_ISEND_COMPLETE_END(dest, data_tag, 0);

	STARPU_MPI_ASSERT_MSG(req, "Invalid return for _starpu_mpi_isend_common");
	*public_req = req;

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_isend(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int dest, int data_tag, MPI_Comm comm)
{
	return starpu_mpi_isend_prio(data_handle, public_req, dest, data_tag, 0, comm);
}

int starpu_mpi_isend_detached_prio(starpu_data_handle_t data_handle, int dest, int data_tag, int prio, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();
	_starpu_mpi_isend_common(data_handle, dest, data_tag, comm, 1, 0, prio, callback, arg);
	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_isend_detached(starpu_data_handle_t data_handle, int dest, int data_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	return starpu_mpi_isend_detached_prio(data_handle, dest, data_tag, 0, comm, callback, arg);
}

int starpu_mpi_send_prio(starpu_data_handle_t data_handle, int dest, int data_tag, int prio, MPI_Comm comm)
{
	starpu_mpi_req req;
	MPI_Status status;

	_STARPU_MPI_LOG_IN();
	memset(&status, 0, sizeof(MPI_Status));

	starpu_mpi_isend_prio(data_handle, &req, dest, data_tag, prio, comm);
	starpu_mpi_wait(&req, &status);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_send(starpu_data_handle_t data_handle, int dest, int data_tag, MPI_Comm comm)
{
	return starpu_mpi_send_prio(data_handle, dest, data_tag, 0, comm);
}

int starpu_mpi_issend_prio(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int dest, int data_tag, int prio, MPI_Comm comm)
{
	_STARPU_MPI_LOG_IN();
	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_issend needs a valid starpu_mpi_req");

	struct _starpu_mpi_req *req;
	req = _starpu_mpi_isend_common(data_handle, dest, data_tag, comm, 0, 1, prio, NULL, NULL);

	STARPU_MPI_ASSERT_MSG(req, "Invalid return for _starpu_mpi_isend_common");
	*public_req = req;

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_issend(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int dest, int data_tag, MPI_Comm comm)
{
	return starpu_mpi_issend_prio(data_handle, public_req, dest, data_tag, 0, comm);
}

int starpu_mpi_issend_detached_prio(starpu_data_handle_t data_handle, int dest, int data_tag, int prio, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();

	_starpu_mpi_isend_common(data_handle, dest, data_tag, comm, 1, 1, prio, callback, arg);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_issend_detached(starpu_data_handle_t data_handle, int dest, int data_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	return starpu_mpi_issend_detached_prio(data_handle, dest, data_tag, 0, comm, callback, arg);
}

/********************************************************/
/*                                                      */
/*  Receive functionalities                             */
/*                                                      */
/********************************************************/

static void _starpu_mpi_irecv_data_func(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	_STARPU_MPI_DEBUG(20, "post MPI irecv request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n", req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.rank, req->data_handle, req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);

	_STARPU_MPI_TRACE_IRECV_SUBMIT_BEGIN(req->node_tag.rank, req->node_tag.data_tag);

	//req->ret = MPI_Irecv(req->ptr, req->count, req->datatype, req->srcdst, req->mpi_tag, req->comm, &req->request);
	struct nm_data_s data;
	nm_mpi_nmad_data(&data, (void*)req->ptr, req->datatype, req->count);
	nm_sr_recv_init(req->session, &(req->request));
	nm_sr_recv_unpack_data(req->session, &(req->request), &data);
	nm_sr_recv_irecv(req->session, &(req->request), req->gate, req->node_tag.data_tag, NM_TAG_MASK_FULL);

	_STARPU_MPI_TRACE_IRECV_SUBMIT_END(req->node_tag.rank, req->node_tag.data_tag);

	_starpu_mpi_handle_pending_request(req);

	_STARPU_MPI_LOG_OUT();
}

struct _starpu_mpi_irecv_size_callback
{
	starpu_data_handle_t handle;
	struct _starpu_mpi_req *req;
};

static void _starpu_mpi_irecv_size_callback(void *arg)
{
	struct _starpu_mpi_irecv_size_callback *callback = (struct _starpu_mpi_irecv_size_callback *)arg;

	starpu_data_unregister(callback->handle);
	callback->req->ptr = malloc(callback->req->count);
	STARPU_ASSERT_MSG(callback->req->ptr, "cannot allocate message of size %ld", callback->req->count);
	_starpu_mpi_irecv_data_func(callback->req);
	free(callback);
}

static void _starpu_mpi_irecv_size_func(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	_starpu_mpi_datatype_allocate(req->data_handle, req);
	if (req->registered_datatype == 1)
	{
		req->count = 1;
		req->ptr = starpu_data_get_local_ptr(req->data_handle);
		_starpu_mpi_irecv_data_func(req);
	}
	else
	{
		struct _starpu_mpi_irecv_size_callback *callback = malloc(sizeof(struct _starpu_mpi_irecv_size_callback));
		callback->req = req;
		starpu_variable_data_register(&callback->handle, 0, (uintptr_t)&(callback->req->count), sizeof(callback->req->count));
		_STARPU_MPI_DEBUG(4, "Receiving size with tag %d from node %d\n", req->node_tag.data_tag, req->node_tag.rank);
		_starpu_mpi_irecv_common(callback->handle, req->node_tag.rank, req->node_tag.data_tag, req->node_tag.comm, 1, 0, _starpu_mpi_irecv_size_callback, callback,1);
	}

}

static struct _starpu_mpi_req *_starpu_mpi_irecv_common(starpu_data_handle_t data_handle, int source, int mpi_tag, MPI_Comm comm, unsigned detached, unsigned sync, void (*callback)(void *), void *arg, int sequential_consistency)
{
	return _starpu_mpi_isend_irecv_common(data_handle, source, mpi_tag, comm, detached, sync, 0, callback, arg, RECV_REQ, _starpu_mpi_irecv_size_func, STARPU_W,sequential_consistency);
}

int starpu_mpi_irecv(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int source, int mpi_tag, MPI_Comm comm)
{
	_STARPU_MPI_LOG_IN();
	STARPU_ASSERT_MSG(public_req, "starpu_mpi_irecv needs a valid starpu_mpi_req");

	struct _starpu_mpi_req *req;
	_STARPU_MPI_TRACE_IRECV_COMPLETE_BEGIN(source, mpi_tag);
	req = _starpu_mpi_irecv_common(data_handle, source, mpi_tag, comm, 0, 0, NULL, NULL,1);
	_STARPU_MPI_TRACE_IRECV_COMPLETE_END(source, mpi_tag);

	STARPU_ASSERT_MSG(req, "Invalid return for _starpu_mpi_irecv_common");
	*public_req = req;

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_irecv_detached(starpu_data_handle_t data_handle, int source, int mpi_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();
	_starpu_mpi_irecv_common(data_handle, source, mpi_tag, comm, 1, 0, callback, arg,1);
	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_irecv_detached_sequential_consistency(starpu_data_handle_t data_handle, int source, int data_tag, MPI_Comm comm, void (*callback)(void *), void *arg, int sequential_consistency)
{
	_STARPU_MPI_LOG_IN();

//	// We check if a tag is defined for the data handle, if not,
//	// we define the one given for the communication.
//	// A tag is necessary for the internal mpi engine.
//	int tag = starpu_data_get_tag(data_handle);
//	if (tag == -1)
//		starpu_data_set_tag(data_handle, data_tag);

	_starpu_mpi_irecv_common(data_handle, source, data_tag, comm, 1, 0, callback, arg, sequential_consistency);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_recv(starpu_data_handle_t data_handle, int source, int mpi_tag, MPI_Comm comm, MPI_Status *status)
{
	starpu_mpi_req req;

	_STARPU_MPI_LOG_IN();
	starpu_mpi_irecv(data_handle, &req, source, mpi_tag, comm);
	starpu_mpi_wait(&req, status);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

/********************************************************/
/*                                                      */
/*  Wait functionalities                                */
/*                                                      */
/********************************************************/

int starpu_mpi_wait(starpu_mpi_req *public_req, MPI_Status *status)
{
	_STARPU_MPI_LOG_IN();
	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_wait needs a valid starpu_mpi_req");
	struct _starpu_mpi_req *req = *public_req;
	STARPU_MPI_ASSERT_MSG(!req->detached, "MPI_Wait cannot be called on a detached request");

	/* we must do a test_locked to avoid race condition :
	 * without req_cond could still be used and couldn't be freed)*/
	while (!req->completed || ! piom_cond_test_locked(&(req->req_cond),REQ_FINALIZED))
	{
		piom_cond_wait(&(req->req_cond),REQ_FINALIZED);
	}

	if (status!=MPI_STATUS_IGNORE)
		_starpu_mpi_req_status(req,status);

	_starpu_mpi_request_destroy(req);
	*public_req = NULL;
	_STARPU_MPI_LOG_OUT();
	return MPI_SUCCESS;
}

/********************************************************/
/*                                                      */
/*  Test functionalities                                */
/*                                                      */
/********************************************************/

int starpu_mpi_test(starpu_mpi_req *public_req, int *flag, MPI_Status *status)
{
	_STARPU_MPI_LOG_IN();
	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_test needs a valid starpu_mpi_req");
	struct _starpu_mpi_req *req = *public_req;
	STARPU_MPI_ASSERT_MSG(!req->detached, "MPI_Test cannot be called on a detached request");
	_STARPU_MPI_DEBUG(2, "Test request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n",
			  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.rank, req->data_handle, req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);

	_STARPU_MPI_TRACE_UTESTING_BEGIN(req->node_tag.rank, req->node_tag.data_tag);

	/* we must do a test_locked to avoid race condition :
	 * without req_cond could still be used and couldn't be freed)*/
	*flag = req->completed && piom_cond_test_locked(&(req->req_cond),REQ_FINALIZED);
	if (*flag && status!=MPI_STATUS_IGNORE)
		_starpu_mpi_req_status(req,status);

	_STARPU_MPI_TRACE_UTESTING_END(req->node_tag.rank, req->node_tag.data_tag);

	if(*flag)
	{
		_starpu_mpi_request_destroy(req);
		*public_req = NULL;
	}
	_STARPU_MPI_LOG_OUT();
	return MPI_SUCCESS;
}

/********************************************************/
/*                                                      */
/*  Barrier functionalities                             */
/*                                                      */
/********************************************************/

int starpu_mpi_barrier(MPI_Comm comm)
{
	_STARPU_MPI_LOG_IN();
	int ret;
	//	STARPU_ASSERT_MSG(!barrier_running, "Concurrent starpu_mpi_barrier is not implemented, even on different communicators");
	ret = MPI_Barrier(comm);

	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Barrier returning %d", ret);

	_STARPU_MPI_LOG_OUT();
	return ret;
}

/********************************************************/
/*                                                      */
/*  Progression                                         */
/*                                                      */
/********************************************************/

#ifdef STARPU_VERBOSE
static char *_starpu_mpi_request_type(enum _starpu_mpi_request_type request_type)
{
	switch (request_type)
	{
		case SEND_REQ: return "SEND_REQ";
		case RECV_REQ: return "RECV_REQ";
		case WAIT_REQ: return "WAIT_REQ";
		case TEST_REQ: return "TEST_REQ";
		case BARRIER_REQ: return "BARRIER_REQ";
		default: return "unknown request type";
	}
}
#endif

static void _starpu_mpi_handle_request_termination(struct _starpu_mpi_req *req,nm_sr_event_t event)
{
	_STARPU_MPI_LOG_IN();

	_STARPU_MPI_DEBUG(2, "complete MPI request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n",
			  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.rank, req->data_handle, req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);

	if (req->request_type == RECV_REQ || req->request_type == SEND_REQ)
	{
		if (req->registered_datatype == 0)
		{
			if (req->request_type == SEND_REQ)
			{
				req->waited--;
				// We need to make sure the communication for sending the size
				// has completed, as MPI can re-order messages, let's count
				// recerived message.
				// FIXME concurent access.
				STARPU_ASSERT_MSG(event == NM_SR_EVENT_FINALIZED, "Callback with event %d", event);
				if(req->waited>0)
					return;

			}
			if (req->request_type == RECV_REQ)
				// req->ptr is freed by starpu_data_unpack
				starpu_data_unpack(req->data_handle, req->ptr, req->count);
			else
				free(req->ptr);
		}
		else
		{
			_starpu_mpi_datatype_free(req->data_handle, &req->datatype);
		}
		starpu_data_release(req->data_handle);
	}

	/* Execute the specified callback, if any */
	if (req->callback)
	{
		struct callback_lfstack_cell_s* c = padico_malloc(sizeof(struct callback_lfstack_cell_s));
		c->req = req;
		/* The main thread can exit without waiting
		* the end of the detached request. Callback thread
		* must then be kept alive if they have a callback.*/

		callback_lfstack_push(&callback_stack, c);
		starpu_sem_post(&callback_sem);
	}
	else
	{
		if(req->detached)
		{
			_starpu_mpi_request_destroy(req);
			// a detached request wont be wait/test (and freed inside).
		}
		else
		{
			/* tell anyone potentially waiting on the request that it is
			 * terminated now (should be done after the callback)*/
			req->completed = 1;
			piom_cond_signal(&req->req_cond, REQ_FINALIZED);
		}
		int pending_remaining = STARPU_ATOMIC_ADD(&pending_request, -1);
		if (!running && !pending_remaining)
			starpu_sem_post(&callback_sem);
	}
	_STARPU_MPI_LOG_OUT();
}

void _starpu_mpi_handle_request_termination_callback(nm_sr_event_t event, const nm_sr_event_info_t*event_info, void*ref)
{
	_starpu_mpi_handle_request_termination(ref,event);
}

static void _starpu_mpi_handle_pending_request(struct _starpu_mpi_req *req)
{
	if(req->request_type == SEND_REQ && req->waited>1)
	{
		nm_sr_request_set_ref(&(req->size_req), req);
		nm_sr_request_monitor(req->session, &(req->size_req), NM_SR_EVENT_FINALIZED,_starpu_mpi_handle_request_termination_callback);
	}
	/* the if must be before, because the first callback can directly free
	* a detached request (the second callback free if req->waited>1). */
	nm_sr_request_set_ref(&(req->request), req);

	nm_sr_request_monitor(req->session, &(req->request), NM_SR_EVENT_FINALIZED,_starpu_mpi_handle_request_termination_callback);
}

static void _starpu_mpi_handle_new_request(void *arg)
{
	_STARPU_MPI_LOG_IN();
	struct _starpu_mpi_req *req = arg;
	STARPU_ASSERT_MSG(req, "Invalid request");

	/* submit the request to MPI */
	_STARPU_MPI_DEBUG(2, "Handling new request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n",
			  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.rank, req->data_handle, req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);
	req->func(req);

	_STARPU_MPI_LOG_OUT();
}

static void _starpu_mpi_print_thread_level_support(int thread_level, char *msg)
{
	switch (thread_level)
	{
	case MPI_THREAD_SERIALIZED:
	{
		_STARPU_DISP("MPI%s MPI_THREAD_SERIALIZED; Multiple threads may make MPI calls, but only one at a time.\n", msg);
		break;
	}
	case MPI_THREAD_FUNNELED:
	{
		_STARPU_DISP("MPI%s MPI_THREAD_FUNNELED; The application can safely make calls to StarPU-MPI functions, but should not call directly MPI communication functions.\n", msg);
		break;
	}
	case MPI_THREAD_SINGLE:
	{
		_STARPU_DISP("MPI%s MPI_THREAD_SINGLE; MPI does not have multi-thread support, this might cause problems. The application can make calls to StarPU-MPI functions, but not call directly MPI Communication functions.\n", msg);
		break;
	}
	}
}

static void *_starpu_mpi_progress_thread_func(void *arg)
{
	struct _starpu_mpi_argc_argv *argc_argv = (struct _starpu_mpi_argc_argv *) arg;

	starpu_pthread_setname("MPI");

#ifndef STARPU_SIMGRID
	if (mpi_thread_cpuid >= 0)
		_starpu_bind_thread_on_cpu(mpi_thread_cpuid, STARPU_NOWORKERID);
	_starpu_mpi_do_initialize(argc_argv);
	if (mpi_thread_cpuid >= 0)
		/* In case MPI changed the binding */
		_starpu_bind_thread_on_cpu(mpi_thread_cpuid, STARPU_NOWORKERID);
#endif

	_starpu_mpi_fake_world_size = starpu_get_env_number("STARPU_MPI_FAKE_SIZE");
	_starpu_mpi_fake_world_rank = starpu_get_env_number("STARPU_MPI_FAKE_RANK");

#ifdef STARPU_SIMGRID
	/* Now that MPI is set up, let the rest of simgrid get initialized */
	char **argv_cpy;
	_STARPU_MPI_MALLOC(argv_cpy, *(argc_argv->argc) * sizeof(char*));
	int i;
	for (i = 0; i < *(argc_argv->argc); i++)
		argv_cpy[i] = strdup((*(argc_argv->argv))[i]);
	MSG_process_create_with_arguments("main", smpi_simulated_main_, NULL, _starpu_simgrid_get_host_by_name("MAIN"), *(argc_argv->argc), argv_cpy);
	/* And set TSD for us */
	void **tsd;
	_STARPU_CALLOC(tsd, MAX_TSD + 1, sizeof(void*));
	if (!smpi_process_set_user_data)
	{
		_STARPU_ERROR("Your version of simgrid does not provide smpi_process_set_user_data, we can not continue without it\n");
	}
	smpi_process_set_user_data(tsd);
#endif

#ifdef STARPU_USE_FXT
	_starpu_fxt_wait_initialisation();
#endif //STARPU_USE_FXT

	{
		_STARPU_MPI_TRACE_START(argc_argv->rank, argc_argv->world_size);
#ifdef STARPU_USE_FXT
		starpu_profiling_set_id(argc_argv->rank);
#endif //STARPU_USE_FXT
	}

	_starpu_mpi_add_sync_point_in_fxt();
	_starpu_mpi_comm_amounts_init(argc_argv->comm);
	_starpu_mpi_cache_init(argc_argv->comm);
	_starpu_mpi_select_node_init();
	_starpu_mpi_datatype_init();

	/* notify the main thread that the progression thread is ready */
	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
	running = 1;
	STARPU_PTHREAD_COND_SIGNAL(&progress_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

	while (1)
	{
		struct callback_lfstack_cell_s* c = callback_lfstack_pop(&callback_stack);
		int err=0;

		if(running || pending_request>0)
		{/* shall we block ? */
			err = starpu_sem_wait(&callback_sem);
			//running pending_request can change while waiting
		}
		if(c==NULL)
		{
			c = callback_lfstack_pop(&callback_stack);
			if (c == NULL)
			{
				if(running && pending_request>0)
				{
					STARPU_ASSERT_MSG(c!=NULL, "Callback thread awakened without callback ready with error %d.",err);
				}
				else
				{
					if (pending_request==0)
						break;
				}
				continue;
			}
		}


		c->req->callback(c->req->callback_arg);
		if (c->req->detached)
		{
			_starpu_mpi_request_destroy(c->req);
		}
		else
		{
			c->req->completed=1;
			piom_cond_signal(&(c->req->req_cond), REQ_FINALIZED);
		}
		STARPU_ATOMIC_ADD( &pending_request, -1);
		/* we signal that the request is completed.*/

		free(c);

	}
	STARPU_ASSERT_MSG(callback_lfstack_pop(&callback_stack)==NULL, "List of callback not empty.");
	STARPU_ASSERT_MSG(pending_request==0, "Request still pending.");

	if (argc_argv->initialize_mpi)
	{
		_STARPU_MPI_DEBUG(3, "Calling MPI_Finalize()\n");
		MPI_Finalize();
	}

	starpu_sem_destroy(&callback_sem);
	free(argc_argv);
	return NULL;
}

/********************************************************/
/*                                                      */
/*  (De)Initialization methods                          */
/*                                                      */
/********************************************************/

// #ifdef STARPU_MPI_ACTIVITY
// static int hookid = - 1;
// #endif /* STARPU_MPI_ACTIVITY */

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

	starpu_sem_init(&callback_sem, 0, 0);
	running = 0;
	mpi_thread_cpuid = starpu_get_env_number_default("STARPU_MPI_THREAD_CPUID", -1);

	STARPU_PTHREAD_CREATE(&progress_thread, NULL, _starpu_mpi_progress_thread_func, argc_argv);

        STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
        while (!running)
                STARPU_PTHREAD_COND_WAIT(&progress_cond, &progress_mutex);
        STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

        return 0;
}

void _starpu_mpi_progress_shutdown(int *value)
{
	/* kill the progression thread */
        STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
        running = 0;
        STARPU_PTHREAD_COND_BROADCAST(&progress_cond);
        STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

	starpu_sem_post(&callback_sem);

	starpu_pthread_join(progress_thread, &value);

        STARPU_PTHREAD_MUTEX_DESTROY(&progress_mutex);
        STARPU_PTHREAD_COND_DESTROY(&progress_cond);
}

void _starpu_mpi_data_clear(starpu_data_handle_t data_handle)
{
	_starpu_mpi_cache_data_clear(data_handle);
	free(data_handle->mpi_data);
}

void starpu_mpi_data_register_comm(starpu_data_handle_t data_handle, int tag, int rank, MPI_Comm comm)
{
	struct _starpu_mpi_data *mpi_data;
	if (data_handle->mpi_data)
	{
		mpi_data = data_handle->mpi_data;
	}
	else
	{
		_STARPU_CALLOC(mpi_data, 1, sizeof(struct _starpu_mpi_data));
		mpi_data->magic = 42;
		mpi_data->node_tag.data_tag = -1;
		mpi_data->node_tag.rank = -1;
		mpi_data->node_tag.comm = MPI_COMM_WORLD;
		data_handle->mpi_data = mpi_data;
		_starpu_mpi_cache_data_init(data_handle);
		_starpu_data_set_unregister_hook(data_handle, _starpu_mpi_data_clear);
	}

	if (tag != -1)
	{
		mpi_data->node_tag.data_tag = tag;
	}
	if (rank != -1)
	{
		_STARPU_MPI_TRACE_DATA_SET_RANK(data_handle, rank);
		mpi_data->node_tag.rank = rank;
		mpi_data->node_tag.comm = comm;
	}
}

void starpu_mpi_data_set_rank_comm(starpu_data_handle_t handle, int rank, MPI_Comm comm)
{
	starpu_mpi_data_register_comm(handle, -1, rank, comm);
}

void starpu_mpi_data_set_tag(starpu_data_handle_t handle, int tag)
{
	starpu_mpi_data_register_comm(handle, tag, -1, MPI_COMM_WORLD);
}

int starpu_mpi_data_get_rank(starpu_data_handle_t data)
{
	STARPU_ASSERT_MSG(data->mpi_data, "starpu_mpi_data_register MUST be called for data %p\n", data);
	return ((struct _starpu_mpi_data *)(data->mpi_data))->node_tag.rank;
}

int starpu_mpi_data_get_tag(starpu_data_handle_t data)
{
	STARPU_ASSERT_MSG(data->mpi_data, "starpu_mpi_data_register MUST be called for data %p\n", data);
	return ((struct _starpu_mpi_data *)(data->mpi_data))->node_tag.data_tag;
}

void starpu_mpi_get_data_on_node_detached(MPI_Comm comm, starpu_data_handle_t data_handle, int node, void (*callback)(void*), void *arg)
{
	int me, rank, tag;

	rank = starpu_mpi_data_get_rank(data_handle);
	if (rank == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register() or starpu_mpi_data_register()\n");
	}

	starpu_mpi_comm_rank(comm, &me);
	if (node == rank)
		return;

	tag = starpu_mpi_data_get_tag(data_handle);
	if (tag == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register() or starpu_mpi_data_register()\n");
	}

	if (me == node)
	{
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_received = _starpu_mpi_cache_received_data_set(data_handle);
		if (already_received == 0)
		{
			_STARPU_MPI_DEBUG(1, "Receiving data %p from %d\n", data_handle, rank);
			starpu_mpi_irecv_detached(data_handle, rank, tag, comm, callback, arg);
		}
	}
	else if (me == rank)
	{
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_sent = _starpu_mpi_cache_sent_data_set(data_handle, node);
		if (already_sent == 0)
		{
			_STARPU_MPI_DEBUG(1, "Sending data %p to %d\n", data_handle, node);
			starpu_mpi_isend_detached(data_handle, node, tag, comm, NULL, NULL);
		}
	}
}

void starpu_mpi_get_data_on_node(MPI_Comm comm, starpu_data_handle_t data_handle, int node)
{
	int me, rank, tag;

	rank = starpu_mpi_data_get_rank(data_handle);
	if (rank == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register\n");
	}

	starpu_mpi_comm_rank(comm, &me);
	if (node == rank)
		return;

	tag = starpu_mpi_data_get_tag(data_handle);
	if (tag == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register\n");
	}

	if (me == node)
	{
		MPI_Status status;
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_received = _starpu_mpi_cache_received_data_set(data_handle);
		if (already_received == 0)
		{
			_STARPU_MPI_DEBUG(1, "Receiving data %p from %d\n", data_handle, rank);
			starpu_mpi_recv(data_handle, rank, tag, comm, &status);
		}
	}
	else if (me == rank)
	{
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_sent = _starpu_mpi_cache_sent_data_set(data_handle, node);
		if (already_sent == 0)
		{
			_STARPU_MPI_DEBUG(1, "Sending data %p to %d\n", data_handle, node);
			starpu_mpi_send(data_handle, node, tag, comm);
		}
	}
}

void starpu_mpi_get_data_on_all_nodes_detached(MPI_Comm comm, starpu_data_handle_t data_handle)
{
	int size, i;
	starpu_mpi_comm_size(comm, &size);
#ifdef STARPU_DEVEL
#warning TODO: use binary communication tree to optimize broadcast
#endif
	for (i = 0; i < size; i++)
		starpu_mpi_get_data_on_node_detached(comm, data_handle, i, NULL, NULL);
}

void starpu_mpi_data_migrate(MPI_Comm comm, starpu_data_handle_t data, int new_rank)
{
	int old_rank = starpu_mpi_data_get_rank(data);
	if (new_rank == old_rank)
		/* Already there */
		return;

	/* First submit data migration if it's not already on destination */
	starpu_mpi_get_data_on_node_detached(comm, data, new_rank, NULL, NULL);

	/* And note new owner */
	starpu_mpi_data_set_rank_comm(data, new_rank, comm);

	/* Flush cache in all other nodes */
	/* TODO: Ideally we'd transmit the knowledge of who owns it */
	starpu_mpi_cache_flush(comm, data);
	return;
}

int starpu_mpi_wait_for_all(MPI_Comm comm)
{
	int mpi = 1;
	int task = 1;
	while (task || mpi)
	{
		task = _starpu_task_wait_for_all_and_return_nb_waited_tasks();
		mpi = starpu_mpi_barrier(comm);
	}
	return 0;
}

