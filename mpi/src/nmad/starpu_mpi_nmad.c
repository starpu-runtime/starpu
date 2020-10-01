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
#include <starpu_mpi_select_node.h>
#include <starpu_mpi_init.h>
#include <common/config.h>
#include <common/thread.h>
#include <datawizard/coherency.h>
#include <core/task.h>
#include <core/topology.h>

#ifdef STARPU_USE_MPI_NMAD

#include <nm_sendrecv_interface.h>
#include <nm_mpi_nmad.h>
#include "starpu_mpi_nmad_backend.h"

static void _starpu_mpi_handle_request_termination(struct _starpu_mpi_req *req,nm_sr_event_t event);
#ifdef STARPU_VERBOSE
char *_starpu_mpi_request_type(enum _starpu_mpi_request_type request_type);
#endif

static void _starpu_mpi_handle_pending_request(struct _starpu_mpi_req *req);
static void _starpu_mpi_add_sync_point_in_fxt(void);

/* Condition to wake up waiting for all current MPI requests to finish */
static starpu_pthread_t progress_thread;
static starpu_pthread_cond_t progress_cond;
static starpu_pthread_mutex_t progress_mutex;
static volatile int running = 0;

extern struct _starpu_mpi_req *_starpu_mpi_irecv_common(starpu_data_handle_t data_handle, int source, int data_tag, MPI_Comm comm, unsigned detached, unsigned sync, void (*callback)(void *), void *arg, int sequential_consistency, int is_internal_req, starpu_ssize_t count);

/* Count requests posted by the application and not yet submitted to MPI, i.e pushed into the new_requests list */

static volatile int pending_request = 0;

#define REQ_FINALIZED 0x1

PUK_LFSTACK_TYPE(callback, struct _starpu_mpi_req *req;);
static callback_lfstack_t callback_stack;

static starpu_sem_t callback_sem;

/********************************************************/
/*                                                      */
/*  Send/Receive functionalities                        */
/*                                                      */
/********************************************************/

void _starpu_mpi_req_willpost(struct _starpu_mpi_req *req STARPU_ATTRIBUTE_UNUSED)
{
	STARPU_ATOMIC_ADD( &pending_request, 1);
}

/********************************************************/
/*                                                      */
/*  Send functionalities                                */
/*                                                      */
/********************************************************/

static void _starpu_mpi_isend_data_func(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	_STARPU_MPI_DEBUG(30, "post NM isend request %p type %s tag %ld src %d data %p datasize %ld ptr %p datatype '%s' count %d registered_datatype %d sync %d\n", req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, starpu_data_get_size(req->data_handle), req->ptr, req->datatype_name, (int)req->count, req->registered_datatype, req->sync);

	_starpu_mpi_comm_amounts_inc(req->node_tag.node.comm, req->node_tag.node.rank, req->datatype, req->count);

	_STARPU_MPI_TRACE_ISEND_SUBMIT_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag, 0);

	struct nm_data_s data;
	nm_mpi_nmad_data_get(&data, (void*)req->ptr, req->datatype, req->count);
	nm_sr_send_init(req->backend->session, &(req->backend->data_request));
	nm_sr_send_pack_data(req->backend->session, &(req->backend->data_request), &data);
	nm_sr_send_set_priority(req->backend->session, &req->backend->data_request, req->prio);

	if (req->sync == 0)
	{
		req->ret = nm_sr_send_isend(req->backend->session, &(req->backend->data_request), req->backend->gate, req->node_tag.data_tag);
		STARPU_ASSERT_MSG(req->ret == NM_ESUCCESS, "MPI_Isend returning %d", req->ret);
	}
	else
	{
		req->ret = nm_sr_send_issend(req->backend->session, &(req->backend->data_request), req->backend->gate, req->node_tag.data_tag);
		STARPU_ASSERT_MSG(req->ret == NM_ESUCCESS, "MPI_Issend returning %d", req->ret);
	}

	_STARPU_MPI_TRACE_ISEND_SUBMIT_END(req->node_tag.node.rank, req->node_tag.data_tag, starpu_data_get_size(req->data_handle), req->pre_sync_jobid);

	_starpu_mpi_handle_pending_request(req);

	_STARPU_MPI_LOG_OUT();
}

void _starpu_mpi_isend_size_func(struct _starpu_mpi_req *req)
{
	_starpu_mpi_datatype_allocate(req->data_handle, req);

	if (req->registered_datatype == 1)
	{
		req->backend->waited = 1;
		req->count = 1;
		req->ptr = starpu_data_handle_to_pointer(req->data_handle, STARPU_MAIN_RAM);
	}
	else
	{
		starpu_ssize_t psize = -1;
		int ret;
		req->backend->waited =2;

		// Do not pack the data, just try to find out the size
		starpu_data_pack(req->data_handle, NULL, &psize);

		if (psize != -1)
		{
			// We already know the size of the data, let's send it to overlap with the packing of the data
			_STARPU_MPI_DEBUG(20, "Sending size %ld (%ld %s) to node %d (first call to pack)\n", psize, sizeof(req->count), "MPI_BYTE", req->node_tag.node.rank);
			req->count = psize;
			//ret = nm_sr_isend(nm_mpi_communicator_get_session(p_req->p_comm),nm_mpi_communicator_get_gate(p_comm,req->srcdst), req->mpi_tag,&req->count, sizeof(req->count), &req->backend->size_req);
			ret = nm_sr_isend(req->backend->session,req->backend->gate, req->node_tag.data_tag,&req->count, sizeof(req->count), &req->backend->size_req);

			//	ret = MPI_Isend(&req->count, sizeof(req->count), MPI_BYTE, req->srcdst, req->mpi_tag, req->comm, &req->backend->size_req);
			STARPU_ASSERT_MSG(ret == NM_ESUCCESS, "when sending size, nm_sr_isend returning %d", ret);
		}

		// Pack the data
		starpu_data_pack(req->data_handle, &req->ptr, &req->count);
		if (psize == -1)
		{
			// We know the size now, let's send it
			_STARPU_MPI_DEBUG(1, "Sending size %ld (%ld %s) with tag %ld to node %d (second call to pack)\n", req->count, sizeof(req->count), "MPI_BYTE", req->node_tag.data_tag, req->node_tag.node.rank);
			ret = nm_sr_isend(req->backend->session,req->backend->gate, req->node_tag.data_tag,&req->count, sizeof(req->count), &req->backend->size_req);
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

/********************************************************/
/*                                                      */
/*  Receive functionalities                             */
/*                                                      */
/********************************************************/

static void _starpu_mpi_irecv_data_func(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	_STARPU_MPI_DEBUG(20, "post NM irecv request %p type %s tag %ld src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n", req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);

	_STARPU_MPI_TRACE_IRECV_SUBMIT_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag);

	//req->ret = MPI_Irecv(req->ptr, req->count, req->datatype, req->srcdst, req->mpi_tag, req->comm, &req->request);
	struct nm_data_s data;
	nm_mpi_nmad_data_get(&data, (void*)req->ptr, req->datatype, req->count);
	nm_sr_recv_init(req->backend->session, &(req->backend->data_request));
	nm_sr_recv_unpack_data(req->backend->session, &(req->backend->data_request), &data);
	nm_sr_recv_irecv(req->backend->session, &(req->backend->data_request), req->backend->gate, req->node_tag.data_tag, NM_TAG_MASK_FULL);

	_STARPU_MPI_TRACE_IRECV_SUBMIT_END(req->node_tag.node.rank, req->node_tag.data_tag);

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
	callback->req->ptr = (void *)starpu_malloc_on_node_flags(STARPU_MAIN_RAM, callback->req->count, 0);
	STARPU_ASSERT_MSG(callback->req->ptr, "cannot allocate message of size %ld", callback->req->count);
	_starpu_mpi_irecv_data_func(callback->req);
	free(callback);
}

void _starpu_mpi_irecv_size_func(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	_starpu_mpi_datatype_allocate(req->data_handle, req);
	if (req->registered_datatype == 1)
	{
		req->count = 1;
		req->ptr = starpu_data_handle_to_pointer(req->data_handle, STARPU_MAIN_RAM);
		_starpu_mpi_irecv_data_func(req);
	}
	else
	{
		struct _starpu_mpi_irecv_size_callback *callback = malloc(sizeof(struct _starpu_mpi_irecv_size_callback));
		callback->req = req;
		starpu_variable_data_register(&callback->handle, 0, (uintptr_t)&(callback->req->count), sizeof(callback->req->count));
		_STARPU_MPI_DEBUG(4, "Receiving size with tag %ld from node %d\n", req->node_tag.data_tag, req->node_tag.node.rank);
		_starpu_mpi_irecv_common(callback->handle, req->node_tag.node.rank, req->node_tag.data_tag, req->node_tag.node.comm, 1, 0, _starpu_mpi_irecv_size_callback, callback,1,0,0);
	}

}

/********************************************************/
/*                                                      */
/*  Wait functionalities                                */
/*                                                      */
/********************************************************/

#define _starpu_mpi_req_status(PUBLIC_REQ,STATUS) do {			\
	STATUS->MPI_SOURCE=PUBLIC_REQ->node_tag.node.rank; /**< field name mandatory by spec */ \
	STATUS->MPI_TAG=PUBLIC_REQ->node_tag.data_tag;    /**< field name mandatory by spec */ \
	STATUS->MPI_ERROR=PUBLIC_REQ->ret;  /**< field name mandatory by spec */ \
	STATUS->size=PUBLIC_REQ->count;       /**< size of data received */ \
	STATUS->cancelled=0;  /**< whether request was cancelled */	\
} while(0)

int _starpu_mpi_wait(starpu_mpi_req *public_req, MPI_Status *status)
{
	_STARPU_MPI_LOG_IN();
	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_wait needs a valid starpu_mpi_req");
	struct _starpu_mpi_req *req = *public_req;
	STARPU_MPI_ASSERT_MSG(!req->detached, "MPI_Wait cannot be called on a detached request");

	/* we must do a test_locked to avoid race condition :
	 * without req_cond could still be used and couldn't be freed)*/
	while (!req->completed || ! piom_cond_test_locked(&(req->backend->req_cond),REQ_FINALIZED))
	{
		piom_cond_wait(&(req->backend->req_cond),REQ_FINALIZED);
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

int _starpu_mpi_test(starpu_mpi_req *public_req, int *flag, MPI_Status *status)
{
	_STARPU_MPI_LOG_IN();
	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_test needs a valid starpu_mpi_req");
	struct _starpu_mpi_req *req = *public_req;
	STARPU_MPI_ASSERT_MSG(!req->detached, "MPI_Test cannot be called on a detached request");
	_STARPU_MPI_DEBUG(2, "Test request %p type %s tag %ld src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n",
			  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);

	STARPU_VALGRIND_YIELD();

	_STARPU_MPI_TRACE_UTESTING_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag);

	/* we must do a test_locked to avoid race condition :
	 * without req_cond could still be used and couldn't be freed)*/
	*flag = req->completed && piom_cond_test_locked(&(req->backend->req_cond),REQ_FINALIZED);
	if (*flag && status!=MPI_STATUS_IGNORE)
		_starpu_mpi_req_status(req,status);

	_STARPU_MPI_TRACE_UTESTING_END(req->node_tag.node.rank, req->node_tag.data_tag);

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

int _starpu_mpi_barrier(MPI_Comm comm)
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
char *_starpu_mpi_request_type(enum _starpu_mpi_request_type request_type)
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

	_STARPU_MPI_DEBUG(2, "complete MPI request %p type %s tag %ld src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n",
			  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);

	if (req->request_type == RECV_REQ || req->request_type == SEND_REQ)
	{
		if (req->registered_datatype == 0)
		{
			if(req->backend->waited == 1)
			        nm_mpi_nmad_data_release(req->datatype);
			if (req->request_type == SEND_REQ)
			{
				req->backend->waited--;
				// We need to make sure the communication for sending the size
				// has completed, as MPI can re-order messages, let's count
				// recerived message.
				// FIXME concurent access.
				STARPU_ASSERT_MSG(event == NM_SR_EVENT_FINALIZED, "Callback with event %d", event);
				if(req->backend->waited>0)
					return;

			}
			if (req->request_type == RECV_REQ)
				// req->ptr is freed by starpu_data_unpack
				starpu_data_unpack(req->data_handle, req->ptr, req->count);
			else
				starpu_free_on_node_flags(STARPU_MAIN_RAM, (uintptr_t) req->ptr, req->count, 0);
		}
		else
		{
		        nm_mpi_nmad_data_release(req->datatype);
			_starpu_mpi_datatype_free(req->data_handle, &req->datatype);
		}
	}
	_STARPU_MPI_TRACE_TERMINATED(req, req->node_tag.node.rank, req->node_tag.data_tag);
	_starpu_mpi_release_req_data(req);

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
			piom_cond_signal(&req->backend->req_cond, REQ_FINALIZED);
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
	if(req->request_type == SEND_REQ && req->backend->waited>1)
	{
		nm_sr_request_set_ref(&(req->backend->size_req), req);
		nm_sr_request_monitor(req->backend->session, &(req->backend->size_req), NM_SR_EVENT_FINALIZED,_starpu_mpi_handle_request_termination_callback);
	}
	/* the if must be before, because the first callback can directly free
	* a detached request (the second callback free if req->backend->waited>1). */
	nm_sr_request_set_ref(&(req->backend->data_request), req);

	nm_sr_request_monitor(req->backend->session, &(req->backend->data_request), NM_SR_EVENT_FINALIZED,_starpu_mpi_handle_request_termination_callback);
}

void _starpu_mpi_coop_sends_build_tree(struct _starpu_mpi_coop_sends *coop_sends)
{
	/* TODO: turn them into redirects & forwards */
}

void _starpu_mpi_submit_coop_sends(struct _starpu_mpi_coop_sends *coop_sends, int submit_control, int submit_data)
{
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
	STARPU_ASSERT_MSG(req, "Invalid request");

	/* submit the request to MPI directly from submitter */
	_STARPU_MPI_DEBUG(2, "Handling new request %p type %s tag %ld src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n",
			  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);
	req->func(req);

	_STARPU_MPI_LOG_OUT();
}

static void *_starpu_mpi_progress_thread_func(void *arg)
{
	struct _starpu_mpi_argc_argv *argc_argv = (struct _starpu_mpi_argc_argv *) arg;

#ifndef STARPU_SIMGRID
	if (!_starpu_mpi_nobind && starpu_bind_thread_on(_starpu_mpi_thread_cpuid, 0, "MPI") < 0)
	{
		_STARPU_DISP("No core was available for the MPI thread. You should use STARPU_RESERVE_NCPU to leave one core available for MPI, or specify one core less in STARPU_NCPU\n");
	}
#endif

#ifdef STARPU_SIMGRID
	/* Now that MPI is set up, let the rest of simgrid get initialized */
	char **argv_cpy;
	_STARPU_MPI_MALLOC(argv_cpy, *(argc_argv->argc) * sizeof(char*));
	int i;
	for (i = 0; i < *(argc_argv->argc); i++)
		argv_cpy[i] = strdup((*(argc_argv->argv))[i]);
#ifdef HAVE_SG_ACTOR_DATA
	_starpu_simgrid_actor_create("main", smpi_simulated_main_, _starpu_simgrid_get_host_by_name("MAIN"), *(argc_argv->argc), argv_cpy);
#else
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
#endif

	_starpu_mpi_comm_amounts_init(argc_argv->comm);
	_starpu_mpi_cache_init(argc_argv->comm);
	_starpu_mpi_select_node_init();
	_starpu_mpi_datatype_init();

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
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

	while (1)
	{
		struct callback_lfstack_cell_s* c = callback_lfstack_pop(&callback_stack);
		int err=0;

		if(running || pending_request>0)
		{
			/* shall we block ? */
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
			piom_cond_signal(&(c->req->backend->req_cond), REQ_FINALIZED);
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

	_starpu_mpi_env_init();

	/* This function calls MPI_Init_thread if needed, and it initializes internal NMAD/Pioman variables,
	 * required for piom_ltask_set_bound_thread_indexes() */
	_starpu_mpi_do_initialize(argc_argv);

	callback_lfstack_init(&callback_stack);

	if (!_starpu_mpi_nobind && _starpu_mpi_thread_cpuid < 0)
	{
		_starpu_mpi_thread_cpuid = starpu_get_next_bindid(STARPU_THREAD_ACTIVE, NULL, 0);
	}

	/* Tell pioman to use a bound thread for communication progression:
	 * share the same core as StarPU's MPI thread, the MPI thread has very low activity with NMAD backend */
#ifdef HAVE_PIOM_LTASK_SET_BOUND_THREAD_OS_INDEXES
	/* We prefer to give the OS index of the core, because StarPU can have
	 * a different vision of the topology, especially if STARPU_WORKERS_GETBIND
	 * is enabled */
	int indexes[1] = { starpu_get_pu_os_index((unsigned) _starpu_mpi_thread_cpuid) };
	if (!_starpu_mpi_nobind)
		piom_ltask_set_bound_thread_os_indexes(HWLOC_OBJ_PU, indexes, 1);
#else
	int indexes[1] = { _starpu_mpi_thread_cpuid };
	if (!_starpu_mpi_nobind)
		piom_ltask_set_bound_thread_indexes(HWLOC_OBJ_PU, indexes, 1);
#endif

	/* Register some hooks for communication progress if needed */
	int polling_point_prog, polling_point_idle;
	char *s_prog_hooks = starpu_getenv("STARPU_MPI_NMAD_PROG_HOOKS");
	char *s_idle_hooks = starpu_getenv("STARPU_MPI_NMAD_IDLE_HOOKS");

	if(!s_prog_hooks)
	{
		polling_point_prog = 0;
	}
	else
	{
		polling_point_prog =
			(strcmp(s_prog_hooks, "FORCED") == 0) ? PIOM_POLL_POINT_FORCED :
			(strcmp(s_prog_hooks, "SINGLE") == 0) ? PIOM_POLL_POINT_SINGLE :
			(strcmp(s_prog_hooks, "HOOK")   == 0) ? PIOM_POLL_POINT_HOOK :
			0;
	}

	if(!s_idle_hooks)
	{
		polling_point_idle = 0;
	}
	else
	{
		polling_point_idle =
			(strcmp(s_idle_hooks, "FORCED") == 0) ? PIOM_POLL_POINT_FORCED :
			(strcmp(s_idle_hooks, "SINGLE") == 0) ? PIOM_POLL_POINT_SINGLE :
			(strcmp(s_idle_hooks, "HOOK")   == 0) ? PIOM_POLL_POINT_HOOK :
			0;
	}

	if(polling_point_prog)
	{
		starpu_progression_hook_register((unsigned (*)(void *))&piom_ltask_schedule, (void *)&polling_point_prog);
	}

	if(polling_point_idle)
	{
		starpu_idle_hook_register((unsigned (*)(void *))&piom_ltask_schedule, (void *)&polling_point_idle);
	}

	/* Launch thread used for nmad callbacks */
	STARPU_PTHREAD_CREATE(&progress_thread, NULL, _starpu_mpi_progress_thread_func, argc_argv);

        STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
        while (!running)
                STARPU_PTHREAD_COND_WAIT(&progress_cond, &progress_mutex);
        STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

        return 0;
}

void _starpu_mpi_progress_shutdown(void **value)
{
	/* kill the progression thread */
        STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
        running = 0;
        STARPU_PTHREAD_COND_BROADCAST(&progress_cond);
        STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

	starpu_sem_post(&callback_sem);

	STARPU_PTHREAD_JOIN(progress_thread, value);

	callback_lfstack_destroy(&callback_stack);

        STARPU_PTHREAD_MUTEX_DESTROY(&progress_mutex);
        STARPU_PTHREAD_COND_DESTROY(&progress_cond);
}

static int64_t _starpu_mpi_tag_max = INT64_MAX;

int starpu_mpi_comm_get_attr(MPI_Comm comm, int keyval, void *attribute_val, int *flag)
{
	(void) comm;
	if (keyval == STARPU_MPI_TAG_UB)
	{
		if ((uint64_t) _starpu_mpi_tag_max > NM_TAG_MAX)
			_starpu_mpi_tag_max = NM_TAG_MAX;
		/* manage case where nmad max tag causes overflow if represented as starpu tag */
		*(int64_t **)attribute_val = &_starpu_mpi_tag_max;
		*flag = 1;
	}
	else
	{
		*flag = 0;
	}
	return 0;
}

#endif /* STARPU_USE_MPI_NMAD*/
