/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <common/config.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <starpu_mpi.h>
#include <starpu_mpi_datatype.h>
#include <starpu_mpi_private.h>
#include <starpu_mpi_cache.h>
#include <starpu_profiling.h>
#include <starpu_mpi_stats.h>
#include <starpu_mpi_cache.h>
#include <starpu_mpi_select_node.h>
#include <starpu_mpi_init.h>
#include <common/thread.h>
#include <datawizard/coherency.h>
#include <core/task.h>
#include <core/topology.h>

#ifdef STARPU_USE_FXT
#include <starpu_mpi_fxt.h>
#endif

#ifdef STARPU_USE_MPI_NMAD
#include <nm_mpi_nmad.h>
#include <nm_mcast_interface.h>
#include <nm_sendrecv_interface.h>

#include "starpu_mpi_nmad_coop.h"
#include "starpu_mpi_nmad_backend.h"
#include "starpu_mpi_nmad_unknown_datatype.h"

void _starpu_mpi_handle_request_termination(struct _starpu_mpi_req *req);
void _starpu_mpi_handle_pending_request(struct _starpu_mpi_req *req);
static inline void _starpu_mpi_request_end(struct _starpu_mpi_req* req, int post_callback_sem);
static inline void _starpu_mpi_request_try_end(struct _starpu_mpi_req* req, int post_callback_sem);


/* Condition to wake up waiting for all current MPI requests to finish */
static starpu_pthread_t progress_thread;
static starpu_pthread_cond_t progress_cond;
static starpu_pthread_mutex_t progress_mutex;
static volatile int running = 0;

static starpu_pthread_cond_t mpi_wait_for_all_running_cond;
static int mpi_wait_for_all_running = 0;
static starpu_pthread_mutex_t mpi_wait_for_all_running_mutex;

/* Count running requests: this counter is incremented just before StarPU
 * submits a MPI request, and decremented when a MPI request finishes. */
static volatile int nb_pending_requests = 0;

#define REQ_FINALIZED 0x1

PUK_LFSTACK_TYPE(callback, struct _starpu_mpi_req *req;);
static callback_lfstack_t callback_stack;

static starpu_sem_t callback_sem;

static int nmad_mcast_started = 0;


/********************************************************/
/*                                                      */
/*  Send/Receive functionalities                        */
/*                                                      */
/********************************************************/

void _starpu_mpi_req_willpost(struct _starpu_mpi_req *req STARPU_ATTRIBUTE_UNUSED)
{
	int new_nb = STARPU_ATOMIC_ADD( &nb_pending_requests, 1);
	(void)new_nb;
}

/********************************************************/
/*                                                      */
/*  Send functionalities                                */
/*                                                      */
/********************************************************/

static void _starpu_mpi_isend_known_datatype(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	STARPU_ASSERT_MSG(req->registered_datatype == 1, "Datatype is not registered, it cannot be sent through this way !");

	_STARPU_MPI_DEBUG(30, "post NM isend request %p type %s tag %ld src %d data %p datasize %ld ptr %p datatype '%s' count %d registered_datatype %d sync %d\n", req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, starpu_data_get_size(req->data_handle), req->ptr, req->datatype_name, (int)req->count, req->registered_datatype, req->sync);

	_starpu_mpi_comm_amounts_inc(req->node_tag.node.comm, req->node_tag.node.rank, req->datatype, req->count);

	_STARPU_MPI_TRACE_ISEND_SUBMIT_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag, 0);

	struct nm_data_s data;
	nm_mpi_nmad_data_get(&data, (void*)req->ptr, req->datatype, req->count);
	nm_sr_send_init(req->backend->session, &(req->backend->data_request));
	nm_sr_send_pack_data(req->backend->session, &(req->backend->data_request), &data);
	nm_sr_send_set_priority(req->backend->session, &req->backend->data_request, req->prio);

	// this trace event is the start of the communication link:
	_STARPU_MPI_TRACE_ISEND_SUBMIT_END(_STARPU_MPI_FUT_POINT_TO_POINT_SEND, req, req->prio);

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

	_starpu_mpi_handle_pending_request(req);

	_STARPU_MPI_LOG_OUT();
}

void _starpu_mpi_isend_func(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	_starpu_mpi_datatype_allocate(req->data_handle, req);

	if (req->registered_datatype == 1)
	{
		req->count = 1;
		req->ptr = starpu_data_handle_to_pointer(req->data_handle, req->node);

		_starpu_mpi_isend_known_datatype(req);
	}
	else
	{
		_starpu_mpi_isend_unknown_datatype(req);
	}

	_STARPU_MPI_LOG_OUT();
}

/********************************************************/
/*                                                      */
/*  Receive functionalities                             */
/*                                                      */
/********************************************************/

static void _starpu_mpi_irecv_known_datatype(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	STARPU_ASSERT_MSG(req->registered_datatype == 1, "Datatype is not registered, it cannot be received through this way !");

	_STARPU_MPI_DEBUG(20, "post NM irecv request %p type %s tag %ld src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n", req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);

	_STARPU_MPI_TRACE_IRECV_SUBMIT_BEGIN(req->node_tag.node.rank, req->node_tag.data_tag);

	struct nm_data_s data;
	nm_mpi_nmad_data_get(&data, (void*)req->ptr, req->datatype, req->count);
	nm_sr_recv_init(req->backend->session, &(req->backend->data_request));
	nm_sr_recv_unpack_data(req->backend->session, &(req->backend->data_request), &data);
	nm_sr_recv_irecv(req->backend->session, &(req->backend->data_request), req->backend->gate, req->node_tag.data_tag, NM_TAG_MASK_FULL);

	_STARPU_MPI_TRACE_IRECV_SUBMIT_END(req->node_tag.node.rank, req->node_tag.data_tag);

	_starpu_mpi_handle_pending_request(req);

	_STARPU_MPI_LOG_OUT();
}

void _starpu_mpi_irecv_func(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	_starpu_mpi_datatype_allocate(req->data_handle, req);
	if (req->registered_datatype == 1)
	{
		req->count = 1;
		req->ptr = starpu_data_handle_to_pointer(req->data_handle, req->node);
		_starpu_mpi_irecv_known_datatype(req);
	}
	else
	{
		_starpu_mpi_irecv_unknown_datatype(req);
	}

	_STARPU_MPI_LOG_OUT();
}

/********************************************************/
/*                                                      */
/*  Wait functionalities                                */
/*                                                      */
/********************************************************/

#define _starpu_mpi_req_status(PUBLIC_REQ,STATUS) do { \
	STATUS->MPI_SOURCE=PUBLIC_REQ->node_tag.node.rank; /**< field name mandatory by spec */ \
	STATUS->MPI_TAG=PUBLIC_REQ->node_tag.data_tag;    /**< field name mandatory by spec */ \
	STATUS->MPI_ERROR=PUBLIC_REQ->ret;  /**< field name mandatory by spec */ \
	STATUS->size=PUBLIC_REQ->count;       /**< size of data received */ \
	STATUS->cancelled=0;  /**< whether request was cancelled */ \
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

	_starpu_mpi_request_try_end(req, 1);
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
		_starpu_mpi_request_try_end(req, 1);
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

	int ret = MPI_Barrier(comm);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Barrier returning %d", ret);

	_STARPU_MPI_LOG_OUT();
	return ret;
}

int _starpu_mpi_wait_for_all(MPI_Comm comm)
{
	(void) comm;
	_STARPU_MPI_LOG_IN();

	STARPU_PTHREAD_MUTEX_LOCK(&mpi_wait_for_all_running_mutex);
	STARPU_MPI_ASSERT_MSG(!mpi_wait_for_all_running, "Concurrent starpu_mpi_wait_for_all is not implemented, even on different communicators");
	mpi_wait_for_all_running = 1;
	do
	{
		while (nb_pending_requests)
			STARPU_PTHREAD_COND_WAIT(&mpi_wait_for_all_running_cond, &mpi_wait_for_all_running_mutex);
		STARPU_PTHREAD_MUTEX_UNLOCK(&mpi_wait_for_all_running_mutex);

		starpu_task_wait_for_all();

		STARPU_PTHREAD_MUTEX_LOCK(&mpi_wait_for_all_running_mutex);
	} while (nb_pending_requests);
	mpi_wait_for_all_running = 0;
	STARPU_PTHREAD_MUTEX_UNLOCK(&mpi_wait_for_all_running_mutex);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

/********************************************************/
/*                                                      */
/*  Progression                                         */
/*                                                      */
/********************************************************/

/* Completely finalize a request: destroy it and decrement the number of pending requests */
static inline void _starpu_mpi_request_end(struct _starpu_mpi_req* req, int post_callback_sem)
{
	/* Destroying a request and decrementing the number of pending requests
	 * should be done together, so let's wrap these two things in a
	 * function. This means instead of calling _starpu_mpi_request_destroy(),
	 * you should call this function. */

	/* If request went through _starpu_mpi_handle_received_data(), finalized has to be true: */
	assert((req->backend->has_received_data && req->backend->finalized) || !req->backend->has_received_data);

	_starpu_mpi_request_destroy(req);

	int pending_remaining = STARPU_ATOMIC_ADD(&nb_pending_requests, -1);
	assert(pending_remaining >= 0);
	if (!pending_remaining)
	{
		STARPU_PTHREAD_COND_BROADCAST(&mpi_wait_for_all_running_cond);
		if (post_callback_sem && !running)
		{
			starpu_sem_post(&callback_sem);
		}
	}
}

/* Check if the caller has to completely finalize a request and try to do it */
static inline void _starpu_mpi_request_try_end(struct _starpu_mpi_req* req, int post_callback_sem)
{
	_starpu_spin_lock(&req->backend->finalized_to_destroy_lock);
	if (!req->backend->has_received_data || req->backend->finalized)
	{
		_starpu_spin_unlock(&req->backend->finalized_to_destroy_lock);
		_starpu_mpi_request_end(req, post_callback_sem);
	}
	else
	{
		/* Request isn't finalized yet (NewMadeleine still needs it), since
		 * this function should have destroyed the request, tell
		 * _starpu_mpi_handle_request_termination() to destroy it when
		 * NewMadeleine won't need it anymore. */
		req->backend->to_destroy = 1;
		_starpu_spin_unlock(&req->backend->finalized_to_destroy_lock);
	}
}

/* Do required actions when a request is completed (but maybe not finalized!) */
static inline void _starpu_mpi_handle_post_actions(struct _starpu_mpi_req* req)
{
	if (req->callback)
	{
		/* Callbacks are executed outside of this function, later by the
		 * progression thread.
		 * Indeed, this current function is executed by a NewMadeleine handler,
		 * and possibly inside of a PIOman ltask. In such context, some locking
		 * or system calls can be forbidden to avoid any deadlock, thus
		 * callbacks are deported outside of this handler. */
		struct callback_lfstack_cell_s* c = padico_malloc(sizeof(struct callback_lfstack_cell_s));
		c->req = req;
		callback_lfstack_push(&callback_stack, c);

		/* The main thread can exit without waiting
		* the end of the detached request. Callback thread
		* must then be kept alive if they have a callback.*/
		starpu_sem_post(&callback_sem);
	}
	else if(!req->detached)
	{
		/* tell anyone potentially waiting on the request that it is
		 * terminated now (should be done after the callback)*/
		req->completed = 1;
		piom_cond_signal(&req->backend->req_cond, REQ_FINALIZED);
	}
}

/* Function called when data arrived, but NewMadeleine still holds a reference
 * on it (to make progress a broadcast for instance). Application can thus read
 * the data, but not yet write it. */
void _starpu_mpi_handle_received_data(struct _starpu_mpi_req* req)
{
	_STARPU_MPI_LOG_IN();

	assert(req->request_type == RECV_REQ);
	assert(!_starpu_mpi_recv_wait_finalize);
	assert(!req->backend->has_received_data);
	assert(!req->backend->finalized);

	req->backend->has_received_data = 1;

	if (req->registered_datatype == 0)
	{
		/* Without peek_data, we can't unpack data for StarPU's use and keep
		 * the buffer alive for NewMadeleine, so calling
		 * _starpu_mpi_handle_received_data() makes no sense. */
		assert(starpu_data_get_interface_ops(req->data_handle)->peek_data);
		starpu_data_peek_node(req->data_handle, req->node, req->ptr, req->count);
	}

	// Release write acquire on the handle: can unlock tasks waiting to read the handle:
	starpu_data_release_to(req->data_handle, STARPU_R);

	_starpu_mpi_handle_post_actions(req);

	_STARPU_MPI_LOG_OUT();
}

/* Function called when nmad completely finished a request */
void _starpu_mpi_handle_request_termination(struct _starpu_mpi_req* req)
{
	_STARPU_MPI_LOG_IN();

	_STARPU_MPI_DEBUG(2, "complete MPI request %p type %s tag %ld src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n",
			  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.node.rank, req->data_handle, req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);

	assert(!req->backend->finalized);

	if (req->request_type == RECV_REQ || req->request_type == SEND_REQ)
	{
		if (req->registered_datatype == 0)
		{
			if (req->request_type == RECV_REQ)
			{
				if (starpu_data_get_interface_ops(req->data_handle)->peek_data)
				{
					if (!req->backend->has_received_data)
					{
						starpu_data_peek_node(req->data_handle, req->node, req->ptr, req->count);
					}
					starpu_free_on_node_flags(req->node, (uintptr_t) req->ptr, req->count, 0);
				}
				else
				{
					// req->ptr is freed by starpu_data_unpack
					starpu_data_unpack_node(req->data_handle, req->node, req->ptr, req->count);
				}
			}
			else
				starpu_free_on_node_flags(req->node, (uintptr_t) req->ptr, req->count, 0);
		}
		else if (req->backend->posted) // with coop, only one request is really used to do the broadcast, so only posted request really allocates memory for the data:
		{
			nm_mpi_nmad_data_release(req->datatype);
			_starpu_mpi_datatype_free(req->data_handle, &req->datatype);
		}
	}

	// for recv requests, this event is the end of the communication link:
	_STARPU_MPI_TRACE_TERMINATED(req);

	_starpu_mpi_release_req_data(req);

	if (req->backend->has_received_data)
	{
		assert(req->request_type == RECV_REQ);

		/* Callback, test or wait were unlocked by
		 * _starpu_mpi_handle_received_data(), maybe they were already
		 * executed and since the request wasn't finalized yet, they didn't
		 * destroy the request, and we have to do it now: */
		_starpu_spin_lock(&req->backend->finalized_to_destroy_lock);
		req->backend->finalized = 1;
		if (req->backend->to_destroy || req->detached)
		{
			_starpu_spin_unlock(&req->backend->finalized_to_destroy_lock);
			_starpu_mpi_request_end(req, 1);
		}
		else
		{
			_starpu_spin_unlock(&req->backend->finalized_to_destroy_lock);
		}
	}
	else if (!req->callback && req->detached)
	{
		/* This request has no callback and is detached: we have to end it now: */
		_starpu_mpi_request_end(req, 1);
	}
	else
	{
		_starpu_mpi_handle_post_actions(req);
	}

	_STARPU_MPI_LOG_OUT();
}

void _starpu_mpi_handle_request_termination_callback(nm_sr_event_t event STARPU_ATTRIBUTE_UNUSED, const nm_sr_event_info_t* event_info STARPU_ATTRIBUTE_UNUSED, void* ref)
{
	assert(ref != NULL);

	struct _starpu_mpi_req* req = (struct _starpu_mpi_req*) ref;
	req->backend->posted = 1; // a network event was triggered for this request, so it was really posted

	if (event & NM_SR_EVENT_FINALIZED)
	{
		_starpu_mpi_handle_request_termination(req);
	}
	else if (event & NM_SR_EVENT_RECV_COMPLETED && req->request_type == RECV_REQ && !_starpu_mpi_recv_wait_finalize && req->sequential_consistency)
	{
		/* About required sequential consistency:
		 * "If it is 0, user can launch tasks writing in the handle, which will
		 * mix data manipulated by nmad and data manipulated by tasks, this
		 * could break some expected behaviours." (sthibault) */

		/* Unknown datatype case is in starpu_mpi_nmad_unknown_datatype.c */
		assert(req->registered_datatype == 1);

		_starpu_mpi_handle_received_data(req);
	}
}

void _starpu_mpi_handle_pending_request(struct _starpu_mpi_req *req)
{
	assert(req != NULL);
	nm_sr_request_set_ref(&req->backend->data_request, req);
	int ret = nm_sr_request_monitor(req->backend->session, &req->backend->data_request,
									NM_SR_EVENT_FINALIZED | NM_SR_EVENT_RECV_COMPLETED,
									_starpu_mpi_handle_request_termination_callback);
	assert(ret == NM_ESUCCESS);
}

void _starpu_mpi_submit_ready_request(void *arg)
{
	_STARPU_MPI_LOG_IN();
	struct _starpu_mpi_req *req = arg;
	STARPU_ASSERT_MSG(req, "Invalid request");

	if (req->reserved_size)
	{
		/* The core will have really allocated the reception buffer now, release our reservation */
		starpu_memory_deallocate(req->node, req->reserved_size);
		req->reserved_size = 0;
	}

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
		char hostname[65];
		gethostname(hostname, sizeof(hostname));
		_STARPU_DISP("[%s] No core was available for the MPI thread. You should use STARPU_RESERVE_NCPU to leave one core available for MPI, or specify one core less in STARPU_NCPU\n", hostname);
	}
#endif

#ifdef STARPU_SIMGRID
	/* Now that MPI is set up, let the rest of simgrid get initialized */
	char **argv_cpy;
	_STARPU_MPI_MALLOC(argv_cpy, *(argc_argv->argc) * sizeof(char*));
	int i;
	for (i = 0; i < *(argc_argv->argc); i++)
		argv_cpy[i] = strdup((*(argc_argv->argv))[i]);
#if defined(HAVE_SG_ACTOR_DATA) || defined(HAVE_SG_ACTOR_SET_DATA)
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
	_starpu_mpi_fxt_init(argc_argv);
#endif

	if (_starpu_mpi_use_coop_sends)
	{
		if (argc_argv->world_size > 2)
		{
			_starpu_mpi_nmad_coop_init();
			nmad_mcast_started = 1; // to shutdown mcast
		}
		else
		{
			_starpu_mpi_use_coop_sends = 0;
		}
	}

	/* notify the main thread that the progression thread is ready */
	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
	running = 1;
	STARPU_PTHREAD_COND_SIGNAL(&progress_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

	while (1)
	{
		struct callback_lfstack_cell_s* c = callback_lfstack_pop(&callback_stack);
		int err=0;

		if(running || nb_pending_requests>0)
		{
			/* shall we block ? */
			err = starpu_sem_wait(&callback_sem);
			//running nb_pending_requests can change while waiting
		}
		if(c==NULL)
		{
			c = callback_lfstack_pop(&callback_stack);
			if (c == NULL)
			{
				if(running && nb_pending_requests>0)
				{
					STARPU_ASSERT_MSG(c!=NULL, "Callback thread awakened without callback ready with error %d.",err);
				}
				else
				{
					if (nb_pending_requests==0)
						break;
				}
				continue;
			}
		}

		c->req->callback(c->req->callback_arg);
		if (c->req->detached)
		{
			_starpu_mpi_request_try_end(c->req, 0);
		}
		else
		{
			c->req->completed=1;
			piom_cond_signal(&(c->req->backend->req_cond), REQ_FINALIZED);
		}

		free(c);
	}


	/** Now, shutting down MPI **/


	STARPU_ASSERT_MSG(callback_lfstack_pop(&callback_stack)==NULL, "List of callback not empty.");
	STARPU_ASSERT_MSG(nb_pending_requests==0, "Request still pending.");

	/* We cannot rely on _starpu_mpi_use_coop_sends to shutdown mcast:
	 * coops can be disabled with starpu_mpi_coop_sends_set_use() after
	 * initialiazation of mcast. */
	if (nmad_mcast_started)
	{
		_starpu_mpi_nmad_coop_shutdown();
	}

#ifdef STARPU_USE_FXT
	_starpu_mpi_fxt_shutdown();
#endif

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

int _starpu_mpi_progress_init(struct _starpu_mpi_argc_argv *argc_argv)
{
	STARPU_PTHREAD_MUTEX_INIT(&progress_mutex, NULL);
	STARPU_PTHREAD_COND_INIT(&progress_cond, NULL);

	STARPU_PTHREAD_MUTEX_INIT(&mpi_wait_for_all_running_mutex, NULL);
	STARPU_PTHREAD_COND_INIT(&mpi_wait_for_all_running_cond, NULL);

	starpu_sem_init(&callback_sem, 0, 0);
	running = 0;

	_starpu_mpi_env_init();

	/* This function calls MPI_Init_thread if needed, and it initializes internal NMAD/Pioman variables,
	 * required for piom_ltask_set_bound_thread_indexes() */
	_starpu_mpi_do_initialize(argc_argv);

	if (!_starpu_mpi_nobind && _starpu_mpi_thread_cpuid < 0)
	{
		_starpu_mpi_thread_cpuid = starpu_get_next_bindid(STARPU_THREAD_ACTIVE, NULL, 0);
	}

	callback_lfstack_init(&callback_stack);

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
		starpu_progression_hook_register((void *)&piom_ltask_schedule, (void *)&polling_point_prog);
	}

	if(polling_point_idle)
	{
		starpu_idle_hook_register((void *)&piom_ltask_schedule, (void *)&polling_point_idle);
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
	if (!running)
	{
		_STARPU_ERROR("The progress thread was not launched. Was StarPU successfully initialized?\n");
	}

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

	STARPU_PTHREAD_MUTEX_DESTROY(&mpi_wait_for_all_running_mutex);
	STARPU_PTHREAD_COND_DESTROY(&mpi_wait_for_all_running_cond);
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
