/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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
#include <starpu_mpi.h>
#include <starpu_mpi_datatype.h>
#include <starpu_mpi_private.h>
#include <starpu_profiling.h>
#include <starpu_mpi_stats.h>
#include <starpu_mpi_insert_task.h>
#include <common/config.h>
#include <common/thread.h>

static void _starpu_mpi_submit_new_mpi_request(void *arg);
static void _starpu_mpi_handle_request_termination(struct _starpu_mpi_req *req);
#ifdef STARPU_VERBOSE
static char *_starpu_mpi_request_type(enum _starpu_mpi_request_type request_type);
#endif
static struct _starpu_mpi_req *_starpu_mpi_isend_common(starpu_data_handle_t data_handle,
							int dest, int mpi_tag, MPI_Comm comm,
							unsigned detached, void (*callback)(void *), void *arg);
static struct _starpu_mpi_req *_starpu_mpi_irecv_common(starpu_data_handle_t data_handle, int source, int mpi_tag, MPI_Comm comm, unsigned detached, void (*callback)(void *), void *arg);
static void _starpu_mpi_handle_detached_request(struct _starpu_mpi_req *req);

/* The list of requests that have been newly submitted by the application */
static struct _starpu_mpi_req_list *new_requests;

/* The list of detached requests that have already been submitted to MPI */
static struct _starpu_mpi_req_list *detached_requests;
static starpu_pthread_mutex_t detached_requests_mutex;

/* Condition to wake up progression thread */
static starpu_pthread_cond_t cond_progression;
/* Condition to wake up waiting for all current MPI requests to finish */
static starpu_pthread_cond_t cond_finished;
static starpu_pthread_mutex_t mutex;
static starpu_pthread_t progress_thread;
static int running = 0;

/* Count requests posted by the application and not yet submitted to MPI, i.e pushed into the new_requests list */
static starpu_pthread_mutex_t mutex_posted_requests;
static int posted_requests = 0, newer_requests, barrier_running = 0;

#define _STARPU_MPI_INC_POSTED_REQUESTS(value) { _STARPU_PTHREAD_MUTEX_LOCK(&mutex_posted_requests); posted_requests += value; _STARPU_PTHREAD_MUTEX_UNLOCK(&mutex_posted_requests); }

struct _starpu_mpi_envelope
{
	ssize_t psize;
	int mpi_tag;
};

struct _starpu_mpi_copy_handle
{
	starpu_data_handle_t handle;
	struct _starpu_mpi_envelope *env;
	int mpi_tag;
	UT_hash_handle hh;
};

 /********************************************************/
 /*                                                      */
 /*  Hashmap's requests functionalities                  */
 /*                                                      */
 /********************************************************/

static struct _starpu_mpi_req *_starpu_mpi_req_hashmap = NULL;
static struct _starpu_mpi_copy_handle *_starpu_mpi_copy_handle_hashmap = NULL;

static struct _starpu_mpi_req* find_req(int mpi_tag)
{
	struct _starpu_mpi_req* req; // = malloc(sizeof(struct _starpu_mpi_req));

	HASH_FIND_INT(_starpu_mpi_req_hashmap, &mpi_tag, req);

	return req;
}

static void add_req(struct _starpu_mpi_req *req)
{
	struct _starpu_mpi_req *test_req;

	test_req = find_req(req->mpi_tag);

	if (test_req == NULL)
	{
		HASH_ADD_INT(_starpu_mpi_req_hashmap, mpi_tag, req);
		_STARPU_MPI_DEBUG(3, "Adding request %p with tag %d in the hashmap. \n", req, req->mpi_tag);
	}
	else
	{
		_STARPU_MPI_DEBUG(3, "Error add_req : request %p with tag %d already in the hashmap. \n", req, req->mpi_tag);
		int seq_const = starpu_data_get_sequential_consistency_flag(req->data_handle);
		if (seq_const)
		{
			STARPU_ASSERT_MSG(!test_req, "Error add_req : request %p with tag %d wanted to be added to the hashmap, while another request %p with the same tag is already in it. \n Sequential consistency is activated : this is not supported by StarPU.", req, req->mpi_tag, test_req);
		}
		else
		{
			STARPU_ASSERT_MSG(!test_req, "Error add_req : request %p with tag %d wanted to be added to the hashmap, while another request %p with the same tag is already in it. \n Sequential consistency isn't activated for this handle : you should want to add dependencies between requests for which the sequential consistency is deactivated.", req, req->mpi_tag, test_req);
		}
	}
}

static void delete_req(struct _starpu_mpi_req *req)
{
	struct _starpu_mpi_req *test_req;

	test_req = find_req(req->mpi_tag);

	if (test_req != NULL)
	{
		HASH_DEL(_starpu_mpi_req_hashmap, req);
		_STARPU_MPI_DEBUG(3, "Deleting request %p with tag %d from the hashmap. \n", req, req->mpi_tag);
	}
	else
	{
		_STARPU_MPI_DEBUG(3, "Warning delete_req : request %p with tag %d isn't in the hashmap. \n", req, req->mpi_tag);
	}
}

static struct _starpu_mpi_copy_handle* find_chandle(int mpi_tag)
{
	struct _starpu_mpi_copy_handle* chandle;

	HASH_FIND_INT(_starpu_mpi_copy_handle_hashmap, &mpi_tag, chandle);

	return chandle;
}

static void add_chandle(struct _starpu_mpi_copy_handle *chandle)
{
	struct _starpu_mpi_copy_handle *test_chandle;

	test_chandle = find_chandle(chandle->mpi_tag);

	if (test_chandle == NULL)
	{
		HASH_ADD_INT(_starpu_mpi_copy_handle_hashmap, mpi_tag, chandle);
		_STARPU_MPI_DEBUG(3, "Adding copied handle %p with tag %d in the hashmap. \n", chandle, chandle->mpi_tag);
	}
	else
	{
		_STARPU_MPI_DEBUG(3, "Error add_chandle : copied handle %p with tag %d already in the hashmap. \n", chandle, chandle->mpi_tag);
		STARPU_ASSERT(test_chandle != NULL);
	}
}

static void delete_chandle(struct _starpu_mpi_copy_handle *chandle)
{
	struct _starpu_mpi_copy_handle *test_chandle;

	test_chandle = find_chandle(chandle->mpi_tag);

	if (test_chandle != NULL)
	{
		HASH_DEL(_starpu_mpi_copy_handle_hashmap, chandle);
		_STARPU_MPI_DEBUG(3, "Deleting copied handle %p with tag %d from the hashmap. \n", chandle, chandle->mpi_tag);
	}
	else
	{
		_STARPU_MPI_DEBUG(3, "Warning delete_chandle : copied handle %p with tag %d isn't in the hashmap. \n", chandle, chandle->mpi_tag);
	}
}

/********************************************************/
/*                                                      */
/*  Send/Receive functionalities                        */
/*                                                      */
/********************************************************/

static struct _starpu_mpi_req *_starpu_mpi_isend_irecv_common(starpu_data_handle_t data_handle,
							      int srcdst, int mpi_tag, MPI_Comm comm,
							      unsigned detached, void (*callback)(void *), void *arg,
							      enum _starpu_mpi_request_type request_type, void (*func)(struct _starpu_mpi_req *),
							      enum starpu_access_mode mode)
{

	_STARPU_MPI_LOG_IN();
	struct _starpu_mpi_req *req = calloc(1, sizeof(struct _starpu_mpi_req));
	STARPU_ASSERT_MSG(req, "Invalid request");

	_STARPU_MPI_INC_POSTED_REQUESTS(1);

	/* Initialize the request structure */
	req->submitted = 0;
	req->completed = 0;
	_STARPU_PTHREAD_MUTEX_INIT(&req->req_mutex, NULL);
	_STARPU_PTHREAD_COND_INIT(&req->req_cond, NULL);

	req->request_type = request_type;
	req->user_datatype = -1;
	req->count = -1;
	req->data_handle = data_handle;
	req->srcdst = srcdst;
	req->mpi_tag = mpi_tag;
	req->comm = comm;

	req->detached = detached;
	req->callback = callback;
	req->callback_arg = arg;

	req->func = func;

	/* Asynchronously request StarPU to fetch the data in main memory: when
	 * it is available in main memory, _starpu_mpi_submit_new_mpi_request(req) is called and
	 * the request is actually submitted */
	starpu_data_acquire_cb(data_handle, mode, _starpu_mpi_submit_new_mpi_request, (void *)req);

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

	STARPU_ASSERT_MSG(req->ptr, "Pointer containing data to send is invalid");

	_STARPU_MPI_DEBUG(2, "post MPI isend request %p type %s tag %d src %d data %p datasize %ld ptr %p datatype '%s' count %d user_datatype %d \n", req, _starpu_mpi_request_type(req->request_type), req->mpi_tag, req->srcdst, req->data_handle, starpu_handle_get_size(req->data_handle), req->ptr, _starpu_mpi_datatype(req->datatype), (int)req->count, req->user_datatype);

	_starpu_mpi_comm_amounts_inc(req->comm, req->srcdst, req->datatype, req->count);

	TRACE_MPI_ISEND_SUBMIT_BEGIN(req->srcdst, req->mpi_tag, 0);

	req->ret = MPI_Isend(req->ptr, req->count, req->datatype, req->srcdst, _starpu_mpi_tag, req->comm, &req->request);
	STARPU_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Isend returning %d", req->ret);

	TRACE_MPI_ISEND_SUBMIT_END(req->srcdst, req->mpi_tag, 0);

	/* somebody is perhaps waiting for the MPI request to be posted */
	_STARPU_PTHREAD_MUTEX_LOCK(&req->req_mutex);
	req->submitted = 1;
	_STARPU_PTHREAD_COND_BROADCAST(&req->req_cond);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&req->req_mutex);

	_starpu_mpi_handle_detached_request(req);

	_STARPU_MPI_LOG_OUT();
}

static void _starpu_mpi_isend_size_func(struct _starpu_mpi_req *req)
{
	_starpu_mpi_handle_allocate_datatype(req->data_handle, &req->datatype, &req->user_datatype);

	struct _starpu_mpi_envelope* env = calloc(1,sizeof(struct _starpu_mpi_envelope));

	env->mpi_tag = req->mpi_tag;

	if (req->user_datatype == 0)
	{
		req->count = 1;
		req->ptr = starpu_handle_get_local_ptr(req->data_handle);

		env->psize = (ssize_t)req->count;

		_STARPU_MPI_DEBUG(1, "Post MPI isend count (%ld) datatype_size %d request to %d with tag %d\n",req->count,starpu_handle_get_size(req->data_handle),req->srcdst, _starpu_mpi_tag);
		MPI_Isend(env, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->srcdst, _starpu_mpi_tag, req->comm, &req->size_req);
	}
	else
	{
		int ret;

 		// Do not pack the data, just try to find out the size
		starpu_handle_pack_data(req->data_handle, NULL, &(env->psize));

		if (env->psize != -1)
 		{
 			// We already know the size of the data, let's send it to overlap with the packing of the data
			_STARPU_MPI_DEBUG(1, "Sending size %ld (%ld %s) with tag %d to node %d (first call to pack)\n", env->psize, sizeof(req->count), _starpu_mpi_datatype(MPI_BYTE), _starpu_mpi_tag, req->srcdst);
			req->count = env->psize;
			ret = MPI_Isend(env, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->srcdst, _starpu_mpi_tag, req->comm, &req->size_req);
			STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "when sending size, MPI_Isend returning %d", ret);
 		}

 		// Pack the data
 		starpu_handle_pack_data(req->data_handle, &req->ptr, &req->count);
		if (env->psize == -1)
 		{
 			// We know the size now, let's send it
			_STARPU_MPI_DEBUG(1, "Sending size %ld (%ld %s) with tag %d to node %d (second call to pack)\n", env->psize, sizeof(req->count), _starpu_mpi_datatype(MPI_BYTE), _starpu_mpi_tag, req->srcdst);
			ret = MPI_Isend(env, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->srcdst, _starpu_mpi_tag, req->comm, &req->size_req);
			STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "when sending size, MPI_Isend returning %d", ret);
 		}
 		else
 		{
 			// We check the size returned with the 2 calls to pack is the same
			STARPU_ASSERT_MSG(req->count == env->psize, "Calls to pack_data returned different sizes %ld != %ld", req->count, env->psize);
 		}
		// We can send the data now
	}
	_starpu_mpi_isend_data_func(req);
}

static struct _starpu_mpi_req *_starpu_mpi_isend_common(starpu_data_handle_t data_handle,
							int dest, int mpi_tag, MPI_Comm comm,
							unsigned detached, void (*callback)(void *), void *arg)
{
	return _starpu_mpi_isend_irecv_common(data_handle, dest, mpi_tag, comm, detached, callback, arg, SEND_REQ, _starpu_mpi_isend_size_func, STARPU_R);
}

int starpu_mpi_isend(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int dest, int mpi_tag, MPI_Comm comm)
{
	_STARPU_MPI_LOG_IN();
	STARPU_ASSERT_MSG(public_req, "starpu_mpi_isend needs a valid starpu_mpi_req");

	struct _starpu_mpi_req *req;
	req = _starpu_mpi_isend_common(data_handle, dest, mpi_tag, comm, 0, NULL, NULL);

	STARPU_ASSERT_MSG(req, "Invalid return for _starpu_mpi_isend_common");
	*public_req = req;

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_isend_detached(starpu_data_handle_t data_handle,
			      int dest, int mpi_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();
	_starpu_mpi_isend_common(data_handle, dest, mpi_tag, comm, 1, callback, arg);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_send(starpu_data_handle_t data_handle, int dest, int mpi_tag, MPI_Comm comm)
{
	starpu_mpi_req req;
	MPI_Status status;

	_STARPU_MPI_LOG_IN();
	memset(&status, 0, sizeof(MPI_Status));

	starpu_mpi_isend(data_handle, &req, dest, mpi_tag, comm);
	starpu_mpi_wait(&req, &status);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

/********************************************************/
/*                                                      */
/*  receive functionalities                             */
/*                                                      */
/********************************************************/

static void _starpu_mpi_irecv_data_func(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	STARPU_ASSERT_MSG(req->ptr, "Invalid pointer to receive data");

	_STARPU_MPI_DEBUG(2, "post MPI irecv request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d user_datatype %d \n", req, _starpu_mpi_request_type(req->request_type), req->mpi_tag, req->srcdst, req->data_handle, req->ptr, _starpu_mpi_datatype(req->datatype), (int)req->count, req->user_datatype);

	TRACE_MPI_IRECV_SUBMIT_BEGIN(req->srcdst, req->mpi_tag);

	req->ret = MPI_Irecv(req->ptr, req->count, req->datatype, req->srcdst, _starpu_mpi_tag, req->comm, &req->request);
	STARPU_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_IRecv returning %d", req->ret);

	TRACE_MPI_IRECV_SUBMIT_END(req->srcdst, req->mpi_tag);

	/* somebody is perhaps waiting for the MPI request to be posted */
	_STARPU_PTHREAD_MUTEX_LOCK(&req->req_mutex);
	req->submitted = 1;
	_STARPU_PTHREAD_COND_BROADCAST(&req->req_cond);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&req->req_mutex);

	_starpu_mpi_handle_detached_request(req);

	_STARPU_MPI_LOG_OUT();
}

static struct _starpu_mpi_req *_starpu_mpi_irecv_common(starpu_data_handle_t data_handle, int source, int mpi_tag, MPI_Comm comm, unsigned detached, void (*callback)(void *), void *arg)
{
	return _starpu_mpi_isend_irecv_common(data_handle, source, mpi_tag, comm, detached, callback, arg, RECV_REQ, _starpu_mpi_irecv_data_func, STARPU_W);
}

int starpu_mpi_irecv(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int source, int mpi_tag, MPI_Comm comm)
{
	_STARPU_MPI_LOG_IN();
	STARPU_ASSERT_MSG(public_req, "starpu_mpi_irecv needs a valid starpu_mpi_req");

	struct _starpu_mpi_req *req;
	req = _starpu_mpi_irecv_common(data_handle, source, mpi_tag, comm, 0, NULL, NULL);

	STARPU_ASSERT_MSG(req, "Invalid return for _starpu_mpi_irecv_common");
	*public_req = req;

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_irecv_detached(starpu_data_handle_t data_handle, int source, int mpi_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();
	_starpu_mpi_irecv_common(data_handle, source, mpi_tag, comm, 1, callback, arg);
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

static void _starpu_mpi_wait_func(struct _starpu_mpi_req *waiting_req)
{
	_STARPU_MPI_LOG_IN();
	/* Which is the mpi request we are waiting for ? */
	struct _starpu_mpi_req *req = waiting_req->other_request;

	TRACE_MPI_UWAIT_BEGIN(req->srcdst, req->mpi_tag);

	req->ret = MPI_Wait(&req->request, waiting_req->status);
	STARPU_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Wait returning %d", req->ret);

	TRACE_MPI_UWAIT_END(req->srcdst, req->mpi_tag);

	_starpu_mpi_handle_request_termination(req);
	_STARPU_MPI_LOG_OUT();
}

int starpu_mpi_wait(starpu_mpi_req *public_req, MPI_Status *status)
{
	_STARPU_MPI_LOG_IN();
	int ret;
	struct _starpu_mpi_req *waiting_req = calloc(1, sizeof(struct _starpu_mpi_req));
	STARPU_ASSERT_MSG(waiting_req, "Allocation failed");
	struct _starpu_mpi_req *req = *public_req;

	_STARPU_MPI_INC_POSTED_REQUESTS(1);

	/* We cannot try to complete a MPI request that was not actually posted
	 * to MPI yet. */
	_STARPU_PTHREAD_MUTEX_LOCK(&(req->req_mutex));
	while (!(req->submitted))
		_STARPU_PTHREAD_COND_WAIT(&(req->req_cond), &(req->req_mutex));
	_STARPU_PTHREAD_MUTEX_UNLOCK(&(req->req_mutex));

	/* Initialize the request structure */
	_STARPU_PTHREAD_MUTEX_INIT(&(waiting_req->req_mutex), NULL);
	_STARPU_PTHREAD_COND_INIT(&(waiting_req->req_cond), NULL);
	waiting_req->status = status;
	waiting_req->other_request = req;
	waiting_req->func = _starpu_mpi_wait_func;
	waiting_req->request_type = WAIT_REQ;

	_starpu_mpi_submit_new_mpi_request(waiting_req);

	/* We wait for the MPI request to finish */
	_STARPU_PTHREAD_MUTEX_LOCK(&req->req_mutex);
	while (!req->completed)
		_STARPU_PTHREAD_COND_WAIT(&req->req_cond, &req->req_mutex);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&req->req_mutex);

	ret = req->ret;

	/* The internal request structure was automatically allocated */
	*public_req = NULL;
	free(req);

	free(waiting_req);
	_STARPU_MPI_LOG_OUT();
	return ret;
}

/********************************************************/
/*                                                      */
/*  Test functionalities                                */
/*                                                      */
/********************************************************/

static void _starpu_mpi_test_func(struct _starpu_mpi_req *testing_req)
{
	_STARPU_MPI_LOG_IN();
	/* Which is the mpi request we are testing for ? */
	struct _starpu_mpi_req *req = testing_req->other_request;

	_STARPU_MPI_DEBUG(2, "Test request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d user_datatype %d \n",
			  req, _starpu_mpi_request_type(req->request_type), req->mpi_tag, req->srcdst, req->data_handle, req->ptr, _starpu_mpi_datatype(req->datatype), (int)req->count, req->user_datatype);

	TRACE_MPI_UTESTING_BEGIN(req->srcdst, req->mpi_tag);

	req->ret = MPI_Test(&req->request, testing_req->flag, testing_req->status);
	STARPU_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Test returning %d", req->ret);

	TRACE_MPI_UTESTING_END(req->srcdst, req->mpi_tag);

	if (*testing_req->flag)
	{
		testing_req->ret = req->ret;
		_starpu_mpi_handle_request_termination(req);
	}

	_STARPU_PTHREAD_MUTEX_LOCK(&testing_req->req_mutex);
	testing_req->completed = 1;
	_STARPU_PTHREAD_COND_SIGNAL(&testing_req->req_cond);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&testing_req->req_mutex);
	_STARPU_MPI_LOG_OUT();
}

int starpu_mpi_test(starpu_mpi_req *public_req, int *flag, MPI_Status *status)
{
	_STARPU_MPI_LOG_IN();
	int ret = 0;

	STARPU_ASSERT_MSG(public_req, "starpu_mpi_test needs a valid starpu_mpi_req");

	struct _starpu_mpi_req *req = *public_req;

	STARPU_ASSERT_MSG(!req->detached, "MPI_Test cannot be called on a detached request");

	_STARPU_PTHREAD_MUTEX_LOCK(&req->req_mutex);
	unsigned submitted = req->submitted;
	_STARPU_PTHREAD_MUTEX_UNLOCK(&req->req_mutex);

	if (submitted)
	{
		struct _starpu_mpi_req *testing_req = calloc(1, sizeof(struct _starpu_mpi_req));
		STARPU_ASSERT_MSG(testing_req, "allocation failed");
		//		memset(testing_req, 0, sizeof(struct _starpu_mpi_req));

		/* Initialize the request structure */
		_STARPU_PTHREAD_MUTEX_INIT(&(testing_req->req_mutex), NULL);
		_STARPU_PTHREAD_COND_INIT(&(testing_req->req_cond), NULL);
		testing_req->flag = flag;
		testing_req->status = status;
		testing_req->other_request = req;
		testing_req->func = _starpu_mpi_test_func;
		testing_req->completed = 0;
		testing_req->request_type = TEST_REQ;

		_STARPU_MPI_INC_POSTED_REQUESTS(1);
		_starpu_mpi_submit_new_mpi_request(testing_req);

		/* We wait for the test request to finish */
		_STARPU_PTHREAD_MUTEX_LOCK(&(testing_req->req_mutex));
		while (!(testing_req->completed))
			_STARPU_PTHREAD_COND_WAIT(&(testing_req->req_cond), &(testing_req->req_mutex));
		_STARPU_PTHREAD_MUTEX_UNLOCK(&(testing_req->req_mutex));

		ret = testing_req->ret;

		if (*(testing_req->flag))
		{
			/* The request was completed so we free the internal
			 * request structure which was automatically allocated
			 * */
			*public_req = NULL;
			free(req);
		}

		free(testing_req);
	}
	else
	{
		*flag = 0;
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

	barrier_req->ret = MPI_Barrier(barrier_req->comm);
	STARPU_ASSERT_MSG(barrier_req->ret == MPI_SUCCESS, "MPI_Barrier returning %d", barrier_req->ret);

	_starpu_mpi_handle_request_termination(barrier_req);
	_STARPU_MPI_LOG_OUT();
}

int starpu_mpi_barrier(MPI_Comm comm)
{
	_STARPU_MPI_LOG_IN();
	int ret;
	struct _starpu_mpi_req *barrier_req = calloc(1, sizeof(struct _starpu_mpi_req));
	STARPU_ASSERT_MSG(barrier_req, "allocation failed");

	/* First wait for *both* all tasks and MPI requests to finish, in case
	 * some tasks generate MPI requests, MPI requests generate tasks, etc.
	 */
	_STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	STARPU_ASSERT_MSG(!barrier_running, "Concurrent starpu_mpi_barrier is not implemented, even on different communicators");
	barrier_running = 1;
	do
	{
		while (posted_requests)
			/* Wait for all current MPI requests to finish */
			_STARPU_PTHREAD_COND_WAIT(&cond_finished, &mutex);
		/* No current request, clear flag */
		newer_requests = 0;
		_STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		/* Now wait for all tasks */
		starpu_task_wait_for_all();
		_STARPU_PTHREAD_MUTEX_LOCK(&mutex);
		/* Check newer_requests again, in case some MPI requests
		 * triggered by tasks completed and triggered tasks between
		 * wait_for_all finished and we take the lock */
	} while (posted_requests || newer_requests);
	barrier_running = 0;
	_STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	/* Initialize the request structure */
	_STARPU_PTHREAD_MUTEX_INIT(&(barrier_req->req_mutex), NULL);
	_STARPU_PTHREAD_COND_INIT(&(barrier_req->req_cond), NULL);
	barrier_req->func = _starpu_mpi_barrier_func;
	barrier_req->request_type = BARRIER_REQ;
	barrier_req->comm = comm;

	_STARPU_MPI_INC_POSTED_REQUESTS(1);
	_starpu_mpi_submit_new_mpi_request(barrier_req);

	/* We wait for the MPI request to finish */
	_STARPU_PTHREAD_MUTEX_LOCK(&barrier_req->req_mutex);
	while (!barrier_req->completed)
		_STARPU_PTHREAD_COND_WAIT(&barrier_req->req_cond, &barrier_req->req_mutex);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&barrier_req->req_mutex);

	ret = barrier_req->ret;

	free(barrier_req);
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

static void _starpu_mpi_handle_request_termination(struct _starpu_mpi_req *req)
{
	int ret;

	_STARPU_MPI_LOG_IN();

	_STARPU_MPI_DEBUG(2, "complete MPI request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d user_datatype %d \n",
			  req, _starpu_mpi_request_type(req->request_type), req->mpi_tag, req->srcdst, req->data_handle, req->ptr, _starpu_mpi_datatype(req->datatype), (int)req->count, req->user_datatype);

	if (req->request_type == RECV_REQ || req->request_type == SEND_REQ)
	{
		if (req->user_datatype == 1)
		{
			if (req->request_type == SEND_REQ)
			{
				// We already know the request to send the size is completed, we just call MPI_Test to make sure that the request object is deallocated
				MPI_Status status;
				int flag;
				ret = MPI_Test(&req->size_req, &flag, &status);
				STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Test returning %d", ret);
				STARPU_ASSERT_MSG(flag, "MPI_Test returning flag %d", flag);
			}
			if (req->request_type == RECV_REQ)
				// req->ptr is freed by starpu_handle_unpack_data
				starpu_handle_unpack_data(req->data_handle, req->ptr, req->count);
			else
				free(req->ptr);
		}
		else
		{
			struct _starpu_mpi_copy_handle *chandle = find_chandle(starpu_data_get_tag(req->data_handle));
			if (chandle && (req->data_handle != chandle->handle))
			{
				_STARPU_MPI_DEBUG(3, "Handling deleting of copy_handle structure from the hashmap..\n");
				delete_chandle(chandle);
				free(chandle);
			}
			else
			{
				_starpu_mpi_handle_free_datatype(req->data_handle, &req->datatype);
			}
		}
		starpu_data_release(req->data_handle);
	}

	/* Execute the specified callback, if any */
	if (req->callback)
		req->callback(req->callback_arg);

	/* tell anyone potentially waiting on the request that it is
	 * terminated now */
	_STARPU_PTHREAD_MUTEX_LOCK(&req->req_mutex);
	req->completed = 1;
	_STARPU_PTHREAD_COND_BROADCAST(&req->req_cond);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&req->req_mutex);
	_STARPU_MPI_LOG_OUT();
}

struct _starpu_mpi_copy_cb_args
{
	starpu_data_handle_t data_handle;
	starpu_data_handle_t copy_handle;
	struct _starpu_mpi_req *req;
};

static void _starpu_mpi_copy_cb(void* arg)
{
	struct _starpu_mpi_copy_cb_args *args = arg;

	struct starpu_data_interface_ops *itf = starpu_handle_get_interface(args->copy_handle);
	void* itf_src = starpu_data_get_interface_on_node(args->copy_handle,0);
	void* itf_dst = starpu_data_get_interface_on_node(args->data_handle,0);

	if (!itf->copy_methods->ram_to_ram)
	{
		_STARPU_MPI_DEBUG(3, "Initiating any_to_any copy..\n");
		itf->copy_methods->any_to_any(itf_src, 0, itf_dst, 0, NULL);
	}
	else
	{
		_STARPU_MPI_DEBUG(3, "Initiating ram_to_ram copy..\n");
		itf->copy_methods->ram_to_ram(itf_src, 0, itf_dst, 0);
	}

	_STARPU_MPI_DEBUG(3, "Done, handling release of copy_handle..\n");
	starpu_data_release(args->copy_handle);

	_STARPU_MPI_DEBUG(3, "Done, handling unregister of copy_handle..\n");
	starpu_data_unregister_submit(args->copy_handle);

	_STARPU_MPI_DEBUG(3, "Done, handling request %p termination of the already received request\n",args->req);
	_starpu_mpi_handle_request_termination(args->req);

	free(args);
}

static void _starpu_mpi_submit_new_mpi_request(void *arg)
{
	_STARPU_MPI_LOG_IN();
	struct _starpu_mpi_req *req = arg;

	_STARPU_MPI_INC_POSTED_REQUESTS(-1);

	_STARPU_PTHREAD_MUTEX_LOCK(&mutex);

	if (req->request_type == RECV_REQ)
	{
		/* test whether the receive request has already been submitted internally by StarPU-MPI*/
		struct _starpu_mpi_copy_handle *chandle = find_chandle(req->mpi_tag);

		/* Case : the request has already been submitted internally by StarPU.
		 * We'll asynchronously ask a Read permission over the temporary handle, so as when
		 * the internal receive will be over, the _starpu_mpi_copy_cb function will be called to
		 * bring the data back to the original data handle associated to the request.*/
		if (chandle && (req->data_handle != chandle->handle))
		{
			_STARPU_MPI_DEBUG(3, "The RECV request %p with tag %d has already been received, copying previously received data into handle's pointer..\n", req, req->mpi_tag);

			struct _starpu_mpi_copy_cb_args *cb_args = malloc(sizeof(struct _starpu_mpi_copy_cb_args));
			cb_args->data_handle = req->data_handle;
			cb_args->copy_handle = chandle->handle;
			cb_args->req = req;

			_STARPU_MPI_DEBUG(3, "Calling data_acquire_cb on starpu_mpi_copy_cb..\n");
			starpu_data_acquire_cb(chandle->handle,STARPU_R,_starpu_mpi_copy_cb,(void*) cb_args);
		}
		else
		{
			/* Case : the request is the internal receive request submitted by StarPU-MPI to receive
			 * incoming data without a matching pending receive already submitted by the application.
			 * We immediately allocate the pointer associated to the data_handle, and pushing it into
			 * the list of new_requests, so as the real MPI request can be submitted before the next
			 * submission of the envelope-catching request. */
			if (chandle && (req->data_handle == chandle->handle))
			{
				_starpu_mpi_handle_allocate_datatype(req->data_handle, &req->datatype, &req->user_datatype);
				if (req->user_datatype == 0)
				{
					req->count = 1;
					req->ptr = starpu_handle_get_local_ptr(req->data_handle);
				}
				else
				{
					req->count = chandle->env->psize;
					req->ptr = malloc(req->count);

					STARPU_ASSERT_MSG(req->ptr, "cannot allocate message of size %ld\n", req->count);
				}

				_starpu_mpi_req_list_push_front(new_requests, req);

				_STARPU_MPI_DEBUG(3, "Pushing internal starpu_mpi_irecv request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d user_datatype %d \n", req, _starpu_mpi_request_type(req->request_type), req->mpi_tag, req->srcdst, req->data_handle, req->ptr, _starpu_mpi_datatype(req->datatype), (int)req->count, req->user_datatype);
			}
			/* Case : a classic receive request with no send received earlier than expected.
			 * We just add the pending receive request to the requests' hashmap. */
			else
			{
				add_req(req);
			}

			newer_requests = 1;
			_STARPU_PTHREAD_COND_BROADCAST(&cond_progression);
		}
	}
	else
	{
		_starpu_mpi_req_list_push_front(new_requests, req);

		newer_requests = 1;
		_STARPU_MPI_DEBUG(3, "Pushing new request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d user_datatype %d \n",
				  req, _starpu_mpi_request_type(req->request_type), req->mpi_tag, req->srcdst, req->data_handle, req->ptr, _starpu_mpi_datatype(req->datatype), (int)req->count, req->user_datatype);
		_STARPU_PTHREAD_COND_BROADCAST(&cond_progression);
	}

	_STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
	_STARPU_MPI_LOG_OUT();
}

#ifdef STARPU_MPI_ACTIVITY
static unsigned _starpu_mpi_progression_hook_func(void *arg __attribute__((unused)))
{
	unsigned may_block = 1;

	_STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	if (!_starpu_mpi_req_list_empty(detached_requests))
	{
		_STARPU_PTHREAD_COND_SIGNAL(&cond_progression);
		may_block = 0;
	}
	_STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	return may_block;
}
#endif /* STARPU_MPI_ACTIVITY */

static void _starpu_mpi_test_detached_requests(void)
{
	_STARPU_MPI_LOG_IN();
	int flag;
	MPI_Status status;
	struct _starpu_mpi_req *req, *next_req;

	_STARPU_PTHREAD_MUTEX_LOCK(&detached_requests_mutex);

	for (req = _starpu_mpi_req_list_begin(detached_requests);
		req != _starpu_mpi_req_list_end(detached_requests);
		req = next_req)
	{
		next_req = _starpu_mpi_req_list_next(req);

		_STARPU_PTHREAD_MUTEX_UNLOCK(&detached_requests_mutex);

		//_STARPU_MPI_DEBUG(3, "Test detached request %p - mpitag %d - TYPE %s %d\n", &req->request, req->mpi_tag, _starpu_mpi_request_type(req->request_type), req->srcdst);
		req->ret = MPI_Test(&req->request, &flag, &status);

		STARPU_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Test returning %d", req->ret);

		if (flag)
		{
			if (req->request_type == RECV_REQ)
			{
				TRACE_MPI_IRECV_COMPLETE_BEGIN(req->srcdst, req->mpi_tag);
			}
			else if (req->request_type == SEND_REQ)
			{
				TRACE_MPI_ISEND_COMPLETE_BEGIN(req->srcdst, req->mpi_tag, 0);
			}

			_starpu_mpi_handle_request_termination(req);

			if (req->request_type == RECV_REQ)
			{
				TRACE_MPI_IRECV_COMPLETE_END(req->srcdst, req->mpi_tag);
			}
			else if (req->request_type == SEND_REQ)
			{
				TRACE_MPI_ISEND_COMPLETE_END(req->srcdst, req->mpi_tag, 0);
			}
		}

		_STARPU_PTHREAD_MUTEX_LOCK(&detached_requests_mutex);

		if (flag)
		{
			_starpu_mpi_req_list_erase(detached_requests, req);
			free(req);
		}

	}

	_STARPU_PTHREAD_MUTEX_UNLOCK(&detached_requests_mutex);
	_STARPU_MPI_LOG_OUT();
}

static void _starpu_mpi_handle_detached_request(struct _starpu_mpi_req *req)
{
	if (req->detached)
	{
		_STARPU_PTHREAD_MUTEX_LOCK(&mutex);
		_starpu_mpi_req_list_push_front(detached_requests, req);
		_STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

		starpu_wake_all_blocked_workers();

		/* put the submitted request into the list of pending requests
		 * so that it can be handled by the progression mechanisms */
		_STARPU_PTHREAD_MUTEX_LOCK(&mutex);
		_STARPU_PTHREAD_COND_SIGNAL(&cond_progression);
		_STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
	}
}

static void _starpu_mpi_handle_new_request(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();
	STARPU_ASSERT_MSG(req, "Invalid request");

	/* submit the request to MPI */
	_STARPU_MPI_DEBUG(2, "Handling new request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d user_datatype %d \n",
			  req, _starpu_mpi_request_type(req->request_type), req->mpi_tag, req->srcdst, req->data_handle, req->ptr, _starpu_mpi_datatype(req->datatype), (int)req->count, req->user_datatype);
	req->func(req);

	_STARPU_MPI_LOG_OUT();
}

struct _starpu_mpi_argc_argv
{
	int initialize_mpi;
	int *argc;
	char ***argv;
};

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

	if (argc_argv->initialize_mpi)
	{
		int thread_support;
		_STARPU_DEBUG("Calling MPI_Init_thread\n");
		if (MPI_Init_thread(argc_argv->argc, argc_argv->argv, MPI_THREAD_SERIALIZED, &thread_support) != MPI_SUCCESS)
		{
			_STARPU_ERROR("MPI_Init_thread failed\n");
		}
		_starpu_mpi_print_thread_level_support(thread_support, "_Init_thread level =");
	}
	else
	{
		int provided;
		MPI_Query_thread(&provided);
		_starpu_mpi_print_thread_level_support(provided, " has been initialized with");
	}

	{
	     int rank, worldsize;
	     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	     MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	     TRACE_MPI_START(rank, worldsize);
#ifdef STARPU_USE_FXT
	     starpu_set_profiling_id(rank);
#endif //STARPU_USE_FXT
	}

	/* notify the main thread that the progression thread is ready */
	_STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	running = 1;
	_STARPU_PTHREAD_COND_SIGNAL(&cond_progression);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	_STARPU_PTHREAD_MUTEX_LOCK(&mutex);

 	struct _starpu_mpi_envelope *recv_env = calloc(1,sizeof(struct _starpu_mpi_envelope));

 	MPI_Request header_req;
 	int header_req_submitted = 0;

	while (running || posted_requests || !(_starpu_mpi_req_list_empty(new_requests)) || !(_starpu_mpi_req_list_empty(detached_requests)))
	{
		/* shall we block ? */
		_STARPU_MPI_DEBUG(3, "HASH_COUNT(_starpu_mpi_req_hashmap) = %d\n",HASH_COUNT(_starpu_mpi_req_hashmap));
		unsigned block = _starpu_mpi_req_list_empty(new_requests) && (HASH_COUNT(_starpu_mpi_req_hashmap) == 0);

#ifndef STARPU_MPI_ACTIVITY
		block = block && _starpu_mpi_req_list_empty(detached_requests);
#endif /* STARPU_MPI_ACTIVITY */

		if (block)
		{
			_STARPU_MPI_DEBUG(3, "NO MORE REQUESTS TO HANDLE\n");

			TRACE_MPI_SLEEP_BEGIN();

			if (barrier_running)
				/* Tell mpi_barrier */
				_STARPU_PTHREAD_COND_SIGNAL(&cond_finished);
			_STARPU_PTHREAD_COND_WAIT(&cond_progression, &mutex);

			TRACE_MPI_SLEEP_END();
		}

		/* get one request */
		struct _starpu_mpi_req *req;
		while (!_starpu_mpi_req_list_empty(new_requests))
		{
			req = _starpu_mpi_req_list_pop_back(new_requests);

			/* handling a request is likely to block for a while
			 * (on a sync_data_with_mem call), we want to let the
			 * application submit requests in the meantime, so we
			 * release the lock. */
			_STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
			_starpu_mpi_handle_new_request(req);
			_STARPU_PTHREAD_MUTEX_LOCK(&mutex);
		}

		/* If there is no currently submitted header_req submitted to catch envelopes from senders, and there is some pending receive
		 * requests in our side, we resubmit a header request. */
		if ((HASH_COUNT(_starpu_mpi_req_hashmap) > 0) && (header_req_submitted == 0) && (HASH_COUNT(_starpu_mpi_copy_handle_hashmap) == 0))
		{
			MPI_Irecv(recv_env, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, MPI_ANY_SOURCE, _starpu_mpi_tag, MPI_COMM_WORLD, &header_req);

			_STARPU_MPI_DEBUG(3, "Submit of header_req OK!\n");
			header_req_submitted = 1;
		}

		/* test whether there are some terminated "detached request" */
		_STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		_starpu_mpi_test_detached_requests();
		_STARPU_PTHREAD_MUTEX_LOCK(&mutex);

		if (header_req_submitted == 1)
		{
			int flag,res;
			MPI_Status status;
			_STARPU_MPI_DEBUG(3, "Test of header_req\n");

			/* test whether an envelope has arrived. */
			res = MPI_Test(&header_req, &flag, &status);
			STARPU_ASSERT(res == MPI_SUCCESS);

			if (flag)
			{
				_STARPU_MPI_DEBUG(3, "header_req received !\n");

				_STARPU_MPI_DEBUG(3, "Searching for request with tag %d, size %ld ..\n",recv_env->mpi_tag, recv_env->psize);

				struct _starpu_mpi_req *found_req = find_req(recv_env->mpi_tag);

				/* Case : a data will arrive before the matching receive has been submitted in our side of the application.
				 * We will allow a temporary handle to store the incoming data, by submitting a starpu_mpi_irecv_detached
				 * on this handle, and register this so as the StarPU-MPI layer can remember it.*/
				if (!found_req)
				{
					_STARPU_MPI_DEBUG(3, "Request with tag %d not found, creating a copy_handle to receive incoming data..\n",recv_env->mpi_tag);

					starpu_data_handle_t data_handle = NULL;

					while(!(data_handle))
					{
						data_handle = starpu_get_data_handle_from_tag(recv_env->mpi_tag);
					}
					STARPU_ASSERT(data_handle);

					struct _starpu_mpi_copy_handle* chandle = malloc(sizeof(struct _starpu_mpi_copy_handle));
					STARPU_ASSERT(chandle);

					chandle->mpi_tag = recv_env->mpi_tag;
					chandle->env = recv_env;
					starpu_data_register_same(&chandle->handle, data_handle);
					add_chandle(chandle);

					_STARPU_MPI_DEBUG(3, "Posting internal starpu_irecv_detached on copy_handle with tag %d from src %d ..\n", chandle->mpi_tag, status.MPI_SOURCE);

					res = starpu_mpi_irecv_detached(chandle->handle,status.MPI_SOURCE,chandle->mpi_tag,MPI_COMM_WORLD,NULL,NULL);
					STARPU_ASSERT(res == MPI_SUCCESS);

					_STARPU_MPI_DEBUG(3, "Success of starpu_irecv_detached on copy_handle with tag %d from src %d ..\n", chandle->mpi_tag, status.MPI_SOURCE);
				}
				/* Case : a matching receive has been found for the incoming data, we handle the correct allocation of the pointer associated to
				 * the data handle, then submit the corresponding receive with _starpu_mpi_handle_new_request. */
				else
				{
					_STARPU_MPI_DEBUG(3, "Found !\n");

					delete_req(found_req);

					_starpu_mpi_handle_allocate_datatype(found_req->data_handle, &found_req->datatype, &found_req->user_datatype);
					if (found_req->user_datatype == 0)
					{
						found_req->count = 1;
						found_req->ptr = starpu_handle_get_local_ptr(found_req->data_handle);
					}
					else
					{
						found_req->count = recv_env->psize;
						found_req->ptr = malloc(found_req->count);

						STARPU_ASSERT_MSG(found_req->ptr, "cannot allocate message of size %ld\n", found_req->count);
					}

					_STARPU_MPI_DEBUG(3, "Handling new request... \n");
					/* handling a request is likely to block for a while
					 * (on a sync_data_with_mem call), we want to let the
					 * application submit requests in the meantime, so we
					 * release the lock. */
					_STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
					_starpu_mpi_handle_new_request(found_req);
					_STARPU_PTHREAD_MUTEX_LOCK(&mutex);
				}
				header_req_submitted = 0;
			}
			else
			{
				_STARPU_MPI_DEBUG(3, "Nothing received, continue ..\n");
			}
		}
	}

	STARPU_ASSERT_MSG(_starpu_mpi_req_list_empty(detached_requests), "List of detached requests not empty");
	STARPU_ASSERT_MSG(_starpu_mpi_req_list_empty(new_requests), "List of new requests not empty");
	STARPU_ASSERT_MSG(posted_requests == 0, "Number of posted request is not zero");
	STARPU_ASSERT_MSG(HASH_COUNT(_starpu_mpi_req_hashmap) == 0, "Number of receive requests left is not zero");

	if (argc_argv->initialize_mpi)
	{
		_STARPU_MPI_DEBUG(3, "Calling MPI_Finalize()\n");
		MPI_Finalize();
	}

	_STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	free(argc_argv);
	free(recv_env);

	return NULL;
}

/********************************************************/
/*                                                      */
/*  (De)Initialization methods                          */
/*                                                      */
/********************************************************/

#ifdef STARPU_MPI_ACTIVITY
static int hookid = - 1;
#endif /* STARPU_MPI_ACTIVITY */

static void _starpu_mpi_add_sync_point_in_fxt(void)
{
#ifdef STARPU_USE_FXT
	int rank;
	int worldsize;
	int ret;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

	ret = MPI_Barrier(MPI_COMM_WORLD);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Barrier returning %d", ret);

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
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Bcast returning %d", ret);

	TRACE_MPI_BARRIER(rank, worldsize, random_number);

	_STARPU_MPI_DEBUG(3, "unique key %x\n", random_number);
#endif
}

static
int _starpu_mpi_initialize(int *argc, char ***argv, int initialize_mpi)
{
	_STARPU_PTHREAD_MUTEX_INIT(&mutex, NULL);
	_STARPU_PTHREAD_COND_INIT(&cond_progression, NULL);
	_STARPU_PTHREAD_COND_INIT(&cond_finished, NULL);
	new_requests = _starpu_mpi_req_list_new();

	_STARPU_PTHREAD_MUTEX_INIT(&detached_requests_mutex, NULL);
	detached_requests = _starpu_mpi_req_list_new();

	_STARPU_PTHREAD_MUTEX_INIT(&mutex_posted_requests, NULL);

	struct _starpu_mpi_argc_argv *argc_argv = malloc(sizeof(struct _starpu_mpi_argc_argv));
	argc_argv->initialize_mpi = initialize_mpi;
	argc_argv->argc = argc;
	argc_argv->argv = argv;

	_STARPU_PTHREAD_CREATE("MPI progress", &progress_thread, NULL, _starpu_mpi_progress_thread_func, argc_argv);

	_STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	while (!running)
		_STARPU_PTHREAD_COND_WAIT(&cond_progression, &mutex);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

#ifdef STARPU_MPI_ACTIVITY
	hookid = starpu_progression_hook_register(progression_hook_func, NULL);
	STARPU_ASSERT_MSG(hookid >= 0, "starpu_progression_hook_register failed");
#endif /* STARPU_MPI_ACTIVITY */

	_starpu_mpi_add_sync_point_in_fxt();
	_starpu_mpi_comm_amounts_init(MPI_COMM_WORLD);
	_starpu_mpi_cache_init(MPI_COMM_WORLD);
	return 0;
}

int starpu_mpi_init(int *argc, char ***argv, int initialize_mpi)
{
	return _starpu_mpi_initialize(argc, argv, initialize_mpi);
}

int starpu_mpi_initialize(void)
{
	return _starpu_mpi_initialize(NULL, NULL, 0);
}

int starpu_mpi_initialize_extended(int *rank, int *world_size)
{
	int ret;

	ret = _starpu_mpi_initialize(NULL, NULL, 1);
	if (ret == 0)
	{
		_STARPU_DEBUG("Calling MPI_Comm_rank\n");
		MPI_Comm_rank(MPI_COMM_WORLD, rank);
		MPI_Comm_size(MPI_COMM_WORLD, world_size);
	}
	return ret;
}

int starpu_mpi_shutdown(void)
{
	void *value;
	int rank, world_size;

	/* We need to get the rank before calling MPI_Finalize to pass to _starpu_mpi_comm_amounts_display() */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	/* kill the progression thread */
	_STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	running = 0;
	_STARPU_PTHREAD_COND_BROADCAST(&cond_progression);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	starpu_pthread_join(progress_thread, &value);

#ifdef STARPU_MPI_ACTIVITY
	starpu_progression_hook_deregister(hookid);
#endif /* STARPU_MPI_ACTIVITY */

	TRACE_MPI_STOP(rank, world_size);

	/* free the request queues */
	_starpu_mpi_req_list_delete(detached_requests);
	_starpu_mpi_req_list_delete(new_requests);

	_starpu_mpi_comm_amounts_display(rank);
	_starpu_mpi_comm_amounts_free();
	_starpu_mpi_cache_free(world_size);

	return 0;
}
