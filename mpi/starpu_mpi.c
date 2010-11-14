/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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

#include <stdlib.h>
#include <starpu_mpi.h>
#include <starpu_mpi_datatype.h>
#include <starpu_mpi_private.h>

//#define STARPU_MPI_VERBOSE	1

#ifdef STARPU_MPI_VERBOSE
#  define _STARPU_MPI_DEBUG(fmt, args ...) { int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);                        \
                                             fprintf(stderr, "[%d][starpu_mpi][%s] " fmt , rank, __func__ ,##args); \
                                             fflush(stderr); }
#else
#  define _STARPU_MPI_DEBUG(fmt, args ...)
#endif

#ifdef STARPU_MPI_VERBOSE
#  define _STARPU_MPI_LOG_IN()             { int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);                        \
                                             fprintf(stderr, "[%d][starpu_mpi][%s] -->\n", rank, __func__ ); \
                                             fflush(stderr); }
#  define _STARPU_MPI_LOG_OUT()            { int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);                        \
                                             fprintf(stderr, "[%d][starpu_mpi][%s] <--\n", rank, __func__ ); \
                                             fflush(stderr); }
#else
#  define _STARPU_MPI_LOG_IN()
#  define _STARPU_MPI_LOG_OUT()
#endif

/* TODO find a better way to select the polling method (perhaps during the
 * configuration) */
//#define USE_STARPU_ACTIVITY	1

static void submit_mpi_req(void *arg);
static void handle_request_termination(struct starpu_mpi_req_s *req);

/* The list of requests that have been newly submitted by the application */
static starpu_mpi_req_list_t new_requests; 

/* The list of detached requests that have already been submitted to MPI */
static starpu_mpi_req_list_t detached_requests;
static pthread_mutex_t detached_requests_mutex;

static pthread_cond_t cond;
static pthread_mutex_t mutex;
static pthread_t progress_thread;
static int running = 0;

#if 0
void starpu_mpi_debug(FILE *stream, const char *format, ...) {
        int rank;
	va_list args;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stream, "[%d] ", rank);
        va_start(args, format);
        vfprintf(stream, format, args);
        va_end(args);
        fflush(stream);
}
#endif

/*
 *	Isend
 */

static void starpu_mpi_isend_func(struct starpu_mpi_req_s *req)
{
        _STARPU_MPI_LOG_IN();
	void *ptr = starpu_mpi_handle_to_ptr(req->data_handle);

        _STARPU_MPI_DEBUG("post MPI isend tag %x dst %d ptr %p req %p\n", req->mpi_tag, req->srcdst, ptr, &req->request);

	starpu_mpi_handle_to_datatype(req->data_handle, &req->datatype);

        req->ret = MPI_Isend(ptr, 1, req->datatype, req->srcdst, req->mpi_tag, req->comm, &req->request);
        STARPU_ASSERT(req->ret == MPI_SUCCESS);

	TRACE_MPI_ISEND(req->srcdst, req->mpi_tag, 0);

	/* somebody is perhaps waiting for the MPI request to be posted */
	PTHREAD_MUTEX_LOCK(&req->req_mutex);
	req->submitted = 1;
	PTHREAD_COND_BROADCAST(&req->req_cond);
	PTHREAD_MUTEX_UNLOCK(&req->req_mutex);
        _STARPU_MPI_LOG_OUT();
}

static struct starpu_mpi_req_s *_starpu_mpi_isend_common(starpu_data_handle data_handle,
				int dest, int mpi_tag, MPI_Comm comm,
				unsigned detached, void (*callback)(void *), void *arg)
{
	struct starpu_mpi_req_s *req = calloc(1, sizeof(struct starpu_mpi_req_s));
	STARPU_ASSERT(req);

        _STARPU_MPI_LOG_IN();
	/* Initialize the request structure */
	req->submitted = 0;
	req->completed = 0;
	PTHREAD_MUTEX_INIT(&req->req_mutex, NULL);
	PTHREAD_COND_INIT(&req->req_cond, NULL);

	req->request_type = SEND_REQ;

	req->data_handle = data_handle;
	req->srcdst = dest;
	req->mpi_tag = mpi_tag;
	req->comm = comm;
	req->func = starpu_mpi_isend_func;

	req->detached = detached;
	req->callback = callback;
	req->callback_arg = arg;

	/* Asynchronously request StarPU to fetch the data in main memory: when
	 * it is available in main memory, submit_mpi_req(req) is called and
	 * the request is actually submitted  */
	starpu_data_acquire_cb(data_handle, STARPU_R, submit_mpi_req, (void *)req);

        _STARPU_MPI_LOG_OUT();
	return req;
}

int starpu_mpi_isend(starpu_data_handle data_handle, starpu_mpi_req *public_req, int dest, int mpi_tag, MPI_Comm comm)
{
        _STARPU_MPI_LOG_IN();
	STARPU_ASSERT(public_req);

	struct starpu_mpi_req_s *req;
	req = _starpu_mpi_isend_common(data_handle, dest, mpi_tag, comm, 0, NULL, NULL);

	STARPU_ASSERT(req);
	*public_req = req;

        _STARPU_MPI_LOG_OUT();
	return 0;
}

/*
 *	Isend (detached)
 */

int starpu_mpi_isend_detached(starpu_data_handle data_handle,
				int dest, int mpi_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
        _STARPU_MPI_LOG_IN();
	_starpu_mpi_isend_common(data_handle, dest, mpi_tag, comm, 1, callback, arg);

        _STARPU_MPI_LOG_OUT();
	return 0;
}

/*
 *	Irecv
 */

static void starpu_mpi_irecv_func(struct starpu_mpi_req_s *req)
{
        _STARPU_MPI_LOG_IN();
	void *ptr = starpu_mpi_handle_to_ptr(req->data_handle);
	STARPU_ASSERT(ptr);

	starpu_mpi_handle_to_datatype(req->data_handle, &req->datatype);

	_STARPU_MPI_DEBUG("post MPI irecv tag %x src %d ptr %p req %p datatype %d\n", req->mpi_tag, req->srcdst, ptr, &req->request, req->datatype);

        req->ret = MPI_Irecv(ptr, 1, req->datatype, req->srcdst, req->mpi_tag, req->comm, &req->request);
        STARPU_ASSERT(req->ret == MPI_SUCCESS);

	/* somebody is perhaps waiting for the MPI request to be posted */
	PTHREAD_MUTEX_LOCK(&req->req_mutex);
	req->submitted = 1;
	PTHREAD_COND_BROADCAST(&req->req_cond);
	PTHREAD_MUTEX_UNLOCK(&req->req_mutex);
        _STARPU_MPI_LOG_OUT();
}

static struct starpu_mpi_req_s *_starpu_mpi_irecv_common(starpu_data_handle data_handle, int source, int mpi_tag, MPI_Comm comm, unsigned detached, void (*callback)(void *), void *arg)
{
        _STARPU_MPI_LOG_IN();
	struct starpu_mpi_req_s *req = calloc(1, sizeof(struct starpu_mpi_req_s));
	STARPU_ASSERT(req);

	/* Initialize the request structure */
	req->submitted = 0;
	PTHREAD_MUTEX_INIT(&req->req_mutex, NULL);
	PTHREAD_COND_INIT(&req->req_cond, NULL);

	req->request_type = RECV_REQ;

	req->data_handle = data_handle;
	req->srcdst = source;
	req->mpi_tag = mpi_tag;
	req->comm = comm;

	req->detached = detached;
	req->callback = callback;
	req->callback_arg = arg;

	req->func = starpu_mpi_irecv_func;

	/* Asynchronously request StarPU to fetch the data in main memory: when
	 * it is available in main memory, submit_mpi_req(req) is called and
	 * the request is actually submitted  */
	starpu_data_acquire_cb(data_handle, STARPU_W, submit_mpi_req, (void *)req);

        _STARPU_MPI_LOG_OUT();
	return req;
}

int starpu_mpi_irecv(starpu_data_handle data_handle, starpu_mpi_req *public_req, int source, int mpi_tag, MPI_Comm comm)
{
        _STARPU_MPI_LOG_IN();
	STARPU_ASSERT(public_req);

	struct starpu_mpi_req_s *req;
	req = _starpu_mpi_irecv_common(data_handle, source, mpi_tag, comm, 0, NULL, NULL);

	STARPU_ASSERT(req);
	*public_req = req;

        _STARPU_MPI_LOG_OUT();
	return 0;
}

/*
 *	Irecv (detached)
 */

int starpu_mpi_irecv_detached(starpu_data_handle data_handle, int source, int mpi_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
        _STARPU_MPI_LOG_IN();
	_starpu_mpi_irecv_common(data_handle, source, mpi_tag, comm, 1, callback, arg);

        _STARPU_MPI_LOG_OUT();
	return 0;
}


/*
 *	Recv
 */

int starpu_mpi_recv(starpu_data_handle data_handle, int source, int mpi_tag, MPI_Comm comm, MPI_Status *status)
{
	starpu_mpi_req req;

        _STARPU_MPI_LOG_IN();
	starpu_mpi_irecv(data_handle, &req, source, mpi_tag, comm);
	starpu_mpi_wait(&req, status);

        _STARPU_MPI_LOG_OUT();
	return 0;
}

/*
 *	Send
 */

int starpu_mpi_send(starpu_data_handle data_handle, int dest, int mpi_tag, MPI_Comm comm)
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

/*
 *	Wait
 */

static void starpu_mpi_wait_func(struct starpu_mpi_req_s *waiting_req)
{
        _STARPU_MPI_LOG_IN();
	/* Which is the mpi request we are waiting for ? */
	struct starpu_mpi_req_s *req = waiting_req->other_request;

	req->ret = MPI_Wait(&req->request, waiting_req->status);
        STARPU_ASSERT(req->ret == MPI_SUCCESS);

	handle_request_termination(req);
        _STARPU_MPI_LOG_OUT();
}

int starpu_mpi_wait(starpu_mpi_req *public_req, MPI_Status *status)
{
        _STARPU_MPI_LOG_IN();
	int ret;
	struct starpu_mpi_req_s *waiting_req = calloc(1, sizeof(struct starpu_mpi_req_s));
	STARPU_ASSERT(waiting_req);
	struct starpu_mpi_req_s *req = *public_req;

	/* We cannot try to complete a MPI request that was not actually posted
	 * to MPI yet. */
	PTHREAD_MUTEX_LOCK(&(req->req_mutex));
	while (!(req->submitted))
		PTHREAD_COND_WAIT(&(req->req_cond), &(req->req_mutex));
	PTHREAD_MUTEX_UNLOCK(&(req->req_mutex));

	/* Initialize the request structure */
	PTHREAD_MUTEX_INIT(&(waiting_req->req_mutex), NULL);
	PTHREAD_COND_INIT(&(waiting_req->req_cond), NULL);
	waiting_req->status = status;
	waiting_req->other_request = req;
	waiting_req->func = starpu_mpi_wait_func;
	waiting_req->request_type = WAIT_REQ;

	submit_mpi_req(waiting_req);

	/* We wait for the MPI request to finish */
	PTHREAD_MUTEX_LOCK(&req->req_mutex);
	while (!req->completed)
		PTHREAD_COND_WAIT(&req->req_cond, &req->req_mutex);
	PTHREAD_MUTEX_UNLOCK(&req->req_mutex);

	ret = req->ret;

	/* The internal request structure was automatically allocated */
	*public_req = NULL;
	free(req);

        //free(waiting_req);
        _STARPU_MPI_LOG_OUT();
	return ret;
}

/*
 * 	Test
 */

static void starpu_mpi_test_func(struct starpu_mpi_req_s *testing_req)
{
        _STARPU_MPI_LOG_IN();
	/* Which is the mpi request we are testing for ? */
	struct starpu_mpi_req_s *req = testing_req->other_request;

        _STARPU_MPI_DEBUG("Test request %p - mpitag %x - TYPE %s %d\n", &req->request, req->mpi_tag, (req->request_type == RECV_REQ)?"recv : source":"send : dest", req->srcdst);
	req->ret = MPI_Test(&req->request, testing_req->flag, testing_req->status);
        STARPU_ASSERT(req->ret == MPI_SUCCESS);

	if (*testing_req->flag)
	{
		testing_req->ret = req->ret;
		handle_request_termination(req);
	}

	PTHREAD_MUTEX_LOCK(&testing_req->req_mutex);
	testing_req->completed = 1;
	PTHREAD_COND_SIGNAL(&testing_req->req_cond);
	PTHREAD_MUTEX_UNLOCK(&testing_req->req_mutex);
        _STARPU_MPI_LOG_OUT();
}

int starpu_mpi_test(starpu_mpi_req *public_req, int *flag, MPI_Status *status)
{
        _STARPU_MPI_LOG_IN();
	int ret = 0;

	STARPU_ASSERT(public_req);

	struct starpu_mpi_req_s *req = *public_req;

	STARPU_ASSERT(!req->detached);

	PTHREAD_MUTEX_LOCK(&req->req_mutex);
	unsigned submitted = req->submitted;
	PTHREAD_MUTEX_UNLOCK(&req->req_mutex);

	if (submitted)
	{
		struct starpu_mpi_req_s *testing_req = calloc(1, sizeof(struct starpu_mpi_req_s));
                STARPU_ASSERT(testing_req);
                //		memset(testing_req, 0, sizeof(struct starpu_mpi_req_s));

		/* Initialize the request structure */
		PTHREAD_MUTEX_INIT(&(testing_req->req_mutex), NULL);
		PTHREAD_COND_INIT(&(testing_req->req_cond), NULL);
		testing_req->flag = flag;
		testing_req->status = status;
		testing_req->other_request = req;
		testing_req->func = starpu_mpi_test_func;
		testing_req->completed = 0;
                testing_req->request_type = TEST_REQ;

		submit_mpi_req(testing_req);
	
		/* We wait for the test request to finish */
		PTHREAD_MUTEX_LOCK(&(testing_req->req_mutex));
		while (!(testing_req->completed))
                        PTHREAD_COND_WAIT(&(testing_req->req_cond), &(testing_req->req_mutex));
		PTHREAD_MUTEX_UNLOCK(&(testing_req->req_mutex));

		ret = testing_req->ret;

		if (*(testing_req->flag))
		{
			/* The request was completed so we free the internal
			 * request structure which was automatically allocated
			 * */
			*public_req = NULL;
			free(req);
		}
	}
	else {
		*flag = 0;
	}

        _STARPU_MPI_LOG_OUT();
	return ret;
}

/*
 *	Barrier
 */

static void starpu_mpi_barrier_func(struct starpu_mpi_req_s *barrier_req)
{
        _STARPU_MPI_LOG_IN();

	barrier_req->ret = MPI_Barrier(barrier_req->comm);
        STARPU_ASSERT(barrier_req->ret == MPI_SUCCESS);

	handle_request_termination(barrier_req);
        _STARPU_MPI_LOG_OUT();
}

int starpu_mpi_barrier(MPI_Comm comm)
{
        _STARPU_MPI_LOG_IN();
	int ret;
	struct starpu_mpi_req_s *barrier_req = calloc(1, sizeof(struct starpu_mpi_req_s));
	STARPU_ASSERT(barrier_req);

	/* Initialize the request structure */
	PTHREAD_MUTEX_INIT(&(barrier_req->req_mutex), NULL);
	PTHREAD_COND_INIT(&(barrier_req->req_cond), NULL);
	barrier_req->func = starpu_mpi_barrier_func;
	barrier_req->request_type = BARRIER_REQ;
	barrier_req->comm = comm;

	submit_mpi_req(barrier_req);

	/* We wait for the MPI request to finish */
	PTHREAD_MUTEX_LOCK(&barrier_req->req_mutex);
	while (!barrier_req->completed)
		PTHREAD_COND_WAIT(&barrier_req->req_cond, &barrier_req->req_mutex);
	PTHREAD_MUTEX_UNLOCK(&barrier_req->req_mutex);

	ret = barrier_req->ret;

        //free(waiting_req);
        _STARPU_MPI_LOG_OUT();
	return ret;
}

/*
 *	Requests
 */

static char *starpu_mpi_request_type(unsigned request_type)
{
        switch (request_type)
                {
                case SEND_REQ: return "send";
                case RECV_REQ: return "recv";
                case WAIT_REQ: return "wait";
                case TEST_REQ: return "test";
                case BARRIER_REQ: return "barrier";
                default: return "unknown request type";
                }
}

static void handle_request_termination(struct starpu_mpi_req_s *req)
{
        _STARPU_MPI_LOG_IN();

	_STARPU_MPI_DEBUG("complete MPI (%s %d) req %p - tag %x\n", starpu_mpi_request_type(req->request_type), req->srcdst, &req->request, req->mpi_tag);

        if (req->request_type != BARRIER_REQ) {
                MPI_Type_free(&req->datatype);
                starpu_data_release(req->data_handle);
        }


	if (req->request_type == RECV_REQ)
	{
		TRACE_MPI_IRECV_END(req->srcdst, req->mpi_tag);
	}

	/* Execute the specified callback, if any */
	if (req->callback)
		req->callback(req->callback_arg);

	/* tell anyone potentiallly waiting on the request that it is
	 * terminated now */
	PTHREAD_MUTEX_LOCK(&req->req_mutex);
	req->completed = 1;
	PTHREAD_COND_BROADCAST(&req->req_cond);
	PTHREAD_MUTEX_UNLOCK(&req->req_mutex);
        _STARPU_MPI_LOG_OUT();
}

static void submit_mpi_req(void *arg)
{
        _STARPU_MPI_LOG_IN();
	struct starpu_mpi_req_s *req = arg;

	PTHREAD_MUTEX_LOCK(&mutex);
	starpu_mpi_req_list_push_front(new_requests, req);
	PTHREAD_COND_BROADCAST(&cond);
	PTHREAD_MUTEX_UNLOCK(&mutex);
        _STARPU_MPI_LOG_OUT();
}

/*
 *	Scheduler hook
 */

#ifdef USE_STARPU_ACTIVITY
static unsigned progression_hook_func(void *arg __attribute__((unused)))
{
	unsigned may_block = 1;

	PTHREAD_MUTEX_LOCK(&mutex);
	if (!starpu_mpi_req_list_empty(detached_requests))
	{
		PTHREAD_COND_SIGNAL(&cond);
		may_block = 0;
	}
	PTHREAD_MUTEX_UNLOCK(&mutex);

	return may_block;
}
#endif

/*
 *	Progression loop
 */

static void test_detached_requests(void)
{
        _STARPU_MPI_LOG_IN();
	int flag;
	MPI_Status status;
	struct starpu_mpi_req_s *req, *next_req;

	PTHREAD_MUTEX_LOCK(&detached_requests_mutex);

	for (req = starpu_mpi_req_list_begin(detached_requests);
		req != starpu_mpi_req_list_end(detached_requests);
		req = next_req)
	{
		next_req = starpu_mpi_req_list_next(req);

		PTHREAD_MUTEX_UNLOCK(&detached_requests_mutex);

                _STARPU_MPI_DEBUG("Test detached request %p - mpitag %x - TYPE %s %d\n", &req->request, req->mpi_tag, (req->request_type == RECV_REQ)?"recv : source":"send : dest", req->srcdst);
		req->ret = MPI_Test(&req->request, &flag, &status);
		STARPU_ASSERT(req->ret == MPI_SUCCESS);

		if (flag)
		{
			handle_request_termination(req);
		}

		PTHREAD_MUTEX_LOCK(&detached_requests_mutex);

		if (flag)
			starpu_mpi_req_list_erase(detached_requests, req);

#warning TODO fix memleak
		/* Detached requests are automatically allocated by the lib */
		//if (req->detached)
		//	free(req);
	}
	
	PTHREAD_MUTEX_UNLOCK(&detached_requests_mutex);
        _STARPU_MPI_LOG_OUT();
}

static void handle_new_request(struct starpu_mpi_req_s *req)
{
        _STARPU_MPI_LOG_IN();
	STARPU_ASSERT(req);

	/* submit the request to MPI */
	req->func(req);

	if (req->detached)
	{
		PTHREAD_MUTEX_LOCK(&mutex);
		starpu_mpi_req_list_push_front(detached_requests, req);
		PTHREAD_MUTEX_UNLOCK(&mutex);

		starpu_wake_all_blocked_workers();

		/* put the submitted request into the list of pending requests
		 * so that it can be handled by the progression mechanisms */
		PTHREAD_MUTEX_LOCK(&mutex);
		PTHREAD_COND_SIGNAL(&cond);
		PTHREAD_MUTEX_UNLOCK(&mutex);
	}
        _STARPU_MPI_LOG_OUT();
}

static void *progress_thread_func(void *arg)
{
        int *initialize_mpi = (int *) arg;

        if (*initialize_mpi) {
#warning get real argc and argv from the application
                int argc = 0;
                char **argv = NULL;
                int thread_support;
                if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &thread_support) != MPI_SUCCESS) {
                        fprintf(stderr,"MPI_Init_thread failed\n");
                        exit(1);
                }
                if (thread_support == MPI_THREAD_FUNNELED)
                        fprintf(stderr,"Warning: MPI only has funneled thread support, not serialized, hoping this will work\n");
                if (thread_support < MPI_THREAD_FUNNELED)
                        fprintf(stderr,"Warning: MPI does not have thread support!\n");
        }

	/* notify the main thread that the progression thread is ready */
	PTHREAD_MUTEX_LOCK(&mutex);
	running = 1;
	PTHREAD_COND_SIGNAL(&cond);
	PTHREAD_MUTEX_UNLOCK(&mutex);

	PTHREAD_MUTEX_LOCK(&mutex);
	while (running || !(starpu_mpi_req_list_empty(new_requests)) || !(starpu_mpi_req_list_empty(detached_requests))) {
		/* shall we block ? */
		unsigned block = starpu_mpi_req_list_empty(new_requests);

#ifndef USE_STARPU_ACTIVITY
		block = block && starpu_mpi_req_list_empty(detached_requests);
#endif

		if (block)
		{
                        _STARPU_MPI_DEBUG("NO MORE REQUESTS TO HANDLE\n");
			PTHREAD_COND_WAIT(&cond, &mutex);
		}

		/* test whether there are some terminated "detached request" */
		PTHREAD_MUTEX_UNLOCK(&mutex);
		test_detached_requests();
		PTHREAD_MUTEX_LOCK(&mutex);

		/* get one request */
		struct starpu_mpi_req_s *req;
		while (!starpu_mpi_req_list_empty(new_requests))
		{
			req = starpu_mpi_req_list_pop_back(new_requests);

			/* handling a request is likely to block for a while
			 * (on a sync_data_with_mem call), we want to let the
			 * application submit requests in the meantime, so we
			 * release the lock.  */
			PTHREAD_MUTEX_UNLOCK(&mutex);
			handle_new_request(req);
			PTHREAD_MUTEX_LOCK(&mutex);
		}
	}

	STARPU_ASSERT(starpu_mpi_req_list_empty(detached_requests));
	STARPU_ASSERT(starpu_mpi_req_list_empty(new_requests));

	PTHREAD_MUTEX_UNLOCK(&mutex);

	return NULL;
}

/*
 *	(De)Initialization methods 
 */

#ifdef USE_STARPU_ACTIVITY
static int hookid = - 1;
#endif

static void _starpu_mpi_add_sync_point_in_fxt(void)
{
#ifdef STARPU_USE_FXT
	int rank;
	int worldsize;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	
	int barrier_ret = MPI_Barrier(MPI_COMM_WORLD);
	STARPU_ASSERT(barrier_ret == MPI_SUCCESS);

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
		
	MPI_Bcast(&random_number, 1, MPI_INT, 0, MPI_COMM_WORLD);

	TRACE_MPI_BARRIER(rank, worldsize, random_number);

	_STARPU_MPI_DEBUG("unique key %x\n", random_number);
#endif
}


int starpu_mpi_initialize(void)
{
        return starpu_mpi_initialize_extended(0, NULL, NULL);
}

int starpu_mpi_initialize_extended(int initialize_mpi, int *rank, int *world_size)
{
	PTHREAD_MUTEX_INIT(&mutex, NULL);
	PTHREAD_COND_INIT(&cond, NULL);
	new_requests = starpu_mpi_req_list_new();

	PTHREAD_MUTEX_INIT(&detached_requests_mutex, NULL);
	detached_requests = starpu_mpi_req_list_new();

	int ret = pthread_create(&progress_thread, NULL, progress_thread_func, (void *)&initialize_mpi);

	PTHREAD_MUTEX_LOCK(&mutex);
	while (!running)
		PTHREAD_COND_WAIT(&cond, &mutex);
	PTHREAD_MUTEX_UNLOCK(&mutex);

        if (initialize_mpi) {
                MPI_Comm_rank(MPI_COMM_WORLD, rank);
                MPI_Comm_size(MPI_COMM_WORLD, world_size);
        }

#ifdef USE_STARPU_ACTIVITY
	hookid = starpu_progression_hook_register(progression_hook_func, NULL);
	STARPU_ASSERT(hookid >= 0);
#endif

	_starpu_mpi_add_sync_point_in_fxt();
	
	return 0;
}

int starpu_mpi_shutdown(void)
{
	void *value;

	/* kill the progression thread */
	PTHREAD_MUTEX_LOCK(&mutex);
	running = 0;
	PTHREAD_COND_BROADCAST(&cond);
	PTHREAD_MUTEX_UNLOCK(&mutex);

	pthread_join(progress_thread, &value);

#ifdef USE_STARPU_ACTIVITY
	starpu_progression_hook_deregister(hookid);
#endif 

	/* free the request queues */
	starpu_mpi_req_list_delete(detached_requests);
	starpu_mpi_req_list_delete(new_requests);

	return 0;
}
