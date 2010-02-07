/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

//#define VERBOSE_STARPU_MPI	1

/* TODO find a better way to select the polling method (perhaps during the
 * configuration) */
#define USE_STARPU_ACTIVITY	1

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

/*
 *	Isend
 */

static void starpu_mpi_isend_func(struct starpu_mpi_req_s *req)
{
	void *ptr = starpu_mpi_handle_to_ptr(req->data_handle);

#ifdef VERBOSE_STARPU_MPI
	int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	fprintf(stderr, "<<<< STARPU MPI >>>> : Rank %d post MPI isend tag %x dst %d ptr %p req %p\n", rank, req->mpi_tag, req->srcdst, ptr, &req->request);
#endif

	starpu_mpi_handle_to_datatype(req->data_handle, &req->datatype);

	//MPI_Isend(ptr, 1, req->datatype, req->srcdst, req->mpi_tag, req->comm, &req->request);
	MPI_Isend(ptr, 1, req->datatype, req->srcdst, req->mpi_tag, req->comm, &req->request);

	TRACE_MPI_ISEND(req->srcdst, req->mpi_tag, 0);

	/* somebody is perhaps waiting for the MPI request to be posted */
	pthread_mutex_lock(&req->req_mutex);
	req->submitted = 1;
	pthread_cond_broadcast(&req->req_cond);
	pthread_mutex_unlock(&req->req_mutex);
}

int starpu_mpi_isend(starpu_data_handle data_handle, starpu_mpi_req *public_req, int dest, int mpi_tag, MPI_Comm comm)
{
	STARPU_ASSERT(public_req);

	struct starpu_mpi_req_s *req = starpu_mpi_req_new();
	STARPU_ASSERT(req);
	*public_req = req;

	memset(req, 0, sizeof(struct starpu_mpi_req_s));

	/* Initialize the request structure */
	req->submitted = 0;
	req->completed = 0;
	pthread_mutex_init(&req->req_mutex, NULL);
	pthread_cond_init(&req->req_cond, NULL);

	req->request_type = SEND_REQ;

	req->data_handle = data_handle;
	req->srcdst = dest;
	req->mpi_tag = mpi_tag;
	req->comm = comm;
	req->detached = 0;
	req->func = starpu_mpi_isend_func;

	/* Asynchronously request StarPU to fetch the data in main memory: when
	 * it is available in main memory, submit_mpi_req(req) is called and
	 * the request is actually submitted  */
	starpu_sync_data_with_mem_non_blocking(data_handle, STARPU_R,
			submit_mpi_req, (void *)req);

	return 0;
}

/*
 *	Isend (detached)
 */

int starpu_mpi_isend_detached(starpu_data_handle data_handle,
				int dest, int mpi_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	struct starpu_mpi_req_s *req = calloc(1, sizeof(struct starpu_mpi_req_s));
	STARPU_ASSERT(req);

	/* Initialize the request structure */
	req->submitted = 0;
	req->completed = 0;
	pthread_mutex_init(&req->req_mutex, NULL);
	pthread_cond_init(&req->req_cond, NULL);

	req->request_type = SEND_REQ;

	req->data_handle = data_handle;
	req->srcdst = dest;
	req->mpi_tag = mpi_tag;
	req->comm = comm;
	req->func = starpu_mpi_isend_func;

	req->detached = 1;
	req->callback = callback;
	req->callback_arg = arg;

	/* Asynchronously request StarPU to fetch the data in main memory: when
	 * it is available in main memory, submit_mpi_req(req) is called and
	 * the request is actually submitted  */
	starpu_sync_data_with_mem_non_blocking(data_handle, STARPU_R,
			submit_mpi_req, (void *)req);

	return 0;
}

/*
 *	Irecv
 */

static void starpu_mpi_irecv_func(struct starpu_mpi_req_s *req)
{
	void *ptr = starpu_mpi_handle_to_ptr(req->data_handle);
	STARPU_ASSERT(ptr);

	starpu_mpi_handle_to_datatype(req->data_handle, &req->datatype);

#ifdef VERBOSE_STARPU_MPI
	int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	fprintf(stderr, "<<<< STARPU MPI >>>> : Rank %d post MPI irecv tag %x src %d ptr %p req %p datatype %d\n", rank, req->mpi_tag, req->srcdst, ptr, &req->request, req->datatype);
#endif

	MPI_Irecv(ptr, 1, req->datatype, req->srcdst, req->mpi_tag, req->comm, &req->request);

	/* somebody is perhaps waiting for the MPI request to be posted */
	pthread_mutex_lock(&req->req_mutex);
	req->submitted = 1;
	pthread_cond_broadcast(&req->req_cond);
	pthread_mutex_unlock(&req->req_mutex);
}

int starpu_mpi_irecv(starpu_data_handle data_handle, starpu_mpi_req *public_req, int source, int mpi_tag, MPI_Comm comm)
{
	STARPU_ASSERT(public_req);

	struct starpu_mpi_req_s *req = starpu_mpi_req_new();
	STARPU_ASSERT(req);
	*public_req = req;

	STARPU_ASSERT(req);

	memset(req, 0, sizeof(struct starpu_mpi_req_s));

	/* Initialize the request structure */
	req->submitted = 0;
	pthread_mutex_init(&req->req_mutex, NULL);
	pthread_cond_init(&req->req_cond, NULL);

	req->request_type = RECV_REQ;

	req->data_handle = data_handle;
	req->srcdst = source;
	req->mpi_tag = mpi_tag;
	req->comm = comm;
	req->detached = 0;

	req->func = starpu_mpi_irecv_func;

	/* Asynchronously request StarPU to fetch the data in main memory: when
	 * it is available in main memory, submit_mpi_req(req) is called and
	 * the request is actually submitted  */
	starpu_sync_data_with_mem_non_blocking(data_handle, STARPU_W,
			submit_mpi_req, (void *)req);

	return 0;
}

/*
 *	Irecv (detached)
 */

int starpu_mpi_irecv_detached(starpu_data_handle data_handle, int source, int mpi_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	struct starpu_mpi_req_s *req = calloc(1, sizeof(struct starpu_mpi_req_s));
	STARPU_ASSERT(req);

	/* Initialize the request structure */
	req->submitted = 0;
	pthread_mutex_init(&req->req_mutex, NULL);
	pthread_cond_init(&req->req_cond, NULL);

	req->request_type = RECV_REQ;

	req->data_handle = data_handle;
	req->srcdst = source;
	req->mpi_tag = mpi_tag;
	req->comm = comm;

	req->detached = 1;
	req->callback = callback;
	req->callback_arg = arg;

	req->func = starpu_mpi_irecv_func;

	/* Asynchronously request StarPU to fetch the data in main memory: when
	 * it is available in main memory, submit_mpi_req(req) is called and
	 * the request is actually submitted  */
	starpu_sync_data_with_mem_non_blocking(data_handle, STARPU_W,
			submit_mpi_req, (void *)req);

	return 0;
}


/*
 *	Recv
 */

int starpu_mpi_recv(starpu_data_handle data_handle,
		int source, int mpi_tag, MPI_Comm comm, MPI_Status *status)
{
	starpu_mpi_req req;

	starpu_mpi_irecv(data_handle, &req, source, mpi_tag, comm);
	starpu_mpi_wait(&req, status);

	return 0;
}

/*
 *	Send
 */

int starpu_mpi_send(starpu_data_handle data_handle,
		int dest, int mpi_tag, MPI_Comm comm)
{
	starpu_mpi_req req;
	MPI_Status status;

	memset(&status, 0, sizeof(MPI_Status));

	starpu_mpi_isend(data_handle, &req, dest, mpi_tag, comm);
	starpu_mpi_wait(&req, &status);

	return 0;
}

/*
 *	Wait
 */

static void starpu_mpi_wait_func(struct starpu_mpi_req_s *waiting_req)
{
	/* Which is the mpi request we are waiting for ? */
	struct starpu_mpi_req_s *req = waiting_req->other_request;

	req->ret = MPI_Wait(&req->request, waiting_req->status);
	handle_request_termination(req);
}

int starpu_mpi_wait(starpu_mpi_req *public_req, MPI_Status *status)
{
	int ret;
	struct starpu_mpi_req_s waiting_req;
	memset(&waiting_req, 0, sizeof(struct starpu_mpi_req_s));

	struct starpu_mpi_req_s *req = *public_req;

	/* We cannot try to complete a MPI request that was not actually posted
	 * to MPI yet. */
	pthread_mutex_lock(&req->req_mutex);
	while (!req->submitted)
		pthread_cond_wait(&req->req_cond, &req->req_mutex);
	pthread_mutex_unlock(&req->req_mutex);

	/* Initialize the request structure */
	pthread_mutex_init(&waiting_req.req_mutex, NULL);
	pthread_cond_init(&waiting_req.req_cond, NULL);
	waiting_req.status = status;
	waiting_req.other_request = req;
	waiting_req.func = starpu_mpi_wait_func;
	
	submit_mpi_req(&waiting_req);

	/* We wait for the MPI request to finish */
	pthread_mutex_lock(&req->req_mutex);
	while (!req->completed)
		pthread_cond_wait(&req->req_cond, &req->req_mutex);
	pthread_mutex_unlock(&req->req_mutex);

	ret = req->ret;

	/* The internal request structure was automatically allocated */
	*public_req = NULL;
	free(req);

	return ret;
}

/*
 * 	Test
 */

static void starpu_mpi_test_func(struct starpu_mpi_req_s *testing_req)
{
	/* Which is the mpi request we are testing for ? */
	struct starpu_mpi_req_s *req = testing_req->other_request;

//	fprintf(stderr, "<<<< STARPU MPI >>>> Test request %p - mpitag %x - TYPE %s %d\n", &req->request, req->mpi_tag, (req->request_type == RECV_REQ)?"recv : source":"send : dest", req->srcdst);
	int ret = MPI_Test(&req->request, testing_req->flag, testing_req->status);

	if (*testing_req->flag)
	{
		testing_req->ret = ret;
		handle_request_termination(req);
	}

	pthread_mutex_lock(&testing_req->req_mutex);
	testing_req->completed = 1;
	pthread_cond_signal(&testing_req->req_cond);
	pthread_mutex_unlock(&testing_req->req_mutex);
}

int starpu_mpi_test(starpu_mpi_req *public_req, int *flag, MPI_Status *status)
{
	int ret = 0;

	STARPU_ASSERT(public_req);

	struct starpu_mpi_req_s *req = *public_req;

	STARPU_ASSERT(!req->detached);

	pthread_mutex_lock(&req->req_mutex);
	unsigned submitted = req->submitted;
	pthread_mutex_unlock(&req->req_mutex);

	if (submitted)
	{
		struct starpu_mpi_req_s testing_req;
		memset(&testing_req, 0, sizeof(struct starpu_mpi_req_s));

		/* Initialize the request structure */
		pthread_mutex_init(&testing_req.req_mutex, NULL);
		pthread_cond_init(&testing_req.req_cond, NULL);
		testing_req.flag = flag;
		testing_req.status = status;
		testing_req.other_request = req;
		testing_req.func = starpu_mpi_test_func;
		testing_req.completed = 0;
		
		submit_mpi_req(&testing_req);
	
		/* We wait for the test request to finish */
		pthread_mutex_lock(&testing_req.req_mutex);
		while (!testing_req.completed)
			pthread_cond_wait(&testing_req.req_cond, &testing_req.req_mutex);
		pthread_mutex_unlock(&testing_req.req_mutex);
	
		ret = testing_req.ret;

		if (*testing_req.flag)
		{
			/* The request was completed so we liberate the
			 * internal request structure which was automatically
			 * allocated */
			*public_req = NULL;
			free(req);
		}
	}
	else {
		*flag = 0;
	}

	return ret;
}

/*
 *	Requests
 */

static void handle_request_termination(struct starpu_mpi_req_s *req)
{
	MPI_Type_free(&req->datatype);
	starpu_release_data_from_mem(req->data_handle);

#ifdef VERBOSE_STARPU_MPI
	int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	fprintf(stderr, "<<<< STARPU MPI >>>> Rank %d : complete MPI (%s %d) req %p - tag %x\n", rank, (req->request_type == RECV_REQ)?"recv : source":"send : dest", req->srcdst, &req->request, req->mpi_tag);
#endif

	if (req->request_type == RECV_REQ)
	{
		TRACE_MPI_IRECV_END(req->srcdst, req->mpi_tag);
	}

	/* Execute the specified callback, if any */
	if (req->callback)
		req->callback(req->callback_arg);

	/* tell anyone potentiallly waiting on the request that it is
	 * terminated now */
	pthread_mutex_lock(&req->req_mutex);
	req->completed = 1;
	pthread_cond_broadcast(&req->req_cond);
	pthread_mutex_unlock(&req->req_mutex);
}

static void submit_mpi_req(void *arg)
{
	struct starpu_mpi_req_s *req = arg;

	pthread_mutex_lock(&mutex);
	starpu_mpi_req_list_push_front(new_requests, req);
	pthread_cond_broadcast(&cond);
	pthread_mutex_unlock(&mutex);
}

/*
 *	Scheduler hook
 */

static unsigned progression_hook_func(void *arg __attribute__((unused)))
{
	unsigned may_block = 1;

	pthread_mutex_lock(&mutex);
	if (!starpu_mpi_req_list_empty(detached_requests))
	{
		pthread_cond_signal(&cond);
		may_block = 0;
	}
	pthread_mutex_unlock(&mutex);

	return may_block;
}

/*
 *	Progression loop
 */

static void test_detached_requests(void)
{
	int flag;
	MPI_Status status;
	struct starpu_mpi_req_s *req, *next_req;

	pthread_mutex_lock(&detached_requests_mutex);

	for (req = starpu_mpi_req_list_begin(detached_requests);
		req != starpu_mpi_req_list_end(detached_requests);
		req = next_req)
	{
		next_req = starpu_mpi_req_list_next(req);

		pthread_mutex_unlock(&detached_requests_mutex);

		int ret = MPI_Test(&req->request, &flag, &status);
		STARPU_ASSERT(ret == MPI_SUCCESS);


#ifdef VERBOSE_STARPU_MPI
//		if ((req->mpi_tag == 0x20003) || (req->mpi_tag == 0x30003))
//		if (0)
		if ((req->mpi_tag == 0x20003))
		{
			int rank;
		        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

			fprintf(stderr, "<<<< STARPU MPI >>>> Rank %d Test request %p - mpitag %x - TYPE %s %d\n", rank, &req->request, req->mpi_tag, (req->request_type == RECV_REQ)?"recv : source":"send : dest", req->srcdst);
		}
#endif


		if (flag)
		{
			handle_request_termination(req);
		}

		pthread_mutex_lock(&detached_requests_mutex);

		if (flag)
			starpu_mpi_req_list_erase(detached_requests, req);

#warning TODO fix memleak
		/* Detached requests are automatically allocated by the lib */
		//if (req->detached)
		//	free(req);
	}
	
	pthread_mutex_unlock(&detached_requests_mutex);
}

static void handle_new_request(struct starpu_mpi_req_s *req)
{
	STARPU_ASSERT(req);

	/* submit the request to MPI */
	req->func(req);

	if (req->detached)
	{
		pthread_mutex_lock(&mutex);
		starpu_mpi_req_list_push_front(detached_requests, req);
		pthread_mutex_unlock(&mutex);

		starpu_wake_all_blocked_workers();

		/* put the submitted request into the list of pending requests
		 * so that it can be handled by the progression mechanisms */
		pthread_mutex_lock(&mutex);
		pthread_cond_signal(&cond);
		pthread_mutex_unlock(&mutex);
	}
}

static void *progress_thread_func(void *arg __attribute__((unused)))
{
	/* notify the main thread that the progression thread is ready */
	pthread_mutex_lock(&mutex);
	running = 1;
	pthread_cond_signal(&cond);
	pthread_mutex_unlock(&mutex);

	pthread_mutex_lock(&mutex);
	while (running) {
		/* shall we block ? */
		unsigned block = starpu_mpi_req_list_empty(new_requests);

#ifndef USE_STARPU_ACTIVITY
		block = block && starpu_mpi_req_list_empty(detached_requests);
#endif

		if (block)
		{
//			int rank;
//		        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//			if (rank == 3)
//			fprintf(stderr, "<<<< STARPU MPI >>>> Rank %d NO MORE REQUESTS TO HANDLE\n", rank);

			pthread_cond_wait(&cond, &mutex);
		}

		if (!running)
			break;		

		/* test whether there are some terminated "detached request" */
		pthread_mutex_unlock(&mutex);
		test_detached_requests();
		pthread_mutex_lock(&mutex);

		/* get one request */
		struct starpu_mpi_req_s *req;
		while (!starpu_mpi_req_list_empty(new_requests))
		{
			req = starpu_mpi_req_list_pop_back(new_requests);

			/* handling a request is likely to block for a while
			 * (on a sync_data_with_mem call), we want to let the
			 * application submit requests in the meantime, so we
			 * release the lock.  */
			pthread_mutex_unlock(&mutex);
			handle_new_request(req);
			pthread_mutex_lock(&mutex);
		}
	}

	pthread_mutex_unlock(&mutex);

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
#ifdef USE_FXT
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

#ifdef VERBOSE
	fprintf(stderr, "StarPU MPI (rank %d): unique key %x\n", rank, random_number);
#endif

#endif
}


int starpu_mpi_initialize(void)
{
	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&cond, NULL);
	new_requests = starpu_mpi_req_list_new();

	pthread_mutex_init(&detached_requests_mutex, NULL);
	detached_requests = starpu_mpi_req_list_new();

	int ret = pthread_create(&progress_thread, NULL, progress_thread_func, NULL);

	pthread_mutex_lock(&mutex);
	while (!running)
		pthread_cond_wait(&cond, &mutex);
	pthread_mutex_unlock(&mutex);

#ifdef USE_STARPU_ACTIVITY
	hookid = starpu_register_progression_hook(progression_hook_func, NULL);
	STARPU_ASSERT(hookid >= 0);
#endif

	_starpu_mpi_add_sync_point_in_fxt();
	
	return 0;
}

int starpu_mpi_shutdown(void)
{
	void *value;

	/* kill the progression thread */
	pthread_mutex_lock(&mutex);
	running = 0;
	pthread_cond_broadcast(&cond);
	pthread_mutex_unlock(&mutex);

	pthread_join(progress_thread, &value);

#ifdef USE_STARPU_ACTIVITY
	starpu_deregister_progression_hook(hookid);
#endif 

	/* liberate the request queues */
	starpu_mpi_req_list_delete(detached_requests);
	starpu_mpi_req_list_delete(new_requests);

	return 0;
}
