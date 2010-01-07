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

#include <starpu_mpi.h>
#include <starpu_mpi_datatype.h>

static void submit_mpi_req(void *arg);
static void handle_request_termination(struct starpu_mpi_req_s *req);

/* The list of requests that have been newly submitted by the application */
static starpu_mpi_req_list_t new_requests; 

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
	starpu_mpi_handle_to_datatype(req->data_handle, &req->datatype);

	MPI_Isend(ptr, 1, req->datatype, req->srcdst, req->mpi_tag, req->comm, &req->request);

	/* somebody is perhaps waiting for the MPI request to be posted */
	pthread_mutex_lock(&req->req_mutex);
	req->submitted = 1;
	pthread_cond_broadcast(&req->req_cond);
	pthread_mutex_unlock(&req->req_mutex);
}

int starpu_mpi_isend(starpu_data_handle data_handle, struct starpu_mpi_req_s *req, int dest, int mpi_tag, MPI_Comm comm)
{
	STARPU_ASSERT(req);

	memset(req, 0, sizeof(struct starpu_mpi_req_s));

	/* Initialize the request structure */
	req->submitted = 0;
	req->completed = 0;
	pthread_mutex_init(&req->req_mutex, NULL);
	pthread_cond_init(&req->req_cond, NULL);

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

int starpu_mpi_isend_detached(starpu_data_handle data_handle, struct starpu_mpi_req_s *req, int dest, int mpi_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	/* TODO */
	return 0;
}

/*
 *	Irecv
 */

static void starpu_mpi_irecv_func(struct starpu_mpi_req_s *req)
{
	void *ptr = starpu_mpi_handle_to_ptr(req->data_handle);
	starpu_mpi_handle_to_datatype(req->data_handle, &req->datatype);

	MPI_Irecv(ptr, 1, req->datatype, req->srcdst, req->mpi_tag, req->comm, &req->request);

	/* somebody is perhaps waiting for the MPI request to be posted */
	pthread_mutex_lock(&req->req_mutex);
	req->submitted = 1;
	pthread_cond_broadcast(&req->req_cond);
	pthread_mutex_unlock(&req->req_mutex);
}

int starpu_mpi_irecv(starpu_data_handle data_handle, struct starpu_mpi_req_s *req, int source, int mpi_tag, MPI_Comm comm)
{
	STARPU_ASSERT(req);

	memset(req, 0, sizeof(struct starpu_mpi_req_s));

	/* Initialize the request structure */
	req->submitted = 0;
	pthread_mutex_init(&req->req_mutex, NULL);
	pthread_cond_init(&req->req_cond, NULL);

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
 *	Recv
 */

int starpu_mpi_recv(starpu_data_handle data_handle,
		int source, int mpi_tag, MPI_Comm comm, MPI_Status *status)
{
	struct starpu_mpi_req_s req;

	memset(&req, 0, sizeof(struct starpu_mpi_req_s));

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
	struct starpu_mpi_req_s req;
	MPI_Status status;

	memset(&req, 0, sizeof(struct starpu_mpi_req_s));
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

int starpu_mpi_wait(struct starpu_mpi_req_s *req, MPI_Status *status)
{
	int ret;
	struct starpu_mpi_req_s waiting_req;

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

	return ret;
}

/*
 * 	Test
 */

static void starpu_mpi_test_func(struct starpu_mpi_req_s *testing_req)
{
	/* Which is the mpi request we are testing for ? */
	struct starpu_mpi_req_s *req = testing_req->other_request;

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

int starpu_mpi_test(struct starpu_mpi_req_s *req, int *flag, MPI_Status *status)
{
	int ret = 0;

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
	}
	else {
		*flag = 0;
	}

	return ret;
}

/*
 *	Requests
 */

void handle_request_termination(struct starpu_mpi_req_s *req)
{
	MPI_Type_free(&req->datatype);
	starpu_release_data_from_mem(req->data_handle);

	/* tell anyone potentiallly waiting on the request that it is
	 * terminated now */
	pthread_mutex_lock(&req->req_mutex);
	req->completed = 1;
	pthread_cond_broadcast(&req->req_cond);
	pthread_mutex_unlock(&req->req_mutex);
	
}

void submit_mpi_req(void *arg)
{
	struct starpu_mpi_req_s *req = arg;

	pthread_mutex_lock(&mutex);
	starpu_mpi_req_list_push_front(new_requests, req);
	pthread_cond_broadcast(&cond);
	pthread_mutex_unlock(&mutex);
}

/*
 *	Progression loop
 */

void handle_new_request(struct starpu_mpi_req_s *req)
{
	STARPU_ASSERT(req);

	/* submit the request to MPI */
	req->func(req);
}

void *progress_thread_func(void *arg __attribute__((unused)))
{
	/* notify the main thread that the progression thread is ready */
	pthread_mutex_lock(&mutex);
	running = 1;
	pthread_cond_signal(&cond);
	pthread_mutex_unlock(&mutex);

	pthread_mutex_lock(&mutex);
	while (running) {
		/* TODO test if there is some "detached request" and progress if this is the case */
		pthread_cond_wait(&cond, &mutex);
		if (!running)
			break;		

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

int starpu_mpi_initialize(void)
{
	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&cond, NULL);

	new_requests = starpu_mpi_req_list_new();

	int ret = pthread_create(&progress_thread, NULL, progress_thread_func, NULL);

	pthread_mutex_lock(&mutex);
	if (!running)
		pthread_cond_wait(&cond, &mutex);
	pthread_mutex_unlock(&mutex);

	return 0;
}

int starpu_mpi_shutdown(void)
{
	/* kill the progression thread */
	pthread_mutex_lock(&mutex);
	running = 0;
	pthread_cond_signal(&cond);
	pthread_mutex_unlock(&mutex);

	void *value;
	pthread_join(progress_thread, &value);

	/* TODO liberate the queues */

	return 0;
}
