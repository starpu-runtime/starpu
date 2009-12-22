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

static void submit_mpi_req(struct starpu_mpi_req_s *req);

static starpu_mpi_req_list_t new_requests; 
static starpu_mpi_req_list_t pending_requests; 

static pthread_cond_t cond;
static pthread_mutex_t mutex;
static pthread_t progress_thread;
static int running = 0;

static void _handle_new_mpi_isend(struct starpu_mpi_req_s *req)
{
	void *ptr = starpu_mpi_handle_to_ptr(req->data_handle);
	starpu_mpi_handle_to_datatype(req->data_handle, &req->datatype);

	MPI_Isend(ptr, 1, req->datatype, req->dst, req->mpi_tag, req->comm, &req->request);
}

int starpu_mpi_isend(starpu_data_handle data_handle, struct starpu_mpi_req_s *req,
		int dest, int mpi_tag, MPI_Comm comm,
		void (*callback)(void *))
{
	req->submitted = 0;
	pthread_mutex_init(&req->req_mutex, NULL);
	pthread_cond_init(&req->req_cond, NULL);

	req->dst = dest;
	req->mpi_tag = mpi_tag;
	req->comm = comm;

	req->handle_new = _handle_new_mpi_isend;

	submit_mpi_req(req);

	return 0;
}

static void _handle_new_mpi_irecv(struct starpu_mpi_req_s *req)
{
	void *ptr = starpu_mpi_handle_to_ptr(req->data_handle);
	starpu_mpi_handle_to_datatype(req->data_handle, &req->datatype);

	MPI_Irecv(ptr, 1, req->datatype, req->src, req->mpi_tag, req->comm, &req->request);
}


/* NB: there is no status field here as we (may) return before the request is
 * actually transmitted to MPI. */
int starpu_mpi_irecv(starpu_data_handle data_handle, struct starpu_mpi_req_s *req,
		int source, int mpi_tag, MPI_Comm comm,
		void (*callback)(void *))
{
	req->submitted = 0;
	pthread_mutex_init(&req->req_mutex, NULL);
	pthread_cond_init(&req->req_cond, NULL);

	req->src = source;
	req->mpi_tag = mpi_tag;
	req->comm = comm;

	req->handle_new = _handle_new_mpi_irecv;

	submit_mpi_req(req);

	return 0;
}

int starpu_mpi_recv(starpu_data_handle data_handle,
		int source, int mpi_tag, MPI_Comm comm, MPI_Status *status)
{
	/* test if we are blocking in a callback .. */
	int ret = starpu_sync_data_with_mem(data_handle, STARPU_W);
	if (ret)
		return ret;

	void *ptr = starpu_mpi_handle_to_ptr(data_handle);
	
	MPI_Datatype datatype;
	starpu_mpi_handle_to_datatype(data_handle, &datatype);
	MPI_Recv(ptr, 1, datatype, source, mpi_tag, comm, status);

	starpu_release_data_from_mem(data_handle);

	return 0;
}

int starpu_mpi_send(starpu_data_handle data_handle,
		int dest, int mpi_tag, MPI_Comm comm)
{
	/* test if we are blocking in a callback .. */
	int ret = starpu_sync_data_with_mem(data_handle, STARPU_R);
	if (ret)
		return ret;

	void *ptr = starpu_mpi_handle_to_ptr(data_handle);
	
	MPI_Status status;
	MPI_Datatype datatype;
	starpu_mpi_handle_to_datatype(data_handle, &datatype);
	MPI_Send(ptr, 1, datatype, dest,  mpi_tag, comm);

	starpu_release_data_from_mem(data_handle);

	return 0;
}

int starpu_mpi_wait(struct starpu_mpi_req_s *req, MPI_Status *status)
{
	int ret;

	pthread_mutex_lock(&req->req_mutex);

	while (!req->submitted)
		pthread_cond_wait(&req->req_cond, &req->req_mutex);

	ret = MPI_Wait(&req->request, status);

	MPI_Type_free(&req->datatype);

	pthread_mutex_unlock(&req->req_mutex);

	return ret;
}

int starpu_mpi_test(struct starpu_mpi_req_s *req, int *flag, MPI_Status *status)
{
	int ret = 0;

	pthread_mutex_lock(&req->req_mutex);

	if (req->submitted)
	{
		ret = MPI_Test(&req->request, flag, status);

		if (*flag)
			MPI_Type_free(&req->datatype);
	}
	else {
		*flag = 0;
	}

	pthread_mutex_unlock(&req->req_mutex);

	return ret;
}

/*
 *	Requests
 */

void handle_request(struct starpu_mpi_req_s *req)
{
	STARPU_ASSERT(req);

	pthread_mutex_lock(&req->req_mutex);

	starpu_sync_data_with_mem(req->data_handle, req->mode);

	/* submit the request to MPI */
	req->handle_new(req);

	/* perhaps somebody is waiting or trying to test */
	req->submitted = 1;
	pthread_cond_broadcast(&req->req_cond);

	pthread_mutex_unlock(&req->req_mutex);
}

static void submit_mpi_req(struct starpu_mpi_req_s *req)
{
	pthread_mutex_lock(&mutex);
	pthread_mutex_lock(&req->req_mutex);

	starpu_mpi_req_list_push_front(new_requests, req);

	pthread_cond_broadcast(&req->req_cond);

	pthread_mutex_unlock(&req->req_mutex);
	pthread_mutex_unlock(&mutex);
}

/*
 *	Progression loop
 */

void *progress_thread_func(void *arg __attribute__((unused)))
{
	/* notify the main thread that the progression thread is ready */
	pthread_mutex_lock(&mutex);
	running = 1;
	pthread_cond_signal(&cond);
	pthread_mutex_unlock(&mutex);

	pthread_mutex_lock(&mutex);
	while (running) {
		pthread_cond_wait(&cond, &mutex);
		if (!running)
			break;		

		while (!starpu_mpi_req_list_empty(new_requests))
		{
			/* get one request */
			struct starpu_mpi_req_s *req;
			req = starpu_mpi_req_list_pop_back(new_requests);

			/* handling a request is likely to block for a while
			 * (on a sync_data_with_mem call), we want to let the
			 * application submit requests in the meantime, so we
			 * release the lock.  */
			pthread_mutex_unlock(&mutex);

			/* handle that request */
			STARPU_ASSERT(req);
			req->handle_new(req);

			pthread_mutex_lock(&mutex);
		}

		pthread_mutex_unlock(&mutex);
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

	/* requests that have not be submitted to MPI yet */
	new_requests = starpu_mpi_req_list_new();
	/* requests that are already submitted and which are not completed yet */
	pending_requests = starpu_mpi_req_list_new();

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

	return 0;
}
