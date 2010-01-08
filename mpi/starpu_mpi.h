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

#ifndef __STARPU_MPI_H__
#define __STARPU_MPI_H__

#include <starpu.h>
#include <mpi.h>
#include <common/list.h>
#include <pthread.h>

LIST_TYPE(starpu_mpi_req,
	/* description of the data at StarPU level */
	starpu_data_handle data_handle;

	/* description of the data to be sent/received */
	void *ptr;
	MPI_Datatype datatype;

	/* who are we talking to ? */
	int srcdst;
	int mpi_tag;
	MPI_Comm comm;

	void (*func)(struct starpu_mpi_req_s *);

	MPI_Status *status;
	MPI_Request request;
	int *flag;

	int ret;
	pthread_mutex_t req_mutex;
	pthread_cond_t req_cond;

	unsigned submitted;
	unsigned completed;

	/* In the case of a Wait/Test request, we are going to post a request
	 * to test the completion of another request */
	struct starpu_mpi_req_s *other_request;

	/* in the case of detached requests */
	unsigned detached;
	void *callback_arg;
	void (*callback)(void *);
);

int starpu_mpi_isend(starpu_data_handle data_handle, struct starpu_mpi_req_s *req,
		int dest, int mpi_tag, MPI_Comm comm);
int starpu_mpi_irecv(starpu_data_handle data_handle, struct starpu_mpi_req_s *req,
		int source, int mpi_tag, MPI_Comm comm);
int starpu_mpi_send(starpu_data_handle data_handle,
		int dest, int mpi_tag, MPI_Comm comm);
int starpu_mpi_recv(starpu_data_handle data_handle,
		int source, int mpi_tag, MPI_Comm comm, MPI_Status *status);
int starpu_mpi_isend_detached(starpu_data_handle data_handle, struct starpu_mpi_req_s *req, int dest, int mpi_tag, MPI_Comm comm, void (*callback)(void *), void *arg);
int starpu_mpi_irecv_detached(starpu_data_handle data_handle, struct starpu_mpi_req_s *req, int source, int mpi_tag, MPI_Comm comm, void (*callback)(void *), void *arg);
int starpu_mpi_wait(struct starpu_mpi_req_s *req, MPI_Status *status);
int starpu_mpi_test(struct starpu_mpi_req_s *req, int *flag, MPI_Status *status);
int starpu_mpi_initialize(void);
int starpu_mpi_shutdown(void);

#endif // __STARPU_MPI_H__
