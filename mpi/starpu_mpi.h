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
	void *ptr;
	MPI_Datatype datatype;
	MPI_Request request;
);

int starpu_mpi_isend(starpu_data_handle data_handle, starpu_mpi_req_t *req,
		int dest, int mpi_tag, MPI_Comm comm,
		void (*callback)(void *));
int starpu_mpi_irecv(starpu_data_handle data_handle, starpu_mpi_req_t *req,
		int source, int mpi_tag, MPI_Comm comm,
		void (*callback)(void *));
int starpu_mpi_send(starpu_data_handle data_handle,
		int dest, int mpi_tag, MPI_Comm comm);
int starpu_mpi_recv(starpu_data_handle data_handle,
		int source, int mpi_tag, MPI_Comm comm, MPI_Status *status);
int starpu_mpi_wait(starpu_mpi_req_t *req);
int starpu_mpi_test(starpu_mpi_req_t *req, int *flag);
int starpu_mpi_initialize(void);
int starpu_mpi_shutdown(void);

#endif // __STARPU_MPI_H__
