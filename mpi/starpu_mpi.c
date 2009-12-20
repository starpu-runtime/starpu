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

pthread_cond_t cond;
pthread_mutex_t mutex;
pthread_t progress_thread;

int starpu_mpi_isend(starpu_data_handle data_handle, starpu_mpi_req_t *req,
		int dest, int mpi_tag, MPI_Comm comm,
		void (*callback)(void *))
{
	return 0;
}

int starpu_mpi_irecv(starpu_data_handle data_handle, starpu_mpi_req_t *req,
		int source, int mpi_tag, MPI_Comm comm,
		void (*callback)(void *))
{
	return 0;
}

int starpu_mpi_recv(starpu_data_handle data_handle,
		int source, int mpi_tag, MPI_Comm comm)
{
	/* TODO test if we are blocking in a callback .. */

	starpu_sync_data_with_mem(data_handle, STARPU_W);

	void *ptr = (void *)starpu_get_vector_local_ptr(data_handle);
	
	MPI_Status status;
	MPI_Datatype datatype;
	starpu_mpi_handle_to_datatype(data_handle, &datatype);
	MPI_Recv(ptr, 1, datatype, source, mpi_tag, comm, &status);

	starpu_release_data_from_mem(data_handle);

	return 0;
}

int starpu_mpi_send(starpu_data_handle data_handle,
		int dest, int mpi_tag, MPI_Comm comm)
{
	/* TODO test if we are blocking in a callback .. */

	starpu_sync_data_with_mem(data_handle, STARPU_R);

	void *ptr = (void *)starpu_get_vector_local_ptr(data_handle);
	
	MPI_Status status;
	MPI_Datatype datatype;
	starpu_mpi_handle_to_datatype(data_handle, &datatype);
	MPI_Send(ptr, 1, datatype, dest,  mpi_tag, comm);

	starpu_release_data_from_mem(data_handle);

	return 0;
}

int starpu_mpi_wait(starpu_mpi_req_t *req)
{
	return 0;
}

int starpu_mpi_test(starpu_mpi_req_t *req, int *flag)
{
	return 0;
}

int starpu_mpi_initialize(void)
{
	return 0;
}

int starpu_mpi_shutdown(void)
{
	return 0;
}
