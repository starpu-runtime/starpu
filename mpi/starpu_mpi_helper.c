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

static void starpu_mpi_unlock_tag_callback(void *arg)
{
	starpu_tag_t *tagptr = arg;

	starpu_tag_notify_from_apps(*tagptr);

	free(tagptr);
}

int starpu_mpi_isend_detached_unlock_tag(starpu_data_handle data_handle, struct starpu_mpi_req_s *req,
				int dest, int mpi_tag, MPI_Comm comm, starpu_tag_t tag)
{
	starpu_tag_t *tagptr = malloc(sizeof(starpu_tag_t));
	*tagptr = tag;
	
	return starpu_mpi_isend_detached(data_handle, req, dest, mpi_tag, comm,
						starpu_mpi_unlock_tag_callback, tagptr);
}


int starpu_mpi_irecv_detached_unlock_tag(starpu_data_handle data_handle, struct starpu_mpi_req_s *req, int source, int mpi_tag, MPI_Comm comm, starpu_tag_t tag)
{
	starpu_tag_t *tagptr = malloc(sizeof(starpu_tag_t));
	*tagptr = tag;
	
	return starpu_mpi_irecv_detached(data_handle, req, source, mpi_tag, comm,
						starpu_mpi_unlock_tag_callback, tagptr);
}
