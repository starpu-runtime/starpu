/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_mpi.h>
#include <starpu_mpi_private.h>

static void starpu_mpi_unlock_tag_callback(void *arg)
{
	starpu_tag_t *tagptr = arg;

	starpu_tag_notify_from_apps(*tagptr);

	free(tagptr);
}

int starpu_mpi_isend_detached_unlock_tag_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm, starpu_tag_t tag)
{
	starpu_tag_t *tagptr;
	_STARPU_MPI_MALLOC(tagptr, sizeof(starpu_tag_t));
	*tagptr = tag;

	return starpu_mpi_isend_detached_prio(data_handle, dest, data_tag, prio, comm, starpu_mpi_unlock_tag_callback, tagptr);
}

int starpu_mpi_isend_detached_unlock_tag(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, starpu_tag_t tag)
{
	return starpu_mpi_isend_detached_unlock_tag_prio(data_handle, dest, data_tag, 0, comm, tag);
}

int starpu_mpi_irecv_detached_unlock_tag(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, starpu_tag_t tag)
{
	starpu_tag_t *tagptr;
	_STARPU_MPI_MALLOC(tagptr, sizeof(starpu_tag_t));
	*tagptr = tag;

	return starpu_mpi_irecv_detached(data_handle, source, data_tag, comm, starpu_mpi_unlock_tag_callback, tagptr);
}

struct arg_array
{
	int array_size;
	starpu_tag_t tag;
};

static void starpu_mpi_array_unlock_callback(void *_arg)
{
	struct arg_array *arg = _arg;

	int remaining = STARPU_ATOMIC_ADD(&arg->array_size, -1);

	if (remaining == 0)
	{
		starpu_tag_notify_from_apps(arg->tag);
		free(arg);
	}
}

int starpu_mpi_isend_array_detached_unlock_tag_prio(unsigned array_size, starpu_data_handle_t *data_handle, int *dest, starpu_mpi_tag_t *data_tag, int *prio, MPI_Comm *comm, starpu_tag_t tag)
{
	if (!array_size)
		return 0;
	struct arg_array *arg;
	_STARPU_MPI_MALLOC(arg, sizeof(struct arg_array));

	arg->array_size = array_size;
	arg->tag = tag;

	unsigned elem;
	for (elem = 0; elem < array_size; elem++)
	{
		int p = 0;
		if (prio)
			p = prio[elem];
		starpu_mpi_isend_detached_prio(data_handle[elem], dest[elem], data_tag[elem], p, comm[elem], starpu_mpi_array_unlock_callback, arg);
	}

	return 0;
}

int starpu_mpi_isend_array_detached_unlock_tag(unsigned array_size, starpu_data_handle_t *data_handle, int *dest, starpu_mpi_tag_t *data_tag, MPI_Comm *comm, starpu_tag_t tag)
{
	return starpu_mpi_isend_array_detached_unlock_tag_prio(array_size, data_handle, dest, data_tag, NULL, comm, tag);
}

int starpu_mpi_irecv_array_detached_unlock_tag(unsigned array_size, starpu_data_handle_t *data_handle, int *source, starpu_mpi_tag_t *data_tag, MPI_Comm *comm, starpu_tag_t tag)
{
	if (!array_size)
		return 0;
	struct arg_array *arg;
	_STARPU_MPI_MALLOC(arg, sizeof(struct arg_array));

	arg->array_size = array_size;
	arg->tag = tag;

	unsigned elem;
	for (elem = 0; elem < array_size; elem++)
	{
		starpu_mpi_irecv_detached(data_handle[elem], source[elem], data_tag[elem], comm[elem], starpu_mpi_array_unlock_callback, arg);
	}

	return 0;
}
