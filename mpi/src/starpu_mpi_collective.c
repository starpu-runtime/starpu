/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2012  Centre National de la Recherche Scientifique
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

#include <mpi.h>
#include <starpu.h>
#include <starpu_mpi.h>
#include <starpu_mpi_private.h>

struct _callback_arg
{
	void (*callback)(void *);
	void *arg;
	int nb;
	int count;
};

void _callback_collective(void *arg)
{
	struct _callback_arg *callback_arg = arg;
	callback_arg->nb ++;
	if (callback_arg->nb == callback_arg->count)
	{
		callback_arg->callback(callback_arg->arg);
		free(callback_arg);
	}
}

int starpu_mpi_scatter_detached(starpu_data_handle_t *data_handles, int count, int root, MPI_Comm comm, void (*scallback)(void *), void *sarg, void (*rcallback)(void *), void *rarg)
{
	int rank;
	int x;
	struct _callback_arg *callback_arg;
	void (*callback_func)(void *);

	MPI_Comm_rank(comm, &rank);

	callback_func = _callback_collective;
	callback_arg = malloc(sizeof(struct _callback_arg));
	callback_arg->count = 0;
	callback_arg->nb = 0;
	callback_arg->callback = (rank == root) ? scallback : rcallback;
	callback_arg->arg = (rank == root) ? sarg : rarg;
	if (callback_arg->callback == NULL)
	{
		free(callback_arg);
		callback_arg = NULL;
		callback_func = NULL;
	}

	if (callback_arg)
	{
		for(x = 0; x < count ;  x++)
		{
			if (data_handles[x])
			{
				int owner = starpu_data_get_rank(data_handles[x]);
				int mpi_tag = starpu_data_get_tag(data_handles[x]);
				STARPU_ASSERT(mpi_tag >= 0);
				if ((rank == root) && (owner != root))
				{
					callback_arg->count ++;
				}
				if ((rank != root) && (owner == rank))
				{
					callback_arg->count ++;
				}
			}
		}
	}

	for(x = 0; x < count ;  x++)
	{
		if (data_handles[x])
		{
			int owner = starpu_data_get_rank(data_handles[x]);
			int mpi_tag = starpu_data_get_tag(data_handles[x]);
			STARPU_ASSERT(mpi_tag >= 0);
			if ((rank == root) && (owner != root))
			{
				//fprintf(stderr, "[%d] Sending data[%d] to %d\n", rank, x, owner);
				starpu_mpi_isend_detached(data_handles[x], owner, mpi_tag, comm, callback_func, callback_arg);
			}
			if ((rank != root) && (owner == rank))
			{
				//fprintf(stderr, "[%d] Receiving data[%d] from %d\n", rank, x, root);
				starpu_mpi_irecv_detached(data_handles[x], root, mpi_tag, comm, callback_func, callback_arg);
			}
		}
	}
	return 0;
}

int starpu_mpi_gather_detached(starpu_data_handle_t *data_handles, int count, int root, MPI_Comm comm, void (*scallback)(void *), void *sarg, void (*rcallback)(void *), void *rarg)
{
	int rank;
	int x;
	struct _callback_arg *callback_arg;
	void (*callback_func)(void *);

	MPI_Comm_rank(comm, &rank);

#ifdef STARPU_DEVEL
#warning TODO: callback_arg needs to be free-ed
#endif
	callback_func = _callback_collective;
	callback_arg = malloc(sizeof(struct _callback_arg));
	callback_arg->count = 0;
	callback_arg->nb = 0;
	callback_arg->callback = (rank == root) ? scallback : rcallback;
	callback_arg->arg = (rank == root) ? sarg : rarg;
	if (callback_arg->callback == NULL)
	{
		free(callback_arg);
		callback_arg = NULL;
		callback_func = NULL;
	}

	if (callback_arg)
	{
		for(x = 0; x < count ;  x++)
		{
			if (data_handles[x])
			{
				int owner = starpu_data_get_rank(data_handles[x]);
				int mpi_tag = starpu_data_get_tag(data_handles[x]);
				STARPU_ASSERT(mpi_tag >= 0);
				if ((rank == root) && (owner != root))
				{
					callback_arg->count ++;
				}
				if ((rank != root) && (owner == rank))
				{
					callback_arg->count ++;
				}
			}
		}
	}

	for(x = 0; x < count ;  x++)
	{
		if (data_handles[x])
		{
			int owner = starpu_data_get_rank(data_handles[x]);
			int mpi_tag = starpu_data_get_tag(data_handles[x]);
			STARPU_ASSERT(mpi_tag >= 0);
			if ((rank == root) && (owner != root))
			{
				//fprintf(stderr, "[%d] Receiving data[%d] from %d\n", rank, x, owner);
				starpu_mpi_irecv_detached(data_handles[x], owner, mpi_tag, comm, callback_func, callback_arg);
			}
			if ((rank != root) && (owner == rank))
			{
				//fprintf(stderr, "[%d] Sending data[%d] to %d\n", rank, x, root);
				starpu_mpi_isend_detached(data_handles[x], root, mpi_tag, comm, callback_func, callback_arg);
			}
		}
	}
	return 0;
}

