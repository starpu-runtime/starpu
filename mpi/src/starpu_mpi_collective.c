/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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

static
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

static
int _callback_set(int rank, starpu_data_handle_t *data_handles, int count, int root, void (*scallback)(void *), void *sarg, void (*rcallback)(void *), void *rarg, void (**callback_func)(void *), struct _callback_arg **callback_arg)
{
	void (*callback)(void *);

	callback = (rank == root) ? scallback : rcallback;
	if (*callback)
	{
		int x;

		*callback_func = _callback_collective;

		_STARPU_MPI_MALLOC(*callback_arg, sizeof(struct _callback_arg));
		(*callback_arg)->count = 0;
		(*callback_arg)->nb = 0;
		(*callback_arg)->callback = (rank == root) ? scallback : rcallback;
		(*callback_arg)->arg = (rank == root) ? sarg : rarg;

		for(x = 0; x < count ; x++)
		{
			if (data_handles[x])
			{
				int owner = starpu_mpi_data_get_rank(data_handles[x]);
				starpu_mpi_tag_t data_tag = starpu_mpi_data_get_tag(data_handles[x]);
				STARPU_ASSERT_MSG(data_tag >= 0, "Invalid tag for data handle");
				if ((rank == root) && (owner != root))
				{
					(*callback_arg)->count ++;
				}
				if ((rank != root) && (owner == rank))
				{
					(*callback_arg)->count ++;
				}
			}
		}

		if (!(*callback_arg)->count)
		{
			free(*callback_arg);
			return 1;
		}
	}

	return 0;
}

int starpu_mpi_scatter_detached(starpu_data_handle_t *data_handles, int count, int root, MPI_Comm comm, void (*scallback)(void *), void *sarg, void (*rcallback)(void *), void *rarg)
{
	int rank;
	int x;
	struct _callback_arg *callback_arg = NULL;
	void (*callback_func)(void *) = NULL;

	starpu_mpi_comm_rank(comm, &rank);

	x = _callback_set(rank, data_handles, count, root, scallback, sarg, rcallback, rarg, &callback_func, &callback_arg);
	if (x == 1)
		return 0;

	for(x = 0; x < count ; x++)
	{
		if (data_handles[x])
		{
			int owner = starpu_mpi_data_get_rank(data_handles[x]);
			starpu_mpi_tag_t data_tag = starpu_mpi_data_get_tag(data_handles[x]);
			STARPU_ASSERT_MSG(data_tag >= 0, "Invalid tag for data handle");
			if ((rank == root) && (owner != root))
			{
				//fprintf(stderr, "[%d] Sending data[%d] to %d\n", rank, x, owner);
				starpu_mpi_isend_detached(data_handles[x], owner, data_tag, comm, callback_func, callback_arg);
			}
			if ((rank != root) && (owner == rank))
			{
				//fprintf(stderr, "[%d] Receiving data[%d] from %d\n", rank, x, root);
				starpu_mpi_irecv_detached(data_handles[x], root, data_tag, comm, callback_func, callback_arg);
			}
		}
	}
	return 0;
}

int starpu_mpi_gather_detached(starpu_data_handle_t *data_handles, int count, int root, MPI_Comm comm, void (*scallback)(void *), void *sarg, void (*rcallback)(void *), void *rarg)
{
	int rank;
	int x;
	struct _callback_arg *callback_arg = NULL;
	void (*callback_func)(void *) = NULL;

	starpu_mpi_comm_rank(comm, &rank);

	x = _callback_set(rank, data_handles, count, root, scallback, sarg, rcallback, rarg, &callback_func, &callback_arg);
	if (x == 1)
		return 0;

	for(x = 0; x < count ; x++)
	{
		if (data_handles[x])
		{
			int owner = starpu_mpi_data_get_rank(data_handles[x]);
			starpu_mpi_tag_t data_tag = starpu_mpi_data_get_tag(data_handles[x]);
			STARPU_ASSERT_MSG(data_tag >= 0, "Invalid tag for data handle");
			if ((rank == root) && (owner != root))
			{
				//fprintf(stderr, "[%d] Receiving data[%d] from %d\n", rank, x, owner);
				starpu_mpi_irecv_detached(data_handles[x], owner, data_tag, comm, callback_func, callback_arg);
			}
			if ((rank != root) && (owner == rank))
			{
				//fprintf(stderr, "[%d] Sending data[%d] to %d\n", rank, x, root);
				starpu_mpi_isend_detached(data_handles[x], root, data_tag, comm, callback_func, callback_arg);
			}
		}
	}
	return 0;
}
