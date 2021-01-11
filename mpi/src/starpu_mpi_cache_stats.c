/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_mpi_cache_stats.h>
#include <common/config.h>
#include <stdio.h>
#include <starpu_mpi_private.h>

/* measure the amount of data transfers between each pair of MPI nodes */
static size_t *comm_cache_amount;
static int world_size;
static int stats_enabled=0;

void _starpu_mpi_cache_stats_init(MPI_Comm comm)
{
	stats_enabled = starpu_get_env_number("STARPU_MPI_CACHE_STATS");
	if (stats_enabled == -1)
	{
		stats_enabled = 0;
	}
	if (stats_enabled == 0)
		return;

	_STARPU_DISP("Warning: StarPU is executed with STARPU_MPI_CACHE_STATS=1, which slows down a bit\n");

	starpu_mpi_comm_size(comm, &world_size);
	_STARPU_MPI_DEBUG(1, "allocating for %d nodes\n", world_size);

	_STARPU_MPI_CALLOC(comm_cache_amount, world_size, sizeof(size_t));
}

void _starpu_mpi_cache_stats_free()
{
	if (stats_enabled == 0)
		return;
	free(comm_cache_amount);
}

void _starpu_mpi_cache_stats_update(unsigned dst, starpu_data_handle_t data_handle, int count)
{
	size_t size;

	if (stats_enabled == 0)
		return;

	size = starpu_data_get_size(data_handle);

	if (count == 1)
	{
		_STARPU_MPI_MSG("[communication cache] + %10ld to   %d\n", (long)size, dst);
	}
	else // count == -1
	{
		_STARPU_MPI_MSG("[communication cache] - %10ld from %d\n", (long)size, dst);
	}

	comm_cache_amount[dst] += count * size;
}

