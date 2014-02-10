/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014  Centre National de la Recherche Scientifique
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

	if (stats_enabled == 0) return;

	if (!getenv("STARPU_SILENT")) fprintf(stderr,"Warning: StarPU is executed with STARPU_MPI_CACHE_STATS=1, which slows down a bit\n");

	MPI_Comm_size(comm, &world_size);
	_STARPU_MPI_DEBUG(1, "allocating for %d nodes\n", world_size);

	comm_cache_amount = (size_t *) calloc(world_size, sizeof(size_t));
}

void _starpu_mpi_cache_stats_free()
{
	if (stats_enabled == 0) return;
	free(comm_cache_amount);
}

void _starpu_mpi_cache_stats_update(int src, unsigned dst, starpu_data_handle_t data_handle, int count)
{
	size_t size;

	if (stats_enabled == 0) return;

	size = starpu_data_get_size(data_handle);

	if (count == 1)
	{
		fprintf(stderr, "[starpu_mpi_cache_stats][%d] adding %ld to %d\n", src, (long)size, dst);
	}
	else // count == -1
	{
		fprintf(stderr, "[starpu_mpi_cache_stats][%d] removing %ld from %d\n", src, (long)size, dst);
	}

	comm_cache_amount[dst] += count * size;
}

