/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Centre National de la Recherche Scientifique
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

#include <starpu_mpi_stats.h>
#include <common/config.h>
#include <stdio.h>
//#define STARPU_MPI_VERBOSE	1
#include <starpu_mpi_private.h>

/* measure the amount of data transfers between each pair of MPI nodes */
static size_t *comm_amount;
static int world_size;
static int stats_enabled=0;

void _starpu_mpi_comm_amounts_init(MPI_Comm comm)
{
	stats_enabled = starpu_get_env_number("STARPU_COMM_STATS");
	if (stats_enabled == -1)
	{
		stats_enabled = 0;
	}

	if (stats_enabled == 0) return;

	if (!getenv("STARPU_SILENT")) fprintf(stderr,"Warning: StarPU is executed with STARPU_COMM_STATS=1, which slows down a bit\n");

	MPI_Comm_size(comm, &world_size);
	_STARPU_MPI_DEBUG("allocating for %d nodes\n", world_size);

	comm_amount = (size_t *) calloc(world_size, sizeof(size_t));
}

void _starpu_mpi_comm_amounts_free()
{
	if (stats_enabled == 0) return;
	free(comm_amount);
}

void _starpu_mpi_comm_amounts_inc(MPI_Comm comm, unsigned dst, MPI_Datatype datatype, int count)
{
	int src, size;

	if (stats_enabled == 0) return;

	MPI_Comm_rank(comm, &src);
	MPI_Type_size(datatype, &size);

	_STARPU_MPI_DEBUG("[%d] adding %d to %d\n", src, count*size, dst);

	comm_amount[dst] += count*size;
}

void starpu_mpi_comm_amounts_retrieve(size_t *comm_amounts)
{
	memcpy(comm_amounts, comm_amount, world_size * sizeof(size_t));
}

void _starpu_mpi_comm_amounts_display(int node)
{
	unsigned dst;
	size_t sum = 0;

	if (stats_enabled == 0) return;

	for (dst = 0; dst < world_size; dst++)
	{
		sum += comm_amount[dst];
	}

	fprintf(stderr, "\n[%d] Communication transfers stats:\nTOTAL transfers %f B\t%f MB\n", node, (float)sum, (float)sum/1024/1024);

	for (dst = 0; dst < world_size; dst++)
	{
		if (comm_amount[dst])
		{
			fprintf(stderr, "\t%d -> %d\t%f B\t%f MB\n",
				node, dst, (float)comm_amount[dst], ((float)comm_amount[dst])/(1024*1024));
		}
	}
}

