/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_mpi_private.h>
#include <starpu_util.h>

/* measure the amount of data transfers between each pair of MPI nodes */
static size_t *comm_amount;
static int world_size;
static int stats_enabled=0;
static double time_init;

void _starpu_mpi_comm_amounts_init(MPI_Comm comm)
{
	stats_enabled = starpu_get_env_number("STARPU_COMM_STATS");
	if (stats_enabled == -1)
	{
		stats_enabled = 0;
	}

	if (stats_enabled == 0)
		return;

	_STARPU_DISP("Warning: StarPU is executed with STARPU_COMM_STATS=1, which slows down a bit\n");

	starpu_mpi_comm_size(comm, &world_size);
	_STARPU_MPI_DEBUG(1, "allocating for %d nodes\n", world_size);

	_STARPU_MPI_CALLOC(comm_amount, world_size, sizeof(size_t));
	time_init = starpu_timing_now();
}

void _starpu_mpi_comm_amounts_shutdown()
{
	if (stats_enabled == 0)
		return;
	free(comm_amount);
}

void _starpu_mpi_comm_amounts_inc(MPI_Comm comm, unsigned dst, MPI_Datatype datatype, int count)
{
	int src, size;

	if (stats_enabled == 0)
		return;

	starpu_mpi_comm_rank(comm, &src);
	MPI_Type_size(datatype, &size);

	_STARPU_MPI_DEBUG(1, "[%d] adding %d to %d\n", src, count*size, dst);

	comm_amount[dst] += count*size;
}

void starpu_mpi_comm_amounts_retrieve(size_t *comm_amounts)
{
	if (stats_enabled == 0)
		return;
	memcpy(comm_amounts, comm_amount, world_size * sizeof(size_t));
}

void _starpu_mpi_comm_amounts_display(FILE *stream, int node)
{
	int dst;
	size_t sum = 0;

	if (stats_enabled == 0)
		return;

	double time = starpu_timing_now() - time_init;

	for (dst = 0; dst < world_size; dst++)
	{
		sum += comm_amount[dst];
	}

	fprintf(stream, "\n[starpu_comm_stats][%d] TOTAL:\t%f B\t%f MB\t %f B/s\t %f MB/s\n", node, (float)sum, (float)sum/1024/1024, (float)sum/(float)time, (float)sum/1204/1024/(float)time);

	for (dst = 0; dst < world_size; dst++)
	{
		if (comm_amount[dst])
			fprintf(stream, "[starpu_comm_stats][%d:%d]\t%f B\t%f MB\t %f B/s\t %f MB/s\n",
				node, dst, (float)comm_amount[dst], ((float)comm_amount[dst])/(1024*1024),
				(float)comm_amount[dst]/(float)time, ((float)comm_amount[dst])/(1024*1024)/(float)time);
	}
}

