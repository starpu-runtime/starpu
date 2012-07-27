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
#ifdef STARPU_COMM_STATS
static size_t *comm_amount;
static int world_size;
#endif /* STARPU_COMM_STATS */

void _starpu_mpi_comm_amounts_init(MPI_Comm comm)
{
#ifdef STARPU_COMM_STATS
	if (!getenv("STARPU_SILENT")) fprintf(stderr,"Warning: StarPU was configured with --enable-comm-stats, which slows down a bit\n");

	MPI_Comm_size(comm, &world_size);
	_STARPU_MPI_DEBUG("allocating for %d nodes\n", world_size);

	comm_amount = (size_t *) calloc(world_size, sizeof(size_t));
#endif /* STARPU_COMM_STATS */
}

void _starpu_mpi_comm_amounts_free()
{
#ifdef STARPU_COMM_STATS
	free(comm_amount);
#endif /* STARPU_COMM_STATS */
}

void _starpu_mpi_comm_amounts_inc(MPI_Comm comm  __attribute__ ((unused)),
				  unsigned dst  __attribute__ ((unused)),
				  MPI_Datatype datatype  __attribute__ ((unused)),
				  int count __attribute__ ((unused)))
{
#ifdef STARPU_COMM_STATS
	int src, size;

	MPI_Comm_rank(comm, &src);
	MPI_Type_size(datatype, &size);

	_STARPU_MPI_DEBUG("[%d] adding %d to %d\n", src, count*size, dst);

	comm_amount[dst] += count*size;
#endif /* STARPU_COMM_STATS */
}

void _starpu_mpi_comm_amounts_display(int node)
{
#ifdef STARPU_COMM_STATS
	unsigned dst;
	size_t sum = 0;

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
#endif /* STARPU_COMM_STATS */
}

