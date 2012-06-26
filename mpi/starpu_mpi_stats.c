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

/* measure the amount of data transfers between each pair of MPI nodes */
#ifdef STARPU_COMM_STATS
static size_t **comm_amount;
static int world_size;
#endif /* STARPU_COMM_STATS */

void _starpu_mpi_comm_amounts_init()
{
#ifdef STARPU_COMM_STATS
	int i;

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	comm_amount = (size_t **) calloc(1, world_size * sizeof(size_t *));
	for(i=0 ; i<world_size ; i++)
	{
		comm_amount[i] = (size_t *) calloc(1, world_size * sizeof(size_t));
	}
#endif /* STARPU_COMM_STATS */
}

void _starpu_mpi_comm_amounts_free()
{
#ifdef STARPU_COMM_STATS
	int i;
	for(i=0 ; i<world_size ; i++)
	{
		free(comm_amount[i]);
	}
	free(comm_amount);
#endif /* STARPU_COMM_STATS */
}

void _starpu_mpi_comm_amounts_inc(MPI_Comm comm  __attribute__ ((unused)),
				  unsigned dst  __attribute__ ((unused)), size_t size  __attribute__ ((unused)))
{
#ifdef STARPU_COMM_STATS
	int src;
	MPI_Comm_rank(comm, &src);
	comm_amount[src][dst] += size;
#endif /* STARPU_COMM_STATS */
}

void _starpu_mpi_comm_amounts_display()
{
#ifdef STARPU_COMM_STATS
	unsigned src, dst;

	size_t sum = 0;

	for (dst = 0; dst < world_size; dst++)
		for (src = 0; src < world_size; src++)
		{
			sum += comm_amount[src][dst];
		}

	fprintf(stderr, "\nCommunication transfers stats:\nTOTAL transfers %f B\t%f MB\n", (float)sum, (float)sum/1024/1024);

	for (dst = 0; dst < world_size; dst++)
		for (src = 0; src < world_size; src++)
		{
			if (comm_amount[src][dst])
			{
				fprintf(stderr, "\t%d <-> %d\t%f B\t%f MB\n",
					src, dst, (float)comm_amount[src][dst] + (float)comm_amount[dst][src],
					((float)comm_amount[src][dst] + (float)comm_amount[dst][src])/(1024*1024));
				fprintf(stderr, "\t\t%d -> %d\t%f B\t%f MB\n",
					src, dst, (float)comm_amount[src][dst],
					((float)comm_amount[src][dst])/(1024*1024));
				fprintf(stderr, "\t\t%d -> %d\t%f B\t%f MB\n",
					dst, src, (float)comm_amount[dst][src],
					((float)comm_amount[dst][src])/(1024*1024));
			}
		}
#endif /* STARPU_COMM_STATS */
}

