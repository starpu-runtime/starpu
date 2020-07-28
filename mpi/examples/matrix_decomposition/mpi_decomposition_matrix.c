/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
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

#include "mpi_cholesky.h"

/* Returns the MPI node number where data indexes index is */
int my_distrib(int y, int x, int nb_nodes)
{
	(void)nb_nodes;
	//return (x+y) % nb_nodes;
	return (x%dblockx)+(y%dblocky)*dblockx;
}


void matrix_display(float ***bmat, int rank, int nodes)
{
	int n;

	if (!display)
		return;

	starpu_mpi_barrier(MPI_COMM_WORLD);

	for (n = 0; n < rank; n++)
		starpu_mpi_barrier(MPI_COMM_WORLD);

	unsigned y;
	printf("[%d] Input :\n", rank);

	for(y=0 ; y<nblocks ; y++)
	{
		unsigned x;
		for(x=0 ; x<=y ; x++)
		{
			if (my_distrib(y, x, nodes) == rank)
			{
				unsigned j;
				printf("Block %u,%u :\n", x, y);
				for (j = 0; j < BLOCKSIZE; j++)
				{
					unsigned i;
					for (i = 0; i < BLOCKSIZE; i++)
					{
						if (x < y || i <= j)
						{
							printf("%2.2f\t", bmat[y][x][j +i*BLOCKSIZE]);
						}
						else
						{
							printf(".\t");
						}
					}
					printf("\n");
				}
			}
		}
	}

	starpu_mpi_barrier(MPI_COMM_WORLD);

	for (n = rank+1; n < nodes; n++)
		starpu_mpi_barrier(MPI_COMM_WORLD);
}

/* Note: bmat is indexed by bmat[m][n][mm+nn*BLOCKSIZE],
 * i.e. the content of the tiles is column-major, but the array of tiles is
 * row-major to keep the m,n notation everywhere */
void matrix_init(float ****bmat, int rank, int nodes, int alloc_everywhere)
{
	unsigned nn,mm,m,n;

	*bmat = malloc(nblocks * sizeof(float **));
	for(m=0 ; m<nblocks ; m++)
	{
		(*bmat)[m] = malloc(nblocks * sizeof(float *));
		for(n=0 ; n<nblocks ; n++)
		{
			int mpi_rank = my_distrib(m, n, nodes);
			if (alloc_everywhere || (mpi_rank == rank))
			{
				starpu_malloc((void **)&(*bmat)[m][n], BLOCKSIZE*BLOCKSIZE*sizeof(float));
				if (mpi_rank == rank)
				for (nn = 0; nn < BLOCKSIZE; nn++)
				{
					for (mm = 0; mm < BLOCKSIZE; mm++)
					{
#ifndef STARPU_SIMGRID
						(*bmat)[m][n][mm +nn*BLOCKSIZE] = (1.0f/(1.0f+(nn+(m*BLOCKSIZE)+mm+(n*BLOCKSIZE)))) + ((nn+(m*BLOCKSIZE) == mm+(n*BLOCKSIZE))?1.0f*size:0.0f);
						//mat[mm +nn*size] = ((nn == mm)?1.0f*size:0.0f);
#endif
					}
				}
			}
		}
	}
}

void matrix_free(float ****bmat, int rank, int nodes, int alloc_everywhere)
{
	unsigned m, n;

	for(m=0 ; m<nblocks ; m++)
	{
		for(n=0 ; n<nblocks ; n++)
		{
			int mpi_rank = my_distrib(m, n, nodes);
			if (alloc_everywhere || (mpi_rank == rank))
			{
				starpu_free((void *)(*bmat)[m][n]);
			}
		}
		free((*bmat)[m]);
	}
	free(*bmat);
}

