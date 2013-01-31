/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2012  Université de Bordeaux 1
 * Copyright (C) 2010  Mehdi Juhoor <mjuhoor@gmail.com>
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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

#include <starpu.h>
#include "mpi_cholesky_matrix.h"
#include "mpi_cholesky_params.h"
#include "mpi_cholesky_codelets.h"

void matrix_display(float ***bmat, int rank)
{
	unsigned i,j,x,y;

	if (display)
	{
		printf("[%d] Input :\n", rank);

		for(y=0 ; y<nblocks ; y++)
		{
			for(x=0 ; x<nblocks ; x++)
			{
				printf("Block %u,%u :\n", x, y);
				for (j = 0; j < BLOCKSIZE; j++)
				{
					for (i = 0; i < BLOCKSIZE; i++)
					{
						if (i <= j)
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
}

void matrix_init(float ****bmat, int rank, int nodes, int alloc_everywhere)
{
	unsigned i,j,x,y;

	*bmat = malloc(nblocks * sizeof(float *));
	for(x=0 ; x<nblocks ; x++)
	{
		(*bmat)[x] = malloc(nblocks * sizeof(float *));
		for(y=0 ; y<nblocks ; y++)
		{
			int mpi_rank = my_distrib(x, y, nodes);
			if (alloc_everywhere || (mpi_rank == rank))
			{
				starpu_malloc((void **)&(*bmat)[x][y], BLOCKSIZE*BLOCKSIZE*sizeof(float));
				for (i = 0; i < BLOCKSIZE; i++)
				{
					for (j = 0; j < BLOCKSIZE; j++)
					{
						(*bmat)[x][y][j +i*BLOCKSIZE] = (1.0f/(1.0f+(i+(x*BLOCKSIZE)+j+(y*BLOCKSIZE)))) + ((i+(x*BLOCKSIZE) == j+(y*BLOCKSIZE))?1.0f*size:0.0f);
						//mat[j +i*size] = ((i == j)?1.0f*size:0.0f);
					}
				}
			}
		}
	}
}

void matrix_free(float ****bmat, int rank, int nodes, int alloc_everywhere)
{
	unsigned x, y;

	for(x=0 ; x<nblocks ; x++)
	{
		for(y=0 ; y<nblocks ; y++)
		{
			int mpi_rank = my_distrib(x, y, nodes);
			if (alloc_everywhere || (mpi_rank == rank))
			{
				starpu_free((void *)(*bmat)[x][y]);
			}
		}
		free((*bmat)[x]);
	}
	free(*bmat);
}

