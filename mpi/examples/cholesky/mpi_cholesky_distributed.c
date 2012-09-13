/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2011  Université de Bordeaux 1
 * Copyright (C) 2010  Mehdi Juhoor <mjuhoor@gmail.com>
 * Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
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

#include <starpu_mpi.h>
#include "mpi_cholesky.h"
#include "mpi_cholesky_models.h"
#include "mpi_cholesky_codelets.h"

/* Returns the MPI node number where data indexes index is */
int my_distrib(int x, int y, int nb_nodes)
{
	return (x+y) % nb_nodes;
}

int main(int argc, char **argv)
{
	/* create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1)
	 * */

	float ***bmat;
	int rank, nodes, ret;

	parse_args(argc, argv);

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	starpu_mpi_initialize_extended(&rank, &nodes);
	starpu_helper_cublas_init();

	unsigned i,j,x,y;
	bmat = malloc(nblocks * sizeof(float *));
	for(x=0 ; x<nblocks ; x++)
	{
		bmat[x] = malloc(nblocks * sizeof(float *));
		for(y=0 ; y<nblocks ; y++)
		{
			int mpi_rank = my_distrib(x, y, nodes);
			if (mpi_rank == rank)
			{
				starpu_malloc((void **)&bmat[x][y], BLOCKSIZE*BLOCKSIZE*sizeof(float));
				for (i = 0; i < BLOCKSIZE; i++)
				{
					for (j = 0; j < BLOCKSIZE; j++)
					{
						bmat[x][y][j +i*BLOCKSIZE] = (1.0f/(1.0f+(i+(x*BLOCKSIZE)+j+(y*BLOCKSIZE)))) + ((i+(x*BLOCKSIZE) == j+(y*BLOCKSIZE))?1.0f*size:0.0f);
						//mat[j +i*size] = ((i == j)?1.0f*size:0.0f);
					}
				}
			}
		}
	}

	dw_cholesky(bmat, size, size/nblocks, nblocks, rank, nodes);

	starpu_mpi_shutdown();

	for(x=0 ; x<nblocks ; x++)
	{
		for(y=0 ; y<nblocks ; y++)
		{
			int mpi_rank = my_distrib(x, y, nodes);
			if (mpi_rank == rank)
			{
				starpu_free((void *)bmat[x][y]);
			}
		}
		free(bmat[x]);
	}
	free(bmat);

	starpu_helper_cublas_shutdown();
	starpu_shutdown();

	return 0;
}
