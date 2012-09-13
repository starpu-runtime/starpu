/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2012  Université de Bordeaux 1
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
	//return (x+y) % nb_nodes;
	return (x%dblockx)+(y%dblocky)*dblockx;
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

	if (dblockx == -1 || dblocky == -1)
	{
	     int factor;
	     dblockx = nodes;
	     dblocky = 1;
	     for(factor=sqrt(nodes) ; factor>1 ; factor--)
	     {
		  if (nodes % factor == 0)
		  {
		       dblockx = nodes/factor;
		       dblocky = factor;
		       break;
		  }
	     }
	}

	unsigned i,j,x,y;
	bmat = malloc(nblocks * sizeof(float *));
	for(x=0 ; x<nblocks ; x++)
	{
		bmat[x] = malloc(nblocks * sizeof(float *));
		for(y=0 ; y<nblocks ; y++)
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

	double timing, flops;
	dw_cholesky(bmat, size, size/nblocks, nblocks, rank, nodes, &timing, &flops);

	starpu_mpi_shutdown();

	if (display)
	{
		printf("[%d] Results :\n", rank);
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

	float *rmat = malloc(size*size*sizeof(float));
	for(x=0 ; x<nblocks ; x++)
	{
		for(y=0 ; y<nblocks ; y++)
		{
			for (i = 0; i < BLOCKSIZE; i++)
			{
				for (j = 0; j < BLOCKSIZE; j++)
				{
					rmat[j+(y*BLOCKSIZE)+(i+(x*BLOCKSIZE))*size] = bmat[x][y][j +i*BLOCKSIZE];
				}
			}
		}
	}

	fprintf(stderr, "[%d] compute explicit LLt ...\n", rank);
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
			if (i > j)
			{
				rmat[j+i*size] = 0.0f; // debug
			}
		}
	}
	float *test_mat = malloc(size*size*sizeof(float));
	STARPU_ASSERT(test_mat);

	SSYRK("L", "N", size, size, 1.0f,
			rmat, size, 0.0f, test_mat, size);

	fprintf(stderr, "[%d] comparing results ...\n", rank);
	if (display)
	{
		for (j = 0; j < size; j++)
		{
			for (i = 0; i < size; i++)
			{
				if (i <= j)
				{
					printf("%2.2f\t", test_mat[j +i*size]);
				}
				else
				{
					printf(".\t");
				}
			}
			printf("\n");
		}
	}

	int correctness = 1;
	for(x = 0; x < nblocks ;  x++)
	{
		for (y = 0; y < nblocks; y++)
		{
			int mpi_rank = my_distrib(x, y, nodes);
			if (mpi_rank == rank)
			{
				for (i = (size/nblocks)*x ; i < (size/nblocks)*x+(size/nblocks); i++)
				{
					for (j = (size/nblocks)*y ; j < (size/nblocks)*y+(size/nblocks); j++)
					{
						if (i <= j)
						{
							float orig = (1.0f/(1.0f+i+j)) + ((i == j)?1.0f*size:0.0f);
							float err = abs(test_mat[j +i*size] - orig);
							if (err > 0.00001)
							{
								fprintf(stderr, "[%d] Error[%u, %u] --> %2.2f != %2.2f (err %2.2f)\n", rank, i, j, test_mat[j +i*size], orig, err);
								correctness = 0;
								flops = 0;
								break;
							}
						}
					}
				}
			}
		}
	}

	for(x=0 ; x<nblocks ; x++)
	{
		for(y=0 ; y<nblocks ; y++)
		{
			starpu_free((void *)bmat[x][y]);
		}
		free(bmat[x]);
	}
	free(bmat);
	free(rmat);
	free(test_mat);

	starpu_helper_cublas_shutdown();
	starpu_shutdown();

	assert(correctness);

	if (rank == 0)
	{
		fprintf(stdout, "Computation time (in ms): %2.2f\n", timing/1000);
		fprintf(stdout, "Synthetic GFlops : %2.2f\n", (flops/timing/1000.0f));
	}

	return 0;
}
