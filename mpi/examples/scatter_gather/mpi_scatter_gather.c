/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Centre National de la Recherche Scientifique
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

/* Returns the MPI node number where data indexes index is */
int my_distrib(int x, int y, int nb_nodes) {
        return (x+y) % nb_nodes;
}

void cpu_codelet(void *descr[], void *_args)
{
	float *block;
	unsigned nx = STARPU_MATRIX_GET_NY(descr[0]);
	unsigned ld = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned i,j;
	int rank;
	float factor;

	block = (float *)STARPU_MATRIX_GET_PTR(descr[0]);
        starpu_unpack_cl_args(_args, &rank, &factor);

	fprintf(stderr,"rank %d factor %f\n", rank, factor);
	for (j = 0; j < nx; j++)
	{
		for (i = 0; i < nx; i++)
		{
			block[j+i*ld] *= factor;
		}
	}
}

static starpu_codelet cl =
{
	.where = STARPU_CPU,
	.cpu_func = cpu_codelet,
	.nbuffers = 1
};


int main(int argc, char **argv)
{
        int rank, nodes;
	float ***bmat;
        starpu_data_handle **data_handles;

	unsigned i,j,x,y;

	unsigned nblocks=4;
	unsigned block_size=1;
	unsigned size = nblocks*block_size;
	unsigned ld = size / nblocks;

	starpu_init(NULL);
	starpu_mpi_initialize_extended(&rank, &nodes);

	if (rank == 0)
	{
		/* Allocate the matrix */
		int block_number=100;
		bmat = malloc(nblocks * sizeof(float *));
		for(x=0 ; x<nblocks ; x++)
		{
			bmat[x] = malloc(nblocks * sizeof(float *));
			for(y=0 ; y<nblocks ; y++)
			{
				float value=1.0;
				starpu_malloc((void **)&bmat[x][y], block_size*block_size*sizeof(float));
				for (i = 0; i < block_size; i++)
				{
					for (j = 0; j < block_size; j++)
					{
						bmat[x][y][j +i*block_size] = block_number + value;
						value++;
					}
				}
				block_number += 100;
			}
		}
	}

#if 0
	// Print matrix
	if (rank == 0)
	{
		for(x=0 ; x<nblocks ; x++)
		{
			for(y=0 ; y<nblocks ; y++)
			{
				for (j = 0; j < block_size; j++)
				{
					for (i = 0; i < block_size; i++)
					{
						fprintf(stderr, "%2.2f\t", bmat[x][y][j+i*block_size]);
					}
					fprintf(stderr,"\n");
				}
				fprintf(stderr,"\n");
			}
		}
	}
#endif

	/* Allocate data handles and register data to StarPU */
        data_handles = malloc(nblocks*sizeof(starpu_data_handle *));
        for(x = 0; x < nblocks ;  x++)
	{
		data_handles[x] = malloc(nblocks*sizeof(starpu_data_handle));
                for (y = 0; y < nblocks; y++)
		{
			int mpi_rank = my_distrib(x, y, nodes);
			if (rank == 0)
				starpu_matrix_data_register(&data_handles[x][y], 0, (uintptr_t)bmat[x][y],
							    ld, size/nblocks, size/nblocks, sizeof(float));
			else {
				if ((mpi_rank == rank) || ((rank == mpi_rank+1 || rank == mpi_rank-1)))
				{
					/* I own that index, or i will need it for my computations */
					fprintf(stderr, "[%d] Owning or neighbor of data[%d][%d]\n", rank, x, y);
					starpu_matrix_data_register(&data_handles[x][y], -1, (uintptr_t)NULL,
								    ld, size/nblocks, size/nblocks, sizeof(float));
				}
				else
				{
					/* I know it's useless to allocate anything for this */
					data_handles[x][y] = NULL;
				}
			}
                        if (data_handles[x][y])
			{
                                starpu_data_set_rank(data_handles[x][y], mpi_rank);
			}
                }
        }

	/* Scatter the matrix among the nodes */
	if (rank == 0)
	{
		for(x = 0; x < nblocks ;  x++)
		{
			for (y = 0; y < nblocks; y++)
			{
				if (data_handles[x][y])
				{
					int owner = starpu_data_get_rank(data_handles[x][y]);
					if (owner != 0)
					{
						fprintf(stderr, "[%d] Sending data[%d][%d] to %d\n", rank, x, y, owner);
						starpu_mpi_isend_detached(data_handles[x][y], owner, owner, MPI_COMM_WORLD, NULL, NULL);
					}
				}
			}
		}
	}
	else {
		for(x = 0; x < nblocks ;  x++)
		{
			for (y = 0; y < nblocks; y++)
			{
				if (data_handles[x][y])
				{
					int owner = starpu_data_get_rank(data_handles[x][y]);
					if (owner == rank)
					{
						fprintf(stderr, "[%d] Receiving data[%d][%d] from %d\n", rank, x, y, 0);
						starpu_mpi_irecv_detached(data_handles[x][y], 0, rank, MPI_COMM_WORLD, NULL, NULL);
					}
				}
			}
		}
	}

	/* Calculation */
	float factor=10.0;
	for(x = 0; x < nblocks ;  x++)
	{
		for (y = 0; y < nblocks; y++)
		{
			int mpi_rank = my_distrib(x, y, nodes);
			if (mpi_rank == rank)
			{
				fprintf(stderr,"[%d] Computing on data[%d][%d]\n", rank, x, y);
				starpu_insert_task(&cl,
						   STARPU_VALUE, &rank, sizeof(rank),
						   STARPU_VALUE, &factor, sizeof(factor),
						   STARPU_RW, data_handles[x][y],
						   0);
				starpu_task_wait_for_all();
			}
			factor+=10.0;
		}
	}

	/* Gather the matrix on main node */
	if (rank == 0)
	{
		for(x = 0; x < nblocks ;  x++)
		{
			for (y = 0; y < nblocks; y++)
			{
				if (data_handles[x][y])
				{
					int owner = starpu_data_get_rank(data_handles[x][y]);
					if (owner != 0)
					{
						fprintf(stderr, "[%d] Receiving data[%d][%d] from %d\n", rank, x, y, owner);
						starpu_mpi_irecv_detached(data_handles[x][y], owner, owner, MPI_COMM_WORLD, NULL, NULL);
					}
				}
			}
		}
	}
	else {
		for(x = 0; x < nblocks ;  x++)
		{
			for (y = 0; y < nblocks; y++)
			{
				if (data_handles[x][y])
				{
					int owner = starpu_data_get_rank(data_handles[x][y]);
					if (owner == rank)
					{
						fprintf(stderr, "[%d] Sending data[%d][%d] to %d\n", rank, x, y, 0);
						starpu_mpi_isend_detached(data_handles[x][y], 0, rank, MPI_COMM_WORLD, NULL, NULL);
					}
				}
			}
		}
	}

	// Print matrix
	if (rank == 0)
	{
		for(x=0 ; x<nblocks ; x++)
		{
			for(y=0 ; y<nblocks ; y++)
			{
				starpu_data_unregister(data_handles[x][y]);
				for (j = 0; j < block_size; j++)
				{
					for (i = 0; i < block_size; i++)
					{
						fprintf(stderr, "%2.2f\t", bmat[x][y][j+i*block_size]);
					}
					fprintf(stderr,"\n");
				}
				fprintf(stderr,"\n");
			}
		}
	}

	// Free memory
        for(x = 0; x < nblocks ;  x++)
	{
		free(data_handles[x]);
	}
        free(data_handles);
	if (rank == 0)
	{
		for(x=0 ; x<nblocks ; x++)
		{
			for(y=0 ; y<nblocks ; y++)
			{
				starpu_free((void *)bmat[x][y]);
			}
			free(bmat[x]);
		}
		free(bmat);
	}


	starpu_mpi_shutdown();
	starpu_shutdown();
	return 0;
}
