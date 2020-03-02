/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <math.h>
#include "helper.h"

void func_cpu(void *descr[], void *_args)
{
	(void)_args;
	unsigned *matrix = (unsigned *)STARPU_MATRIX_GET_PTR(descr[0]);
	int nx = (int)STARPU_MATRIX_GET_NX(descr[0]);
	int ny = (int)STARPU_MATRIX_GET_NY(descr[0]);
	int ld = (int)STARPU_MATRIX_GET_LD(descr[0]);

	int i, j;
	unsigned sum=0;

	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			sum += matrix[i+j*ld];
		}
	}
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			matrix[i+j*ld] = sum;///(nx*ny);
		}
	}
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 1,
#ifdef STARPU_SIMGRID
	.model = &starpu_perfmodel_nop,
#endif
	.modes = {STARPU_RW}
};

#define SIZE 6
#define BLOCKS 3

/* Returns the MPI node number where data indexes index is */
int my_distrib(int x, int y, int nb_nodes)
{
	return (x + y) % nb_nodes;
}


int main(int argc, char **argv)
{
	int rank, size, x, y;
	int ret, value=0;
	unsigned matrix[SIZE*SIZE];
	starpu_data_handle_t data_handles[SIZE][SIZE];

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	for(x = 0; x < SIZE; x++)
	{
		for (y = 0; y < SIZE; y++)
		{
			matrix[x+y*SIZE] = rank*100 + value;
			value++;
		}
	}
#if 1
	for(x = 0; x < SIZE; x++)
	{
		FPRINTF(stdout, "[%d] ", rank);
		for (y = 0; y < SIZE; y++)
		{
			FPRINTF(stdout, "%3u ", matrix[x+y*SIZE]);
		}
		FPRINTF(stdout, "\n");
	}
#endif

	for(x = 0; x < BLOCKS ; x++)
	{
		for (y = 0; y < BLOCKS; y++)
		{
			int mpi_rank = my_distrib(x, y, size);
			if (mpi_rank == rank)
			{
				//FPRINTF(stderr, "[%d] Owning data[%d][%d]\n", rank, x, y);
				starpu_matrix_data_register(&data_handles[x][y], STARPU_MAIN_RAM, (uintptr_t)&(matrix[((SIZE/BLOCKS)*x) + ((SIZE/BLOCKS)*y) * SIZE]),
							    SIZE, SIZE/BLOCKS, SIZE/BLOCKS, sizeof(unsigned));
			}
			else
			{
				/* I don't own this index, but will need it for my computations */
				//FPRINTF(stderr, "[%d] Neighbour of data[%d][%d]\n", rank, x, y);
				starpu_matrix_data_register(&data_handles[x][y], -1, (uintptr_t)&(matrix[((SIZE/BLOCKS)*x) + ((SIZE/BLOCKS)*y) * SIZE]),
							    SIZE, SIZE/BLOCKS, SIZE/BLOCKS, sizeof(unsigned));
			}
			if (data_handles[x][y])
			{
				starpu_mpi_data_register(data_handles[x][y], (y*BLOCKS)+x, mpi_rank);
			}
		}
	}

	for(x = 0; x < BLOCKS; x++)
	{
		for (y = 0; y < BLOCKS; y++)
		{
			ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
						     STARPU_RW, data_handles[x][y],
						     0);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
		}
	}

	FPRINTF(stderr, "Waiting ...\n");
	starpu_task_wait_for_all();

	for(x = 0; x < BLOCKS; x++)
	{
		for (y = 0; y < BLOCKS; y++)
		{
			if (data_handles[x][y])
				starpu_data_unregister(data_handles[x][y]);
		}
	}

	starpu_mpi_shutdown();

#if 1
	for(x = 0; x < SIZE; x++)
	{
		FPRINTF(stdout, "[%d] ", rank);
		for (y = 0; y < SIZE; y++)
		{
			FPRINTF(stdout, "%3u ", matrix[x+y*SIZE]);
		}
		FPRINTF(stdout, "\n");
	}
#endif

	return 0;
}
