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
	unsigned *x = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *y = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);

	FPRINTF(stdout, "VALUES: %u %u\n", *x, *y);
	*x = (*x + *y) / 2;
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_R},
	.model = &starpu_perfmodel_nop,
};

#define X     4
#define Y     5

/* Returns the MPI node number where data indexes index is */
int my_distrib(int x, int y, int nb_nodes)
{
	return (x + y) % nb_nodes;
}


int main(int argc, char **argv)
{
	int rank, size, x, y;
	int value=0, ret;
	unsigned matrix[X][Y];
	starpu_data_handle_t data_handles[X][Y];

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	for(x = 0; x < X; x++)
	{
		for (y = 0; y < Y; y++)
		{
			matrix[x][y] = (rank+1)*10 + value;
			value++;
		}
	}
#if 0
	for(x = 0; x < X; x++)
	{
		FPRINTF(stdout, "[%d] ", rank);
		for (y = 0; y < Y; y++)
		{
			FPRINTF(stdout, "%3d ", matrix[x][y]);
		}
		FPRINTF(stdout, "\n");
	}
#endif

	for(x = 0; x < X; x++)
	{
		for (y = 0; y < Y; y++)
		{
			int mpi_rank = my_distrib(x, y, size);
			if (mpi_rank == rank)
			{
				//FPRINTF(stderr, "[%d] Owning data[%d][%d]\n", rank, x, y);
				starpu_variable_data_register(&data_handles[x][y], STARPU_MAIN_RAM, (uintptr_t)&(matrix[x][y]), sizeof(unsigned));
			}
			else
			{
				/* I don't own this index, but will need it for my computations */
				//FPRINTF(stderr, "[%d] Neighbour of data[%d][%d]\n", rank, x, y);
				starpu_variable_data_register(&data_handles[x][y], -1, (uintptr_t)NULL, sizeof(unsigned));
			}
			if (data_handles[x][y])
			{
				starpu_mpi_data_register(data_handles[x][y], (y*X)+x, mpi_rank);
			}
		}
	}

	ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[1][1], STARPU_R, data_handles[0][1], 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
	ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[3][1], STARPU_R, data_handles[0][1], 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
	ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[0][1], STARPU_R, data_handles[0][0], 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
	ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[3][1], STARPU_R, data_handles[0][1], 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");

	FPRINTF(stderr, "Waiting ...\n");
	starpu_task_wait_for_all();

	for(x = 0; x < X; x++)
	{
		for (y = 0; y < Y; y++)
		{
			if (data_handles[x][y])
				starpu_data_unregister(data_handles[x][y]);
		}
	}
	starpu_mpi_shutdown();

#if 0
	for(x = 0; x < X; x++)
	{
		FPRINTF(stdout, "[%d] ", rank);
		for (y = 0; y < Y; y++)
		{
			FPRINTF(stdout, "%3d ", matrix[x][y]);
		}
		FPRINTF(stdout, "\n");
	}
#endif

	return 0;
}
