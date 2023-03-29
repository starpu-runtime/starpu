/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2023-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "helper.h"

#define X 3
#define Y 4

int main(int argc, char **argv)
{
	int size, rank, mpi_init;
	int ret=0;
	int x, y;
	struct starpu_conf conf;
	int matrix[X][Y];
	starpu_data_handle_t data_handles[X][Y];

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, &conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	int64_t mintag = starpu_mpi_tags_allocate(X*Y);

	for(x = 0; x < X; x++)
	{
		for (y = 0; y < Y; y++)
		{
			int tag = y*Y + x;
			matrix[x][y] = tag;
			starpu_variable_data_register(&data_handles[x][y], STARPU_MAIN_RAM, (uintptr_t)&matrix[x][y], sizeof(matrix[x][y]));
			starpu_mpi_data_register(data_handles[x][y], mintag + tag, 0);
		}
        }

	// Here we can use the data

	for(x = 0; x < X; x++)
	{
		for (y = 0; y < Y; y++)
		{
			starpu_data_unregister(data_handles[x][y]);
		}
	}

	starpu_mpi_tags_free(mintag);

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();
	return rank == 0 ? ret : 0;
}
