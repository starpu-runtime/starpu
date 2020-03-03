/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

int main(int argc, char **argv)
{
	int size, x=789;
	int rank, other_rank;
	int ret;
	starpu_data_handle_t data[2];
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (size % 2)
        {
		FPRINTF(stderr, "We need a even number of processes.\n");
		if (!mpi_init)
			MPI_Finalize();
                return STARPU_TEST_SKIPPED;
        }

	other_rank = rank%2 == 0 ? rank+1 : rank-1;
	FPRINTF_MPI(stderr, "rank %d exchanging with rank %d\n", rank, other_rank);

	if (rank % 2)
	{
		MPI_Send(&rank, 1, MPI_INT, other_rank, 10, MPI_COMM_WORLD);
		FPRINTF(stderr, "[%d] sending %d\n", rank, rank);
	}
	else
	{
		MPI_Recv(&x, 1, MPI_INT, other_rank, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		FPRINTF(stderr, "[%d] received %d\n", rank, x);
	}

        ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, NULL);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	if (rank % 2)
	{
		starpu_variable_data_register(&data[0], STARPU_MAIN_RAM, (uintptr_t)&rank, sizeof(unsigned));
		starpu_variable_data_register(&data[1], STARPU_MAIN_RAM, (uintptr_t)&rank, sizeof(unsigned));
		starpu_mpi_data_register(data[1], 22, 0);
	}
	else
		starpu_variable_data_register(&data[0], -1, (uintptr_t)NULL, sizeof(unsigned));
	starpu_mpi_data_register(data[0], 12, 0);

	if (rank % 2)
	{
		starpu_mpi_req req;
		starpu_mpi_issend(data[1], &req, other_rank, 22, MPI_COMM_WORLD);
		starpu_mpi_send(data[0], other_rank, 12, MPI_COMM_WORLD);
		starpu_mpi_wait(&req, MPI_STATUS_IGNORE);
	}
	else
	{
		int *xx;

		starpu_mpi_recv(data[0], other_rank, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		xx = (int *)starpu_variable_get_local_ptr(data[0]);
		FPRINTF_MPI(stderr, "received %d\n", *xx);
		STARPU_ASSERT_MSG(x==*xx, "Received value %d is incorrect (should be %d)\n", *xx, x);

		starpu_variable_data_register(&data[1], -1, (uintptr_t)NULL, sizeof(unsigned));
		starpu_mpi_data_register(data[1], 22, 0);
		starpu_mpi_recv(data[0],  other_rank, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		xx = (int *)starpu_variable_get_local_ptr(data[0]);
		STARPU_ASSERT_MSG(x==*xx, "Received value %d is incorrect (should be %d)\n", *xx, x);
	}

	starpu_data_unregister(data[0]);
	starpu_data_unregister(data[1]);

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();
	return 0;
}
