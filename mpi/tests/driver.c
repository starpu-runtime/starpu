/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

int main(int argc, char **argv)
{
	int ret, rank, size, i;
	starpu_data_handle_t tab_handle[4];
	int values[4];
	starpu_mpi_req request[2] = {NULL, NULL};
	int mpi_init;

	setenv("STARPU_MPI_DRIVER_CALL_FREQUENCY", "1", 1);
	setenv("STARPU_MPI_DRIVER_TASK_FREQUENCY", "10", 1);

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size%2 != 0)
	{
		FPRINTF_MPI(stderr, "We need a even number of processes.\n");
		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

	for(i=0 ; i<4 ; i++)
	{
		if (i<3 || rank%2)
		{
			// all data are registered on all nodes, but the 4th data which is not registered on the receiving node
			values[i] = (rank+1) * (i+1);
			starpu_variable_data_register(&tab_handle[i], STARPU_MAIN_RAM, (uintptr_t)&values[i], sizeof(values[i]));
			starpu_mpi_data_register(tab_handle[i], i, rank);
		}
	}

	int other_rank = rank%2 == 0 ? rank+1 : rank-1;

	FPRINTF_MPI(stderr, "rank %d exchanging with rank %d\n", rank, other_rank);

	if (rank%2)
	{
		FPRINTF_MPI(stderr, "Sending values %d and %d to node %d\n", values[0], values[3], other_rank);
		// this data will be received as an early registered data
		starpu_mpi_isend(tab_handle[0], &request[0], other_rank, 0, MPI_COMM_WORLD);
		// this data will be received as an early UNregistered data
		starpu_mpi_isend(tab_handle[3], &request[1], other_rank, 3, MPI_COMM_WORLD);

		starpu_mpi_send(tab_handle[1], other_rank, 1, MPI_COMM_WORLD);
		starpu_mpi_recv(tab_handle[2], other_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else
	{
		starpu_mpi_recv(tab_handle[1], other_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		starpu_mpi_send(tab_handle[2], other_rank, 2, MPI_COMM_WORLD);

		// we register the data
		starpu_variable_data_register(&tab_handle[3], -1, (uintptr_t)NULL, sizeof(int));
		starpu_mpi_data_register(tab_handle[3], 3, rank);
		starpu_mpi_irecv(tab_handle[3], &request[1], other_rank, 3, MPI_COMM_WORLD);
		starpu_mpi_irecv(tab_handle[0], &request[0], other_rank, 0, MPI_COMM_WORLD);
	}

	int finished=0;
	while (!finished)
	{
		for(i=0 ; i<2 ; i++)
		{
			if (request[i])
			{
				int flag;
				MPI_Status status;
				starpu_mpi_test(&request[i], &flag, &status);
				if (flag)
					FPRINTF_MPI(stderr, "request[%d] = %d %p\n", i, flag, request[i]);
			}
		}
		finished = request[0] == NULL && request[1] == NULL;
#ifdef STARPU_SIMGRID
		starpu_sleep(0.001);
#endif
	}

	if (rank%2 == 0)
	{
		void *ptr0;
		void *ptr3;

		starpu_data_acquire(tab_handle[0], STARPU_RW);
		ptr0 = starpu_data_get_local_ptr(tab_handle[0]);
		starpu_data_release(tab_handle[0]);

		starpu_data_acquire(tab_handle[3], STARPU_RW);
		ptr3 = starpu_data_get_local_ptr(tab_handle[3]);
		starpu_data_release(tab_handle[3]);

		ret = (*((int *)ptr0) == (other_rank+1)*1) && (*((int *)ptr3) == (other_rank+1)*4);
		ret = !ret;
		FPRINTF_MPI(stderr, "[%s] Received values %d and %d from node %d\n", ret?"FAILURE":"SUCCESS", *((int *)ptr0), *((int *)ptr3), other_rank);
	}

	for(i=0 ; i<4 ; i++)
		starpu_data_unregister(tab_handle[i]);

	starpu_mpi_shutdown();

	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
