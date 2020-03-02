/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
	int ret, rank, size;
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size<3)
	{
		FPRINTF(stderr, "We need more than 2 processes.\n");
		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

	if (rank == 0)
	{
		int n;
		for(n=1 ; n<size ; n++)
		{
			int i, var[2];
			MPI_Status status[3];
			starpu_data_handle_t handle[2];

			FPRINTF_MPI(stderr, "receiving from node %d\n", n);
			for(i=0 ; i<2 ; i++)
				starpu_variable_data_register(&handle[i], STARPU_MAIN_RAM, (uintptr_t)&var[i], sizeof(var[i]));

			starpu_mpi_recv(handle[0], n, 42, MPI_COMM_WORLD, &status[0]);
			starpu_data_acquire(handle[0], STARPU_R);
			STARPU_ASSERT_MSG(var[0] == n, "Received incorrect value <%d> from node <%d>\n", var[0], n);
			FPRINTF_MPI(stderr, "received <%d> from node %d\n", var[0], n);
			starpu_data_release(handle[0]);

			starpu_mpi_recv(handle[0], n, 42, MPI_COMM_WORLD, &status[1]);
			starpu_mpi_recv(handle[1], n, 44, MPI_COMM_WORLD, &status[2]);
			for(i=0 ; i<2 ; i++)
				starpu_data_acquire(handle[i], STARPU_R);
			STARPU_ASSERT_MSG(var[0] == n*2, "Received incorrect value <%d> from node <%d>\n", var[0], n);
			STARPU_ASSERT_MSG(var[1] == n*4, "Received incorrect value <%d> from node <%d>\n", var[0], n);
			FPRINTF_MPI(stderr, "received <%d> and <%d> from node %d\n", var[0], var[1], n);
			for(i=0 ; i<2 ; i++)
				starpu_data_release(handle[i]);
			for(i=0 ; i<2 ; i++)
				starpu_data_unregister(handle[i]);
		}
	}
	else
	{
		int i, var[3];
		starpu_data_handle_t handle[3];

		FPRINTF_MPI(stderr, "sending to node %d\n", 0);
		var[0] = rank;
		var[1] = var[0] * 2;
		var[2] = var[0] * 4;
		for(i=0 ; i<3 ; i++)
			starpu_variable_data_register(&handle[i], STARPU_MAIN_RAM, (uintptr_t)&var[i], sizeof(var[i]));
		starpu_mpi_send(handle[0], 0, 42, MPI_COMM_WORLD);
		starpu_mpi_send(handle[1], 0, 42, MPI_COMM_WORLD);
		starpu_mpi_send(handle[2], 0, 44, MPI_COMM_WORLD);
		for(i=0 ; i<3 ; i++)
			starpu_data_unregister(handle[i]);
	}

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
