/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010, 2014  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012, 2013, 2014, 2015  Centre National de la Recherche Scientifique
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
#include <unistd.h>

int main(int argc, char **argv)
{
	int ret, rank, size, i;
	starpu_data_handle_t tab_handle[4];
	starpu_mpi_req request[2] = {NULL, NULL};

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size%2 != 0)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need a even number of processes.\n");

		MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_init(NULL, NULL, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");

	for(i=0 ; i<4 ; i++)
	{
		if (i<3 || rank%2)
		{
			// all data are registered on all nodes, bu the 4th data which is not registered on the receiving node
			starpu_variable_data_register(&tab_handle[i], STARPU_MAIN_RAM, (uintptr_t)&rank, sizeof(int));
			starpu_mpi_data_register(tab_handle[i], i, rank);
		}
	}

	int other_rank = rank%2 == 0 ? rank+1 : rank-1;

	FPRINTF_MPI(stderr, "rank %d exchanging with rank %d\n", rank, other_rank);

	if (rank%2)
	{
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
	}

	for(i=0 ; i<4 ; i++)
		starpu_data_unregister(tab_handle[i]);

	starpu_mpi_shutdown();
	starpu_shutdown();

	MPI_Finalize();

	return 0;
}
