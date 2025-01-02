/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifdef STARPU_QUICK_CHECK
#  define NITER	16
#else
#  define NITER	2048
#endif
#define SIZE	16

float *tab;
starpu_data_handle_t tab_handle;

int main(int argc, char **argv)
{
	int ret, rank, size;
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size%2 != 0)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need a even number of processes.\n");

		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return rank == 0 ? STARPU_TEST_SKIPPED : 0;
	}

	tab = calloc(SIZE, sizeof(float));

	starpu_vector_data_register(&tab_handle, STARPU_MAIN_RAM, (uintptr_t)tab, SIZE, sizeof(float));

	int nloops = NITER;
	int loop;
	int other_rank = rank%2 == 0 ? rank+1 : rank-1;

	for (loop = 0; loop < nloops; loop++)
	{
		if ((loop % 2) == (rank%2))
		{
			MPI_Status status;
			starpu_mpi_req req;
			ret = starpu_mpi_isend(tab_handle, &req, other_rank, loop, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend");
			ret = starpu_mpi_wait(&req, &status);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");
		}
		else
		{
			MPI_Status status;
			ret = starpu_mpi_recv(tab_handle, other_rank, loop, MPI_COMM_WORLD, &status);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		}
	}

	starpu_data_unregister(tab_handle);
	free(tab);

	starpu_mpi_shutdown();

	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
