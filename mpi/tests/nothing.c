/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This program does nothing. It waits until it is interrupted by the user.
 * Useful to check binding while StarPU is running.
 */

#include <starpu_mpi.h>
#include <unistd.h>
#include "helper.h"


int main(int argc, char **argv)
{
	int ret, rank, worldsize;
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);
	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_pause(); // our program will only wait, no need to stress cores by polling workers

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &worldsize);

	starpu_mpi_barrier(MPI_COMM_WORLD);

	char hostname[65];
	gethostname(hostname, sizeof(hostname));

	printf("[rank %d on %s] ready to wait !\n", rank, hostname);

	if (rank == 0)
	{
		printf("You can now check if thread binding is correct, for instance.\n");
	}

	fflush(stdout);

	while(1)
	{
		sleep(1);
	}

	// TODO: maybe better handle the user interruption ?


	starpu_resume();

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
