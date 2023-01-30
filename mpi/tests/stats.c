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

#if !defined(STARPU_HAVE_UNSETENV) || !defined(STARPU_HAVE_SETENV)
#warning unsetenv or setenv are not defined. Skipping test
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else
int main(int argc, char **argv)
{
	int ret, rank, size;
	int mpi_init;
	int value;
	starpu_data_handle_t handle;
	size_t *stats;

	unsetenv("STARPU_MPI_CACHE");
	unsetenv("STARPU_MPI_STATS");
	unsetenv("STARPU_COMM_STATS");

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size < 2)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 2 processes.\n");

		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return rank == 0 ? STARPU_TEST_SKIPPED : 0;
	}

	stats = calloc(size, sizeof(stats[0]));
	value = rank;
	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));

	if (rank == 0)
	{
		ret = starpu_mpi_send(handle, 1, 42, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
	}
	else if (rank == 1)
	{
		ret = starpu_mpi_recv(handle, 0, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
	}

	starpu_mpi_comm_stats_enable();

	if (rank == 0)
	{
		ret = starpu_mpi_send(handle, 1, 42, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
	}
	else if (rank == 1)
	{
		ret = starpu_mpi_recv(handle, 0, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
	}

	starpu_mpi_comm_stats_disable();

	if (rank == 0)
	{
		ret = starpu_mpi_send(handle, 1, 42, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
	}
	else if (rank == 1)
	{
		ret = starpu_mpi_recv(handle, 0, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
	}

	starpu_data_unregister(handle);

	starpu_mpi_comm_stats_retrieve(stats);
	if (rank == 0)
		STARPU_ASSERT_MSG(stats[1] == sizeof(int), "Comm stats are incorrect %ld != %ld\n", stats[0], (long)sizeof(int));

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
#endif
