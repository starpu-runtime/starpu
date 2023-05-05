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
#include <unistd.h>
#include "helper.h"

void callback(void *arg)
{
	int *completed = arg;
	*completed = 1;
}

#define SIZE 370*000*0000

int main(int argc, char **argv)
{
	int ret, rank, size;
	int mpi_init;
	starpu_data_handle_t handle;
	char *value;
	int comm_completed=42;

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

	value = calloc(SIZE, sizeof(value[0]));
	starpu_vector_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)value, SIZE, sizeof(value[0]));

	if (rank == 1)
	{
		ret = starpu_mpi_send(handle, 0, 1, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend_detached");
	}
	else if (rank == 0)
	{
		ret = starpu_mpi_irecv_detached(handle, 1, 1, MPI_COMM_WORLD, callback, &comm_completed);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
	}

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	if (rank == 0)
	{
		if (comm_completed == 42)
		{
			FPRINTF_MPI(stderr, "comm not completed\n");
			ret = 1;
		}
		else
		{
			FPRINTF_MPI(stderr, "comm completed\n");
		}
	}
	starpu_data_unregister(handle);
	free(value);

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	if (rank == 0 && comm_completed == 42)
	{
		FPRINTF(stderr, "comm still not completed\n");
		ret = 1;
	}

	return (rank == 0) ? ret : 0;
}
