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

void wait_CPU(void *descr[], void *args)
{
	int *var = (int*) STARPU_VARIABLE_GET_PTR(descr[0]);
	int val;

	starpu_codelet_unpack_args(args, &val);
	*var = val;
	starpu_sleep(1);
}

static struct starpu_codelet cl =
{
	.cpu_funcs = { wait_CPU },
	.cpu_funcs_name = { "wait_CPU" },
	.nbuffers = 1,
	.flags = STARPU_CODELET_SIMGRID_EXECUTE,
	.modes = { STARPU_W },
};

int main(int argc, char **argv)
{
	int ret, rank, size;
	starpu_data_handle_t handle;
	int var=-1;
	int mpi_init;
	MPI_Status status;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&var, sizeof(var));

	if (rank == 0)
	{
		int val, n;

		val = 42;
		starpu_task_insert(&cl, STARPU_W, handle, STARPU_VALUE, &val, sizeof(val), 0);

		for(n = 1 ; n < size ; n++)
		{
			FPRINTF_MPI(stderr, "sending data to %d\n", n);
			starpu_mpi_isend_detached(handle, n, 0, MPI_COMM_WORLD, NULL, NULL);
		}

		val = 43;
		starpu_task_insert(&cl, STARPU_W, handle, STARPU_VALUE, &val, sizeof(val), 0);

		for(n = 1 ; n < size ; n++)
		{
			FPRINTF_MPI(stderr, "sending data to %d\n", n);
			starpu_mpi_isend_detached(handle, n, 0, MPI_COMM_WORLD, NULL, NULL);
		}
	}
	else
	{
		starpu_mpi_recv(handle, 0, 0, MPI_COMM_WORLD, &status);
		starpu_data_acquire(handle, STARPU_R);
		STARPU_ASSERT(var == 42);
		starpu_data_release(handle);

		starpu_mpi_recv(handle, 0, 0, MPI_COMM_WORLD, &status);
		starpu_data_acquire(handle, STARPU_R);
		STARPU_ASSERT(var == 43);
		starpu_data_release(handle);
		FPRINTF_MPI(stderr, "received data\n");
	}

	starpu_data_unregister(handle);

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
