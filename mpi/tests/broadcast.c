/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013,2015,2017                           CNRS
 * Copyright (C) 2014-2015,2017-2018                      Université de Bordeaux
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

void wait_CPU(void *descr[], void *_args)
{
	(void)_args;
	int *var = (int*) STARPU_VARIABLE_GET_PTR(descr[0]);
	*var = 42;
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
	int var;
	int mpi_init;
	MPI_Status status;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_init(&argc, &argv, mpi_init);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&var, sizeof(var));

	if (rank == 0)
	{
		starpu_task_insert(&cl, STARPU_W, handle, 0);

		int n;
		for(n = 1 ; n < size ; n++)
		{
			FPRINTF_MPI(stderr, "sending data to %d\n", n);
			starpu_mpi_isend_detached(handle, n, 0, MPI_COMM_WORLD, NULL, NULL);
		}
	}
	else
	{
		starpu_mpi_recv(handle, 0, 0, MPI_COMM_WORLD, &status);
		FPRINTF_MPI(stderr, "received data\n");
	}

	starpu_data_unregister(handle);
	STARPU_ASSERT(var == 42);

	starpu_mpi_shutdown();
	starpu_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
