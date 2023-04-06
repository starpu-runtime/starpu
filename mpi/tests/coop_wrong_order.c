/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void task_cpu_func(void *descr[], void *args)
{
	int *var = (int*) STARPU_VARIABLE_GET_PTR(descr[0]);
	int val;

	printf("running task\n");

	starpu_codelet_unpack_args(args, &val);
	*var = val;
}

static struct starpu_codelet cl =
{
	.cpu_funcs = { task_cpu_func },
	.cpu_funcs_name = { "task_cpu_func" },
	.nbuffers = 1,
	.flags = STARPU_CODELET_SIMGRID_EXECUTE,
	.modes = { STARPU_W },
};

int main(int argc, char **argv)
{
	int ret, rank, size;
	starpu_data_handle_t handle;
	int var = -1;
	int mpi_init;
	MPI_Status status;
	struct starpu_conf conf;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, &conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t) &var, sizeof(var));

	if (rank == 0)
	{
		int val, n;

		val = 42;
		ret = starpu_task_insert(&cl, STARPU_W, handle, STARPU_VALUE, &val, sizeof(val), 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		starpu_mpi_coop_sends_data_handle_nb_sends(handle, size-1);

		for(n = 1 ; n < size ; n++)
		{
			FPRINTF_MPI(stderr, "sending data to %d with tag 1\n", n);
			ret = starpu_mpi_isend_detached(handle, n, 1, MPI_COMM_WORLD, NULL, NULL);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend_detached");
		}

		val = 43;
		ret = starpu_task_insert(&cl, STARPU_W, handle, STARPU_VALUE, &val, sizeof(val), 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		/* The task previously inserted should be enough to detect the coop,
		 * but to be sure, indicate the number of sends requests before really
		 * sending the data: */
		starpu_mpi_coop_sends_data_handle_nb_sends(handle, size-1);

		for(n = 1 ; n < size ; n++)
		{
			FPRINTF_MPI(stderr, "sending data to %d with tag 0\n", n);
			ret = starpu_mpi_isend_detached(handle, n, 0, MPI_COMM_WORLD, NULL, NULL);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend_detached");
		}
	}
	else
	{
		FPRINTF_MPI(stderr, "waiting for data with tag 0\n");
		ret = starpu_mpi_recv(handle, 0, 0, MPI_COMM_WORLD, &status);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		starpu_data_acquire(handle, STARPU_R);
		FPRINTF_MPI(stderr, "received data: %d\n", var);
		STARPU_ASSERT(var == 43);

		starpu_data_release(handle);

		FPRINTF_MPI(stderr, "waiting for data with tag 1\n");
		ret = starpu_mpi_recv(handle, 0, 1, MPI_COMM_WORLD, &status);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		starpu_data_acquire(handle, STARPU_R);
		FPRINTF_MPI(stderr, "received data: %d\n", var);
		STARPU_ASSERT(var == 42);

		starpu_data_release(handle);
	}

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	starpu_mpi_barrier(MPI_COMM_WORLD);

	starpu_data_unregister(handle);

	printf("[%d] end\n", rank);

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
