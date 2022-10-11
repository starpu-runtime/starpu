/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&var, sizeof(var));

	if (rank == 0)
	{
		int val, i;

		starpu_data_handle_t* ack_handles = malloc((size-1) * sizeof(starpu_data_handle_t));
		int* acks = calloc(size - 1, sizeof(int));
		starpu_mpi_req* ack_reqs = calloc((size-1), sizeof(starpu_mpi_req));

		val = 42;
		ret = starpu_task_insert(&cl, STARPU_W, handle, STARPU_VALUE, &val, sizeof(val), 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		/* The task previously inserted should be enough to detect the coop,
		 * but to be sure, indicate the number of sends requests before really
		 * sending the data: */
		starpu_mpi_coop_sends_data_handle_nb_sends(handle, size-1);

		for(i = 1 ; i < size ; i++)
		{
			starpu_variable_data_register(&ack_handles[i-1], STARPU_MAIN_RAM, (uintptr_t) &acks[i-1], sizeof(int));
			ret = starpu_mpi_irecv(ack_handles[i-1], &ack_reqs[i-1], i, 0, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv");

			FPRINTF_MPI(stderr, "sending data to %d\n", i);
			ret = starpu_mpi_isend_detached(handle, i, 0, MPI_COMM_WORLD, NULL, NULL);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend_detached");
		}

		for(i = 0; i < size-1; i++)
		{
			ret = starpu_mpi_wait(&ack_reqs[i], NULL);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");
			starpu_data_acquire(ack_handles[i], STARPU_R);

			STARPU_ASSERT(acks[i] == 1);

			starpu_data_release(ack_handles[i]);
			starpu_data_unregister(ack_handles[i]);
		}

		free(ack_handles);
		free(acks);
		free(ack_reqs);
	}
	else
	{
		starpu_data_handle_t ack_handle;
		int ack = 1;

		starpu_variable_data_register(&ack_handle, STARPU_MAIN_RAM, (uintptr_t) &ack, sizeof(ack));

		ret = starpu_mpi_recv(handle, 0, 0, MPI_COMM_WORLD, &status);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		starpu_data_acquire(handle, STARPU_R);
		printf("[%d] received data: %d\n", rank, var);
		STARPU_ASSERT(var == 42);

		ret = starpu_mpi_send(ack_handle, 0, 0, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
		starpu_data_unregister(ack_handle);

		starpu_data_release(handle);

		FPRINTF_MPI(stderr, "received data\n");
	}

	starpu_data_unregister(handle);

	printf("[%d] end\n", rank);

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
