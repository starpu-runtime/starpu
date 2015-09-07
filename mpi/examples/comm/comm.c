/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015  CNRS
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
 * This example splits the whole set of communicators in subgroups,
 * all communications take place within each subgroups
 */

#include <starpu_mpi.h>
#include "../helper.h"

void func_cpu(void *descr[], STARPU_ATTRIBUTE_UNUSED void *_args)
{
	int *value = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int rank;

	starpu_codelet_unpack_args(_args, &rank);
	FPRINTF_MPI(stderr, "Executing codelet with value %d and rank %d\n", *value, rank);
	STARPU_ASSERT_MSG(*value == rank, "Received value %d is not the expected value %d\n", *value, rank);
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 1,
	.modes = {STARPU_RW}
};

int main(int argc, char **argv)
{
	int size, n, x=789;
	int color;
	MPI_Comm newcomm;
	int rank, newrank;
	int ret;
	starpu_data_handle_t data[2];

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (size < 4)
        {
		FPRINTF(stderr, "We need at least 4 processes.\n");
                MPI_Finalize();
                return STARPU_TEST_SKIPPED;
        }

	color = rank%2;
	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &newcomm);
	MPI_Comm_rank(newcomm, &newrank);
	FPRINTF(stderr, "[%d][%d] color %d\n", rank, newrank, color);

	if (newrank == 0)
	{
		FPRINTF(stderr, "[%d][%d] sending %d\n", rank, newrank, rank);
		MPI_Send(&rank, 1, MPI_INT, 1, 10, newcomm);
	}
	else if (newrank == 1)
	{
		MPI_Recv(&x, 1, MPI_INT, 0, 10, newcomm, MPI_STATUS_IGNORE);
		FPRINTF(stderr, "[%d][%d] received %d\n", rank, newrank, x);
	}

        ret = starpu_init(NULL);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
        ret = starpu_mpi_init_comm(NULL, NULL, 0, newcomm);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");

	if (newrank == 0)
	{
		starpu_variable_data_register(&data[0], STARPU_MAIN_RAM, (uintptr_t)&rank, sizeof(int));
		starpu_variable_data_register(&data[1], STARPU_MAIN_RAM, (uintptr_t)&rank, sizeof(int));
		starpu_mpi_data_register_comm(data[1], 22, 0, newcomm);
	}
	else
		starpu_variable_data_register(&data[0], -1, (uintptr_t)NULL, sizeof(int));
	starpu_mpi_data_register_comm(data[0], 12, 0, newcomm);

	if (newrank == 0)
	{
		starpu_mpi_req req[2];
		starpu_mpi_issend(data[1], &req[0], 1, 22, newcomm);
		starpu_mpi_isend(data[0], &req[1], 1, 12, newcomm);
		starpu_mpi_wait(&req[0], MPI_STATUS_IGNORE);
		starpu_mpi_wait(&req[1], MPI_STATUS_IGNORE);
	}
	else if (newrank == 1)
	{
		int *xx;

		starpu_mpi_recv(data[0], 0, 12, newcomm, MPI_STATUS_IGNORE);
		starpu_data_acquire(data[0], STARPU_RW);
		xx = (int *)starpu_variable_get_local_ptr(data[0]);
		starpu_data_release(data[0]);
		FPRINTF(stderr, "[%d][%d] received %d\n", rank, newrank, *xx);
		STARPU_ASSERT_MSG(x==*xx, "Received value %d is incorrect (should be %d)\n", *xx, x);

		starpu_variable_data_register(&data[1], -1, (uintptr_t)NULL, sizeof(int));
		starpu_mpi_data_register_comm(data[1], 22, 0, newcomm);
		starpu_mpi_recv(data[0], 0, 22, newcomm, MPI_STATUS_IGNORE);
		starpu_data_acquire(data[0], STARPU_RW);
		xx = (int *)starpu_variable_get_local_ptr(data[0]);
		starpu_data_release(data[0]);
		FPRINTF(stderr, "[%d][%d] received %d\n", rank, newrank, *xx);
		STARPU_ASSERT_MSG(x==*xx, "Received value %d is incorrect (should be %d)\n", *xx, x);
	}

	if (newrank == 0 || newrank == 1)
	{
		starpu_mpi_task_insert(newcomm, &mycodelet,
				       STARPU_RW, data[0],
				       STARPU_VALUE, &x, sizeof(x),
				       STARPU_EXECUTE_ON_NODE, 1,
				       0);

		starpu_task_wait_for_all();
		starpu_data_unregister(data[0]);
		starpu_data_unregister(data[1]);
	}

	starpu_mpi_shutdown();
	starpu_shutdown();
	MPI_Comm_free(&newcomm);
        MPI_Finalize();
	return 0;
}
