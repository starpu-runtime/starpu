/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * communications take place both within each subgroups and MPI_COMM_WORLD.
 */

#include <starpu_mpi.h>
#include "../helper.h"

MPI_Comm newcomm;

void func_cpu(void *descr[], void *_args)
{
	int *value = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int rank;

	starpu_codelet_unpack_args(_args, &rank);
	FPRINTF_MPI_COMM(stderr, newcomm, "Executing codelet with value %d and rank %d\n", *value, rank);
	STARPU_ASSERT_MSG(*value == rank, "Received value %d is not the expected value %d\n", *value, rank);
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &starpu_perfmodel_nop,
};

int main(int argc, char **argv)
{
	int size, x;
	int color;
	int rank, newrank;
	int ret;
	starpu_data_handle_t data[3];
	int value = 90;
	int thread_support;
	if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &thread_support) != MPI_SUCCESS)
	{
		fprintf(stderr,"MPI_Init_thread failed\n");
		exit(1);
	}
	if (thread_support == MPI_THREAD_FUNNELED)
		fprintf(stderr,"Warning: MPI only has funneled thread support, not serialized, hoping this will work\n");
	if (thread_support < MPI_THREAD_FUNNELED)
		fprintf(stderr,"Warning: MPI does not have thread support!\n");

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

	ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	ret = starpu_mpi_comm_register(newcomm);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_comm_register");

	if (rank == 0)
	{
		starpu_variable_data_register(&data[2], STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(int));
	}
	else
		starpu_variable_data_register(&data[2], -1, (uintptr_t)NULL, sizeof(int));
	starpu_mpi_data_register_comm(data[2], 44, 0, MPI_COMM_WORLD);

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
		ret = starpu_mpi_issend(data[1], &req[0], 1, 22, newcomm);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_issend");
		ret = starpu_mpi_isend(data[0], &req[1], 1, 12, newcomm);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_isend");
		ret = starpu_mpi_wait(&req[0], MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");
		ret = starpu_mpi_wait(&req[1], MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");
	}
	else if (newrank == 1)
	{
		int *xx;

		ret = starpu_mpi_recv(data[0], 0, 12, newcomm, MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		starpu_data_acquire(data[0], STARPU_R);
		xx = (int *)starpu_variable_get_local_ptr(data[0]);
		FPRINTF(stderr, "[%d][%d] received %d\n", rank, newrank, *xx);
		STARPU_ASSERT_MSG(x==*xx, "Received value %d is incorrect (should be %d)\n", *xx, x);
		starpu_data_release(data[0]);

		starpu_variable_data_register(&data[1], -1, (uintptr_t)NULL, sizeof(int));
		starpu_mpi_data_register_comm(data[1], 22, 0, newcomm);
		ret = starpu_mpi_recv(data[0], 0, 22, newcomm, MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		starpu_data_acquire(data[0], STARPU_R);
		xx = (int *)starpu_variable_get_local_ptr(data[0]);
		FPRINTF(stderr, "[%d][%d] received %d\n", rank, newrank, *xx);
		STARPU_ASSERT_MSG(x==*xx, "Received value %d is incorrect (should be %d)\n", *xx, x);
		starpu_data_release(data[0]);
	}

	if (rank == 0)
	{
		starpu_data_acquire(data[2], STARPU_R);
		int rvalue = *((int *)starpu_variable_get_local_ptr(data[2]));
		starpu_data_release(data[2]);
		FPRINTF_MPI_COMM(stderr, MPI_COMM_WORLD, "sending value %d to %d and receiving from %d\n", rvalue, 1, size-1);
		ret = starpu_mpi_send(data[2], 1, 44, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
		ret = starpu_mpi_recv(data[2], size-1, 44, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		starpu_data_acquire(data[2], STARPU_R);
		int *xx = (int *)starpu_variable_get_local_ptr(data[2]);
		FPRINTF_MPI_COMM(stderr, MPI_COMM_WORLD, "Value back is %d\n", *xx);
		STARPU_ASSERT_MSG(*xx == rvalue + (2*(size-1)), "Received value %d is incorrect (should be %d)\n", *xx, rvalue + (2*(size-1)));
		starpu_data_release(data[2]);
	}
	else
	{
		int next = (rank == size-1) ? 0 : rank+1;
		ret = starpu_mpi_recv(data[2], rank-1, 44, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		starpu_data_acquire(data[2], STARPU_RW);
		int *xx = (int *)starpu_variable_get_local_ptr(data[2]);
		FPRINTF_MPI_COMM(stderr, MPI_COMM_WORLD, "receiving %d from %d and sending %d to %d\n", *xx, rank-1, *xx+2, next);
		*xx = *xx + 2;
		starpu_data_release(data[2]);
		ret = starpu_mpi_send(data[2], next, 44, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
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
	starpu_data_unregister(data[2]);

	starpu_mpi_shutdown();
	MPI_Comm_free(&newcomm);
	MPI_Finalize();
	return 0;
}
