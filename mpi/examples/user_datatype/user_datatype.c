/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "my_interface.h"

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

int main(int argc, char **argv)
{
	int rank, nodes;
	int ret=0;
	int compare=0;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &nodes);

	if (nodes < 2 || (starpu_cpu_worker_get_count() == 0))
	{
		if (rank == 0)
		{
			if (nodes < 2)
				fprintf(stderr, "We need at least 2 processes.\n");
			else
				fprintf(stderr, "We need at least 1 CPU.\n");
		}
		starpu_mpi_shutdown();
		return 77;
	}

	struct starpu_my_data my0 = {.d = 42 , .c = 'n'};
	struct starpu_my_data my1 = {.d = 98 , .c = 'z'};

	starpu_data_handle_t handle0;
	starpu_data_handle_t handle1;

	if (rank == 1)
	{
		my0.d = 0;
		my0.c = 'z';
	}

	starpu_my_data_register(&handle0, STARPU_MAIN_RAM, &my0);
	starpu_my_data_register(&handle1, -1, &my1);

	// Send data directly with MPI
	if (rank == 0)
	{
		MPI_Datatype mpi_datatype;
		_starpu_my_data_datatype_allocate(&mpi_datatype);
		MPI_Send(&my0, 1, mpi_datatype, 1, 42, MPI_COMM_WORLD);
		starpu_my_data_datatype_free(&mpi_datatype);
	}
	else if (rank == 1)
	{
		MPI_Datatype mpi_datatype;
		MPI_Status status;
		struct starpu_my_data myx;
		_starpu_my_data_datatype_allocate(&mpi_datatype);
		MPI_Recv(&myx, 1, mpi_datatype, 0, 42, MPI_COMM_WORLD, &status);
		FPRINTF(stderr, "[mpi] Received value: '%c' %d\n", myx.c, myx.d);
		starpu_my_data_datatype_free(&mpi_datatype);
		STARPU_ASSERT_MSG(myx.d == 42 && myx.c == 'n', "Incorrect received value\n");
	}

	if (rank == 0)
	{
		struct starpu_my_data myx = {.d = 98 , .c = 'z'};
		starpu_data_handle_t handlex;
		starpu_my_data_register(&handlex, STARPU_MAIN_RAM, &myx);
		starpu_mpi_send(handlex, 1, 10, MPI_COMM_WORLD);
		starpu_data_unregister(handlex);
	}
	else if (rank == 1)
	{
		MPI_Status status;
		struct starpu_my_data myx = {.d = 11 , .c = 'a'};
		starpu_data_handle_t handlex;
		starpu_my_data_register(&handlex, STARPU_MAIN_RAM, &myx);
		starpu_mpi_recv(handlex, 0, 10, MPI_COMM_WORLD, &status);
		starpu_data_unregister(handlex);
		FPRINTF(stderr, "[starpu mpi] myx.d=%d myx.c=%c\n", myx.d, myx.c);
		STARPU_ASSERT_MSG(myx.d == 98 && myx.c == 'z', "Incorrect received value\n");
	}

	if (rank == 0)
	{
		int *compare_ptr = &compare;

		starpu_task_insert(&starpu_my_data_display_codelet, STARPU_VALUE, "node0 initial value", strlen("node0 initial value")+1, STARPU_R, handle0, 0);
		starpu_mpi_isend_detached(handle0, 1, 10, MPI_COMM_WORLD, NULL, NULL);
		starpu_mpi_irecv_detached(handle1, 1, 20, MPI_COMM_WORLD, NULL, NULL);

		starpu_task_insert(&starpu_my_data_display_codelet, STARPU_VALUE, "node0 received value", strlen("node0 received value")+1, STARPU_R, handle1, 0);
		starpu_task_insert(&starpu_my_data_compare_codelet, STARPU_R, handle0, STARPU_R, handle1, STARPU_VALUE, &compare_ptr, sizeof(compare_ptr), 0);
	}
	else if (rank == 1)
	{
		starpu_task_insert(&starpu_my_data_display_codelet, STARPU_VALUE, "node1 initial value", strlen("node1 initial value")+1, STARPU_R, handle0, 0);
		starpu_mpi_irecv_detached(handle0, 0, 10, MPI_COMM_WORLD, NULL, NULL);
		starpu_task_insert(&starpu_my_data_display_codelet, STARPU_VALUE, "node1 received value", strlen("node1 received value")+1, STARPU_R, handle0, 0);
		starpu_mpi_isend_detached(handle0, 0, 20, MPI_COMM_WORLD, NULL, NULL);
	}

	starpu_mpi_barrier(MPI_COMM_WORLD);
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);

	starpu_data_unregister(handle0);
	starpu_data_unregister(handle1);

	starpu_my_data_shutdown();
	starpu_mpi_shutdown();

	if (rank == 0)
	{
		FPRINTF(stderr, "[node 0] %s\n", compare==1?"SUCCESS":"FAILURE");
	}

	return (rank == 0) ? !compare : 0;
}
