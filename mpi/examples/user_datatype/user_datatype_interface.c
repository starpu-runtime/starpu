/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2022 Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
	struct starpu_my_data my_data;
	struct starpu_my_data my_data2 = {.d = 77, .c = 'x'};
	starpu_data_handle_t my_handle1;
	starpu_data_handle_t my_handle2;
	starpu_data_handle_t my_handle3;

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

	if (rank == 0)
	{
		my_data.d = 42;
		my_data.c = 'n';
	}
	else
	{
		my_data.d = 0;
		my_data.c = 'z';
	}

	starpu_my_data_register(&my_handle1, STARPU_MAIN_RAM, &my_data2);
	starpu_my_data_register(&my_handle2, STARPU_MAIN_RAM, &my_data2);
	starpu_my_data_register(&my_handle3, STARPU_MAIN_RAM, &my_data);
	starpu_mpi_barrier(MPI_COMM_WORLD);

	if (rank == 0)
	{
		ret = starpu_mpi_send(my_handle1, 1, 10, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
		ret = starpu_mpi_send(my_handle2, 1, 12, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
		ret = starpu_mpi_send(my_handle3, 1, 14, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
	}
	else if (rank == 1)
	{
		starpu_mpi_req req;

		ret = starpu_task_insert(&starpu_my_data_display_codelet, STARPU_VALUE, "node1 initial value", strlen("node1 initial value")+1, STARPU_R, my_handle3, 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		ret = starpu_mpi_irecv(my_handle3, &req, 0, 14, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv");
		ret = starpu_mpi_recv(my_handle2, 0, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		ret = starpu_mpi_recv(my_handle1, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		ret = starpu_mpi_wait(&req, MPI_STATUS_IGNORE);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");
		ret = starpu_task_insert(&starpu_my_data_display_codelet, STARPU_VALUE, "node1 rceived value", strlen("node1 rceived value")+1, STARPU_R, my_handle3, 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	starpu_task_wait_for_all();
	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	starpu_mpi_barrier(MPI_COMM_WORLD);

	starpu_data_unregister(my_handle1);
	starpu_data_unregister(my_handle2);
	starpu_data_unregister(my_handle3);

	starpu_my_data_shutdown();
	starpu_mpi_shutdown();

	return 0;
}
