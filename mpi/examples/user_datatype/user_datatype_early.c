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
	struct starpu_my_data my1 = {.d = 11 , .c = 'a'};

	if (rank == 1)
	{
		my0.d *= 2;
		my0.c += 1;
		my1.d *= 2;
		my1.c += 1;
	}

	starpu_data_handle_t handle0;
	starpu_data_handle_t handle1;
	starpu_my_data_register(&handle0, STARPU_MAIN_RAM, &my0);
	starpu_my_data_register(&handle1, STARPU_MAIN_RAM, &my1);

	if (rank == 0)
	{
		starpu_mpi_send(handle0, 1, 10, MPI_COMM_WORLD);
		starpu_mpi_send(handle1, 1, 20, MPI_COMM_WORLD);
	}
	else if (rank == 1)
	{
		// We want handle0 to be received as early_data and as starpu_mpi_data_register() has not be called, it will be received as raw memory, and then unpacked with MPI_Unpack()
		starpu_task_insert(&starpu_my_data_display_codelet, STARPU_VALUE, "node1 handle0 init value", strlen("node1 handle0 init value")+1, STARPU_R, handle0, 0);
		starpu_task_insert(&starpu_my_data_display_codelet, STARPU_VALUE, "node1 handle1 init value", strlen("node1 handle1 init value")+1, STARPU_R, handle1, 0);
		starpu_mpi_recv(handle1, 0, 20, MPI_COMM_WORLD, NULL);
		starpu_mpi_recv(handle0, 0, 10, MPI_COMM_WORLD, NULL);
		starpu_task_insert(&starpu_my_data_display_codelet, STARPU_VALUE, "node1 handle0 received value", strlen("node1 handle0 received value")+1, STARPU_R, handle0, 0);
		starpu_task_insert(&starpu_my_data_display_codelet, STARPU_VALUE, "node1 handle1 received value", strlen("node1 handle1 received value")+1, STARPU_R, handle1, 0);
	}

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	starpu_mpi_barrier(MPI_COMM_WORLD);

	starpu_data_unregister(handle0);
	starpu_data_unregister(handle1);

	if (rank == 1)
	{
		STARPU_ASSERT_MSG(my0.d == 42 && my0.c == 'n' && my1.d == 11 && my1.c == 'a', "Incorrect received values");
	}

	starpu_my_data_shutdown();
	starpu_mpi_shutdown();

	return 0;
}
