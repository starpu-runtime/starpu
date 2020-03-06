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
#include <math.h>
#include "helper.h"
#include <starpu_mpi_cache.h>

void func_cpu(void *descr[], STARPU_ATTRIBUTE_UNUSED void *_args)
{
	int *value = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	FPRINTF_MPI(stderr, "Executing codelet with value %d\n", *value);
	*value = *value * 2;
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 1,
	.modes = {STARPU_RW}
};

int main(int argc, char **argv)
{
	int size;
	int color;
	MPI_Comm newcomm;
	int rank, newrank;
	int ret;
	unsigned val = 42;
	starpu_data_handle_t data;

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (size < 4)
        {
                if (rank == 0)
                        FPRINTF(stderr, "We need at least 4 processes.\n");

                MPI_Finalize();
                return STARPU_TEST_SKIPPED;
        }

	color = rank%2;
	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &newcomm);
	MPI_Comm_rank(newcomm, &newrank);
	FPRINTF_MPI(stderr, "[%d] color %d\n", newrank, color);

        ret = starpu_init(NULL);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
        ret = starpu_mpi_init(NULL, NULL, 0);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");

	if (newrank == 0)
	{
		val = rank+1;
		starpu_variable_data_register(&data, 0, (uintptr_t)&val, sizeof(val));
	}
	else
		starpu_variable_data_register(&data, -1, (uintptr_t)NULL, sizeof(unsigned));
	starpu_mpi_data_register_comm(data, 42, 0, newcomm);
	FPRINTF_MPI(stderr, "[%d] Registering data %p with tag %d and node %d\n", newrank, data, 42, 0);

	if (newrank == 0)
	{
		FPRINTF_MPI(stderr, "[%d] sending %d\n", newrank, rank);
		MPI_Send(&rank, 1, MPI_INT, 1, 10, newcomm);
		starpu_mpi_send(data, 1, 42, newcomm);
	}
	else
	{
		int x;
		MPI_Recv(&x, 1, MPI_INT, 0, 10, newcomm, NULL);
		FPRINTF_MPI(stderr, "[%d] received %d\n", newrank, x);
		starpu_mpi_recv(data, 0, 42, newcomm, NULL);
	}

	starpu_mpi_insert_task(newcomm, &mycodelet,
			       STARPU_RW, data,
			       STARPU_EXECUTE_ON_NODE, 1,
			       0);

	FPRINTF_MPI(stderr, "Waiting ...\n");
	starpu_task_wait_for_all();

	starpu_data_unregister(data);
	if (newrank == 0)
	{
		FPRINTF_MPI(stderr, "[%d] new value %u\n", newrank, val);
	}

	starpu_mpi_shutdown();
	starpu_shutdown();
        MPI_Finalize();
	return 0;
}
