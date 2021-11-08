/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static void read_func(void *descr[], void *_args)
{
	(void)_args;
	int *a = (void*) STARPU_VARIABLE_GET_PTR(descr[0]);

	FPRINTF_MPI(stderr, "x = %d\n", *a);
}

static struct starpu_codelet read_codelet =
{
	.cpu_funcs = {read_func},
	.nbuffers = 1,
	.modes = {STARPU_R},
#ifdef STARPU_SIMGRID
	.model = &starpu_perfmodel_nop,
#endif
	.name = "read_codelet"
};

static void write_func(void *descr[], void *_args)
{
	int rank, *a;

	a = (void*) STARPU_VARIABLE_GET_PTR(descr[0]);
	starpu_codelet_unpack_args(_args, &rank);

	*a = rank+12;
	FPRINTF_MPI(stderr, "x = %d rank=%d\n", *a, rank);
}

static struct starpu_codelet write_codelet =
{
	.cpu_funcs = {write_func},
	.nbuffers = 1,
	.modes = {STARPU_W},
#ifdef STARPU_SIMGRID
	.model = &starpu_perfmodel_nop,
#endif
	.name = "write_codelet"
};

int main(int argc, char **argv)
{
	int ret, rank, size, node;
	starpu_data_handle_t handle;
	int var=42;
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size<3)
	{
		FPRINTF(stderr, "We need more than 2 processes.\n");
		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return rank == 0 ? STARPU_TEST_SKIPPED : 0;
	}

	if (rank==0)
		starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&var, sizeof(var));
	else
		starpu_variable_data_register(&handle, -1, (uintptr_t)NULL, sizeof(var));
	starpu_mpi_data_register(handle, 42, 0);

	for(node=1 ; node<size ; node++)
	{
		starpu_mpi_task_insert(MPI_COMM_WORLD, &read_codelet, STARPU_R, handle, STARPU_EXECUTE_ON_NODE, node, 0);
	}

	for(node=1 ; node<size ; node++)
	{
		starpu_mpi_data_set_rank(handle, node);
		starpu_mpi_task_insert(MPI_COMM_WORLD, &write_codelet, STARPU_W, handle, STARPU_VALUE, &rank, sizeof(rank), STARPU_EXECUTE_ON_NODE, node, 0);
	}

	starpu_data_unregister(handle);

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
