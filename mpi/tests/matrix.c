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

void func_cpu(void *descr[], void *_args)
{
	(void)_args;
	unsigned *A = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *X = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);
	unsigned *Y = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[2]);

	FPRINTF_MPI(stderr, "VALUES: Y=%3u A=%3u X=%3u\n", *Y, *A, *X);
	*Y = *Y + *A * *X;
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_RW},
	.model = &starpu_perfmodel_nop,
};

#define N 4

int main(int argc, char **argv)
{
	int rank, n;
	int ret;
	unsigned A[N];
	unsigned X[N];
	unsigned Y;
	starpu_data_handle_t data_A[N];
	starpu_data_handle_t data_X[N];
	starpu_data_handle_t data_Y;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);

	if (starpu_cpu_worker_get_count() == 0)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_mpi_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	for(n = 0; n < N; n++)
	{
		A[n] = (n+1)*10;
		X[n] = n+1;
	}
	Y = 0;

	FPRINTF_MPI(stderr, "A = ");
	for(n = 0; n < N; n++)
	{
		FPRINTF(stderr, "%u ", A[n]);
	}
	FPRINTF(stderr, "\n");
	FPRINTF_MPI(stderr, "X = ");
	for(n = 0; n < N; n++)
	{
		FPRINTF(stderr, "%u ", X[n]);
	}
	FPRINTF(stderr, "\n");

	for(n = 0; n < N; n++)
	{
		if (rank == n%2)
			starpu_variable_data_register(&data_A[n], STARPU_MAIN_RAM, (uintptr_t)&A[n], sizeof(unsigned));
		else
			starpu_variable_data_register(&data_A[n], -1, (uintptr_t)NULL, sizeof(unsigned));
		starpu_mpi_data_register(data_A[n], n+100, n%2);
		FPRINTF_MPI(stderr, "Registering A[%d] to %p with tag %d and node %d\n", n, data_A[n], n+100, n%2);

		if (rank == n%2)
			starpu_variable_data_register(&data_X[n], STARPU_MAIN_RAM, (uintptr_t)&X[n], sizeof(unsigned));
		else
			starpu_variable_data_register(&data_X[n], -1, (uintptr_t)NULL, sizeof(unsigned));
		starpu_mpi_data_register(data_X[n], n+200, n%2);
		FPRINTF_MPI(stderr, "Registering X[%d] to %p with tag %d and node %d\n", n, data_X[n], n+200, n%2);
	}
	if (rank == 0)
		starpu_variable_data_register(&data_Y, STARPU_MAIN_RAM, (uintptr_t)&Y, sizeof(unsigned));
	else
		starpu_variable_data_register(&data_Y, -1, (uintptr_t)NULL, sizeof(unsigned));
	starpu_mpi_data_register(data_Y, 10, 0);
	FPRINTF_MPI(stderr, "Registering Y to %p with tag %d and node %d\n", data_Y, 10, 0);

	for(n = 0; n < N; n++)
	{
		ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
					     STARPU_R, data_A[n],
					     STARPU_R, data_X[n],
					     STARPU_RW, data_Y,
					     STARPU_EXECUTE_ON_DATA, data_A[n],
					     0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
	}

	FPRINTF(stderr, "Waiting ...\n");
	starpu_task_wait_for_all();

	for(n = 0; n < N; n++)
	{
		starpu_data_unregister(data_A[n]);
		starpu_data_unregister(data_X[n]);
	}
	starpu_data_unregister(data_Y);

	starpu_mpi_shutdown();

	FPRINTF(stdout, "[%d] Y=%u\n", rank, Y);

#ifndef STARPU_SIMGRID
	if (rank == 0)
	{
		STARPU_ASSERT_MSG(Y==300, "Error when calculating Y=%u\n", Y);
	}
#endif

	return 0;
}
