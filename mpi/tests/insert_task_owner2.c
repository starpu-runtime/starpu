/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
	int *x0 = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int *x1 = (int *)STARPU_VARIABLE_GET_PTR(descr[1]);
	int *x2 = (int *)STARPU_VARIABLE_GET_PTR(descr[2]);
	int *y = (int *)STARPU_VARIABLE_GET_PTR(descr[3]);

	FPRINTF(stderr, "-------> CODELET VALUES: %d %d nan %d\n", *x0, *x1, *y);
	*x2 = *y;
	*y = (*x0 + *x1) * 100;
	*x1 = 12;
	FPRINTF(stderr, "-------> CODELET VALUES: %d %d %d %d\n", *x0, *x1, *x2, *y);
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 4,
	.modes = {STARPU_R, STARPU_RW, STARPU_W, STARPU_RW},
	.model = &starpu_perfmodel_nop,
};

int main(int argc, char **argv)
{
	int rank, size, err;
	int x[3], y=0;
	int oldx[3];
	int i, ret=0;
	starpu_data_handle_t data_handles[4];
	struct starpu_conf conf;
	int mpi_init;

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

	if (starpu_cpu_worker_get_count() == 0)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return rank == 0 ? STARPU_TEST_SKIPPED : 0;
	}

	if (rank == 0)
	{
		for(i=0 ; i<3 ; i++)
		{
			x[i] = 10*(i+1);
			oldx[i] = 10*(i+1);
			starpu_variable_data_register(&data_handles[i], STARPU_MAIN_RAM, (uintptr_t)&x[i], sizeof(x[i]));
		}
		y = -1;
		starpu_variable_data_register(&data_handles[3], -1, (uintptr_t)NULL, sizeof(int));
	}
	else
	{
		for(i=0 ; i<3 ; i++)
		{
			x[i] = -1;
			starpu_variable_data_register(&data_handles[i], -1, (uintptr_t)NULL, sizeof(int));
		}
		y=200;
		starpu_variable_data_register(&data_handles[3], STARPU_MAIN_RAM, (uintptr_t)&y, sizeof(int));
	}
	for(i=0 ; i<3 ; i++)
	{
		starpu_mpi_data_register(data_handles[i], i, 0);
	}
	starpu_mpi_data_register(data_handles[3], 3, 1);

	FPRINTF(stderr, "[%d][init] VALUES: %d %d %d %d\n", rank, x[0], x[1], x[2], y);

	err = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
				     STARPU_R, data_handles[0], STARPU_RW, data_handles[1],
				     STARPU_W, data_handles[2],
				     STARPU_RW, data_handles[3],
				     STARPU_EXECUTE_ON_NODE, 1, 0);
	STARPU_CHECK_RETURN_VALUE(err, "starpu_mpi_task_insert");
	starpu_task_wait_for_all();

	int *values = malloc(4 * sizeof(int));
	for(i=0 ; i<4 ; i++)
	{
		starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD, data_handles[i], 0, NULL, NULL);
		if (rank == 0)
		{
			starpu_data_acquire(data_handles[i], STARPU_R);
			values[i] = *((int *)starpu_data_get_local_ptr(data_handles[i]));
			starpu_data_release(data_handles[i]);
		}
		starpu_data_unregister(data_handles[i]);
	}
	if (rank == 0)
	{
		FPRINTF(stderr, "[%d][local ptr] VALUES: %d %d %d %d\n", rank, values[0], values[1], values[2], values[3]);
		if (values[0] != oldx[0] || values[1] != 12 || values[2] != 200 || values[3] != ((oldx[0] + oldx[1]) * 100))
		{
			FPRINTF(stderr, "[%d][error] values[0] %d != x[0] %d && values[1] %d != 12 && values[2] %d != 200 && values[3] %d != ((x[0] %d + x[1] %d) * 100)\n",
				rank, values[0], oldx[0], values[1], values[2], values[3], oldx[0], oldx[1]);
			ret = 1;
		}
		else
		{
			FPRINTF(stderr, "[%d] correct computation\n", rank);
		}
	}
	FPRINTF(stderr, "[%d][end] VALUES: %d %d %d %d\n", rank, x[0], x[1], x[2], y);

	free(values);
	starpu_mpi_shutdown();

	if (!mpi_init)
		MPI_Finalize();

	return (rank == 0) ? ret : 0;
}
