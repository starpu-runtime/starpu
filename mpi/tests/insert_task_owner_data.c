/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

	*x0 += 1;
	*x1 *= *x1;
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_RW},
	.model = &starpu_perfmodel_nop,
};

int main(int argc, char **argv)
{
	int rank, size, err;
	int x[2];
	int ret, i;
	starpu_data_handle_t data_handles[2];
	int values[2];

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (starpu_cpu_worker_get_count() == 0)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_mpi_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	if (rank == 0)
	{
		x[0] = 11;
		starpu_variable_data_register(&data_handles[0], STARPU_MAIN_RAM, (uintptr_t)&x[0], sizeof(x[0]));
		starpu_variable_data_register(&data_handles[1], -1, (uintptr_t)NULL, sizeof(x[1]));
	}
	else if (rank == 1)
	{
		x[1] = 12;
		starpu_variable_data_register(&data_handles[0], -1, (uintptr_t)NULL, sizeof(x[0]));
		starpu_variable_data_register(&data_handles[1], STARPU_MAIN_RAM, (uintptr_t)&x[1], sizeof(x[1]));
	}
	else
	{
		starpu_variable_data_register(&data_handles[0], -1, (uintptr_t)NULL, sizeof(x[0]));
		starpu_variable_data_register(&data_handles[1], -1, (uintptr_t)NULL, sizeof(x[1]));
	}

	starpu_mpi_data_register(data_handles[0], 0, 0);
	starpu_mpi_data_register(data_handles[1], 1, 1);

	err = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
				     STARPU_RW, data_handles[0], STARPU_RW, data_handles[1],
				     STARPU_EXECUTE_ON_DATA, data_handles[1],
				     0);
	assert(err == 0);
	starpu_task_wait_for_all();

	for(i=0 ; i<2 ; i++)
	{
		starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD, data_handles[i], 0, NULL, NULL);
		if (rank == 0)
		{
			starpu_data_acquire(data_handles[i], STARPU_R);
			values[i] = *((int *)starpu_data_get_local_ptr(data_handles[i]));
			starpu_data_release(data_handles[i]);
		}
	}
	ret = 0;
	if (rank == 0)
	{
		FPRINTF(stderr, "[%d][local ptr] VALUES: %d %d\n", rank, values[0], values[1]);
		if (values[0] != 12 || values[1] != 144)
		{
			ret = EXIT_FAILURE;
		}
	}

	starpu_data_unregister(data_handles[0]);
	starpu_data_unregister(data_handles[1]);

	starpu_mpi_shutdown();

	return ret;
}

