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
	(void)descr;
	(void)_args;
}

struct starpu_codelet mycodelet_r =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 1,
	.modes = {STARPU_R},
	.model = &starpu_perfmodel_nop,
};

int main(int argc, char **argv)
{
	int rank, n;
	int ret;
	unsigned *val;
	starpu_data_handle_t data;
	int in_cache;
	int cache;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);

	cache = starpu_mpi_cache_is_enabled();
	if (cache == 0)
		goto skip;

	val = malloc(sizeof(*val));
	*val = 12;

	if (rank == 0)
		starpu_variable_data_register(&data, STARPU_MAIN_RAM, (uintptr_t)val, sizeof(unsigned));
	else
		starpu_variable_data_register(&data, -1, (uintptr_t)NULL, sizeof(unsigned));
	starpu_mpi_data_register(data, 42, 0);
	FPRINTF_MPI(stderr, "Registering data %p with tag %d and node %d\n", data, 42, 0);

	ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet_r, STARPU_R, data, STARPU_EXECUTE_ON_NODE, 1, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");

	in_cache = starpu_mpi_cached_receive(data);
	if (rank == 1)
	{
		STARPU_ASSERT_MSG(in_cache == 1, "Data should be in cache\n");
	}

	// We clean the cache
	starpu_mpi_cache_set(0);

	// We check the data is no longer in the cache
	in_cache = starpu_mpi_cached_receive(data);
	if (rank == 1)
	{
		STARPU_ASSERT_MSG(in_cache == 0, "Data should NOT be in cache\n");
	}

	ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet_r, STARPU_R, data, STARPU_EXECUTE_ON_NODE, 1, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
	in_cache = starpu_mpi_cached_receive(data);
	if (rank == 1)
	{
		STARPU_ASSERT_MSG(in_cache == 0, "Data should NOT be in cache\n");
	}

	FPRINTF(stderr, "Waiting ...\n");
	starpu_task_wait_for_all();

	starpu_data_unregister(data);
	free(val);

skip:
	starpu_mpi_shutdown();

	return cache == 0 ? STARPU_TEST_SKIPPED : 0;
}
