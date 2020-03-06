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

void func_cpu(STARPU_ATTRIBUTE_UNUSED void *descr[], STARPU_ATTRIBUTE_UNUSED void *_args)
{
}

struct starpu_codelet mycodelet_r =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 1,
	.modes = {STARPU_R}
};

struct starpu_codelet mycodelet_w =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 1,
	.modes = {STARPU_W}
};

struct starpu_codelet mycodelet_rw =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 1,
	.modes = {STARPU_RW}
};

void test(struct starpu_codelet *codelet, enum starpu_data_access_mode mode, starpu_data_handle_t data, int rank, int in_cache)
{
	void *ptr;
	int ret;

	ret = starpu_mpi_insert_task(MPI_COMM_WORLD, codelet, mode, data, STARPU_EXECUTE_ON_NODE, 1, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_insert_task");

	ptr = _starpu_mpi_cache_received_data_get(data, 0);

	if (rank == 1)
	{
	     if (in_cache)
	     {
		     STARPU_ASSERT_MSG(ptr != NULL, "Data should be in cache\n");
	     }
	     else
	     {
		     STARPU_ASSERT_MSG(ptr == NULL, "Data should NOT be in cache\n");
	     }
	}
}

int main(int argc, char **argv)
{
	int rank, n;
	int ret;
	unsigned val;
	starpu_data_handle_t data;

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_init(&argc, &argv, 1);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (starpu_mpi_cache_is_enabled() == 0) goto skip;

	if (rank == 0)
		starpu_variable_data_register(&data, 0, (uintptr_t)&val, sizeof(unsigned));
	else
		starpu_variable_data_register(&data, -1, (uintptr_t)NULL, sizeof(unsigned));
	starpu_mpi_data_register(data, 42, 0);
	FPRINTF_MPI(stderr, "Registering data %p with tag %d and node %d\n", data, 42, 0);

	// We use the same data with different access modes and we check if it is
	// available or not in the cache
	test(&mycodelet_r, STARPU_R, data, rank, 1);
	test(&mycodelet_rw, STARPU_RW, data, rank, 0);
	test(&mycodelet_r, STARPU_R, data, rank, 1);
	test(&mycodelet_r, STARPU_R, data, rank, 1);
	test(&mycodelet_w, STARPU_W, data, rank, 0);

	FPRINTF(stderr, "Waiting ...\n");
	starpu_task_wait_for_all();

	starpu_data_unregister(data);

skip:
	starpu_mpi_shutdown();
	starpu_shutdown();

	return starpu_mpi_cache_is_enabled() == 0 ? STARPU_TEST_SKIPPED : 0;
}
