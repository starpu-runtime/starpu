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

/* This tests that one can register temporary data0 on each MPI node which can mix with common data0 */

#include <starpu_mpi.h>
#include "helper.h"

static void func_add(void *descr[], void *_args)
{
	(void)_args;
	int *a = (void*) STARPU_VARIABLE_GET_PTR(descr[0]);
	const int *b = (void*) STARPU_VARIABLE_GET_PTR(descr[1]);
	const int *c = (void*) STARPU_VARIABLE_GET_PTR(descr[2]);

	*a = *b + *c;
	FPRINTF_MPI(stderr, "%d + %d = %d\n", *b, *c, *a);
}

static struct starpu_codelet codelet_add =
{
	.cpu_funcs = {func_add},
	.nbuffers = 3,
	.modes = {STARPU_W, STARPU_R, STARPU_R},
	.model = &starpu_perfmodel_nop,
	.flags = STARPU_CODELET_SIMGRID_EXECUTE,
};

int main(int argc, char **argv)
{
	int rank, size, n;
	int ret;
	int a;
	int val0 = 0, val1 = 0;
	starpu_data_handle_t data0, data1, tmp0, tmp, tmp2;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size < 2)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 2 processes.\n");

		starpu_mpi_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	if (starpu_mpi_cache_is_enabled() == 0)
		goto skip;

	if (rank == 0)
	{
		val0 = 1;
		starpu_variable_data_register(&data0, STARPU_MAIN_RAM, (uintptr_t)&val0, sizeof(val0));
		starpu_variable_data_register(&data1, -1, (uintptr_t)NULL, sizeof(val0));
		starpu_variable_data_register(&tmp0, -1, (uintptr_t)NULL, sizeof(val0));
		starpu_mpi_data_register(tmp0, -1, 0);
	}
	else if (rank == 1)
	{
		starpu_variable_data_register(&data0, -1, (uintptr_t)NULL, sizeof(val0));
		starpu_variable_data_register(&data1, STARPU_MAIN_RAM, (uintptr_t)&val1, sizeof(val1));
		tmp0 = NULL;
	}
	else
	{
		starpu_variable_data_register(&data0, -1, (uintptr_t)NULL, sizeof(val0));
		starpu_variable_data_register(&data1, -1, (uintptr_t)NULL, sizeof(val0));
		tmp0 = NULL;
	}
	starpu_variable_data_register(&tmp, -1, (uintptr_t)NULL, sizeof(val0));
	starpu_variable_data_register(&tmp2, -1, (uintptr_t)NULL, sizeof(val0));

	starpu_mpi_data_register(data0, 42, 0);
	starpu_mpi_data_register(data1, 43, 1);
	starpu_mpi_data_register(tmp, 44, 0);
	starpu_mpi_data_register(tmp2, -1, STARPU_MPI_PER_NODE);

	/* Test temporary data0 on node 0 only */
	starpu_mpi_task_insert(MPI_COMM_WORLD, &codelet_add, STARPU_W, tmp0, STARPU_R, data0, STARPU_R, data0, 0);

	starpu_mpi_task_insert(MPI_COMM_WORLD, &codelet_add, STARPU_W, data0, STARPU_R, tmp0, STARPU_R, tmp0, 0);

	starpu_mpi_task_insert(MPI_COMM_WORLD, &codelet_add, STARPU_W, tmp, STARPU_R, data0, STARPU_R, data0, 0);

	/* Now make some tmp per-node, so that each node replicates the computation */
	for (n = 0; n < size; n++)
		if (n != 0)
			/* Get the value on all nodes */
			starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD, tmp, n, NULL, NULL);
	starpu_mpi_data_set_rank(tmp, STARPU_MPI_PER_NODE);

	/* This task writes to a per-node data, so will be executed by all nodes */
	starpu_mpi_task_insert(MPI_COMM_WORLD, &codelet_add, STARPU_W, tmp2, STARPU_R, tmp, STARPU_R, tmp, 0);

	/* All MPI nodes have computed the value (no MPI communication here!) */
	starpu_data_acquire_on_node(tmp2, STARPU_MAIN_RAM, STARPU_R);
	STARPU_ASSERT(*(int*)starpu_data_handle_to_pointer(tmp2, STARPU_MAIN_RAM) == 16);
	starpu_data_release_on_node(tmp2, STARPU_MAIN_RAM);

	/* And nodes 0 and 1 do something with it */
	starpu_mpi_task_insert(MPI_COMM_WORLD, &codelet_add, STARPU_W, data0, STARPU_R, tmp, STARPU_R, tmp2, 0);
	starpu_mpi_task_insert(MPI_COMM_WORLD, &codelet_add, STARPU_W, data1, STARPU_R, tmp, STARPU_R, tmp2, 0);

	starpu_task_wait_for_all();

	if (rank == 0)
	{
		starpu_data_unregister(tmp0);
	}
	starpu_data_unregister(data0);
	starpu_data_unregister(data1);
	starpu_data_unregister(tmp);
	starpu_data_unregister(tmp2);

	if (rank == 0)
		STARPU_ASSERT_MSG(val0 == 24, "[rank 0] %d should be %d\n", val0, 24);
	if (rank == 1)
		STARPU_ASSERT_MSG(val1 == 24, "[rank 1] %d should be %d\n", val1, 24);

 skip:
	starpu_mpi_shutdown();

	return 0;
}
