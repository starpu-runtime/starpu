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
	int node;
	int rank;
	(void)descr;

	starpu_codelet_unpack_args(_args, &node);
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	FPRINTF_MPI(stderr, "Expected node: %d - Actual node: %d\n", node, rank);

	assert(node == rank);
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_RW},
	.model = &starpu_perfmodel_nop,
	.name = "insert_task_node_choice"
};

int main(int argc, char **argv)
{
	int ret, rank, size, err, node;
	int x0=32;
	long long x1=23;
	starpu_data_handle_t data_handlesx0;
	starpu_data_handle_t data_handlesx1;

	ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (rank != 0 && rank != 1)
		goto end;

	if (rank == 0)
	{
		starpu_variable_data_register(&data_handlesx0, STARPU_MAIN_RAM, (uintptr_t)&x0, sizeof(x0));
		starpu_variable_data_register(&data_handlesx1, -1, (uintptr_t)NULL, sizeof(x1));
	}
	else
	{
		starpu_variable_data_register(&data_handlesx0, -1, (uintptr_t)NULL, sizeof(x0));
		starpu_variable_data_register(&data_handlesx1, STARPU_MAIN_RAM, (uintptr_t)&x1, sizeof(x1));
	}
	starpu_mpi_data_register(data_handlesx0, 100, 0);
	starpu_mpi_data_register(data_handlesx1, 200, 1);

	node = 0;
	err = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
				     STARPU_VALUE, &node, sizeof(node),
				     STARPU_EXECUTE_ON_NODE, 0,
				     STARPU_RW, data_handlesx0, STARPU_RW, data_handlesx1,
				     0);
	assert(err == 0);

	node = starpu_data_get_rank(data_handlesx1);
	err = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
				     STARPU_VALUE, &node, sizeof(node),
				     STARPU_EXECUTE_ON_DATA, data_handlesx1,
				     STARPU_RW, data_handlesx0, STARPU_RW, data_handlesx1,
				     0);
	assert(err == 0);

	// Node 1 has a long long data which has a bigger size than a
	// int, so it is going to be selected by the node selection
	// policy to execute the codelet
	err = starpu_mpi_node_selection_set_current_policy(STARPU_MPI_NODE_SELECTION_MOST_R_DATA);
	assert(err == 0);
	node = 1;
	err = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
				     STARPU_VALUE, &node, sizeof(node),
				     STARPU_RW, data_handlesx0, STARPU_RW, data_handlesx1,
				     0);
	assert(err == 0);

	FPRINTF_MPI(stderr, "Waiting ...\n");
	starpu_task_wait_for_all();
	starpu_data_unregister(data_handlesx0);
	starpu_data_unregister(data_handlesx1);

end:
	starpu_mpi_shutdown();

	return 0;
}

