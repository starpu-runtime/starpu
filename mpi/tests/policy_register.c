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
#include "helper.h"

void func_cpu(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 2,
	.modes = {STARPU_W, STARPU_W},
	.model = &starpu_perfmodel_nop,
};

int starpu_mpi_select_node_my_policy_0(int me, int nb_nodes, struct starpu_data_descr *descr, int nb_data)
{
	(void) me;
	(void) nb_nodes;
	(void) nb_data;

	starpu_data_handle_t data = descr[0].handle;
	return starpu_data_get_rank(data);
}

int starpu_mpi_select_node_my_policy_1(int me, int nb_nodes, struct starpu_data_descr *descr, int nb_data)
{
	(void) me;
	(void) nb_nodes;
	(void) nb_data;

	starpu_data_handle_t data = descr[1].handle;
	return starpu_data_get_rank(data);
}

int main(int argc, char **argv)
{
	int ret;
	int rank, size;
	int policy;
	struct starpu_task *task;
	starpu_data_handle_t handles[2];
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size < 2)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 2 processes.\n");

		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

	if (rank == 0)
		starpu_variable_data_register(&handles[0], STARPU_MAIN_RAM, (uintptr_t)&policy, sizeof(int));
	else
		starpu_variable_data_register(&handles[0], -1, (uintptr_t)NULL, sizeof(int));
	starpu_mpi_data_register(handles[0], 10, 0);
	if (rank == 1)
		starpu_variable_data_register(&handles[1], STARPU_MAIN_RAM, (uintptr_t)&policy, sizeof(int));
	else
		starpu_variable_data_register(&handles[1], -1, (uintptr_t)NULL, sizeof(int));
	starpu_mpi_data_register(handles[1], 20, 1);

	policy = starpu_mpi_node_selection_register_policy(starpu_mpi_select_node_my_policy_1);
	starpu_mpi_node_selection_set_current_policy(policy);

	task = starpu_mpi_task_build(MPI_COMM_WORLD, &mycodelet,
				     STARPU_W, handles[0], STARPU_W, handles[1],
				     0);
	FPRINTF_MPI(stderr, "Task %p\n", task);
	if (rank == 1)
	{
		STARPU_ASSERT_MSG(task, "Task should be executed by rank 1\n");
		task->destroy = 0;
		starpu_task_destroy(task);
	}
	else
	{
		STARPU_ASSERT_MSG(task == NULL, "Task should be executed by rank 1\n");
	}

	policy = starpu_mpi_node_selection_register_policy(starpu_mpi_select_node_my_policy_0);
	task = starpu_mpi_task_build(MPI_COMM_WORLD, &mycodelet,
				     STARPU_W, handles[0], STARPU_W, handles[1],
				     STARPU_NODE_SELECTION_POLICY, policy,
				     0);
	FPRINTF_MPI(stderr, "Task %p\n", task);
	if (rank == 0)
	{
		STARPU_ASSERT_MSG(task, "Task should be executed by rank 0\n");
		task->destroy = 0;
		starpu_task_destroy(task);
	}
	else
	{
		STARPU_ASSERT_MSG(task == NULL, "Task should be executed by rank 0\n");
	}

	starpu_data_unregister(handles[0]);
	starpu_data_unregister(handles[1]);
	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
