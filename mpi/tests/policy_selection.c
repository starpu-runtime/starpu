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

struct starpu_codelet mycodelet_2 =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 2,
	.modes = {STARPU_W, STARPU_W},
	.model = &starpu_perfmodel_nop,
};
struct starpu_codelet mycodelet_3 =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_W, STARPU_W},
	.model = &starpu_perfmodel_nop,
};

int main(int argc, char **argv)
{
	int ret;
	int rank, size;
	int policy = 12;
	struct starpu_task *task;
	starpu_data_handle_t handles[3];
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);
	(void)mpi_init;

	ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size < 3)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 3 processes.\n");

		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

	if (rank == 0)
	{
		starpu_variable_data_register(&handles[0], STARPU_MAIN_RAM, (uintptr_t)&policy, sizeof(int));
	}
	else
	{
		starpu_variable_data_register(&handles[0], -1, (uintptr_t)NULL, sizeof(int));
	}
	starpu_mpi_data_register(handles[0], 10, 0);

	if (rank == 1)
	{
		starpu_variable_data_register(&handles[1], STARPU_MAIN_RAM, (uintptr_t)&policy, sizeof(int));
	}
	else
	{
		starpu_variable_data_register(&handles[1], -1, (uintptr_t)NULL, sizeof(int));
	}
	starpu_mpi_data_register(handles[1], 20, 1);

	if (rank == 2)
	{
	     starpu_variable_data_register(&handles[2], STARPU_MAIN_RAM, (uintptr_t)&policy, sizeof(int));
	}
	else
	{
	     starpu_variable_data_register(&handles[2], -1, (uintptr_t)NULL, sizeof(int));
	}
	starpu_mpi_data_register(handles[2], 30, 2);

	// Force the execution on node 1
	task = starpu_mpi_task_build(MPI_COMM_WORLD, &mycodelet_3,
				     STARPU_R, handles[2],
				     STARPU_W, handles[0], STARPU_W, handles[1],
				     STARPU_EXECUTE_ON_NODE, 1,
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

	// Force the execution on node 1
	task = starpu_mpi_task_build(MPI_COMM_WORLD, &mycodelet_2,
				     STARPU_W, handles[0], STARPU_W, handles[1],
				     STARPU_EXECUTE_ON_NODE, 1,
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

	// Let StarPU choose the node
	task = starpu_mpi_task_build(MPI_COMM_WORLD, &mycodelet_3,
				     STARPU_R, handles[2],
				     STARPU_W, handles[0], STARPU_W, handles[1],
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
		STARPU_ASSERT_MSG(task == NULL, "Task should be executed by rank 2\n");
	}

	// Let StarPU choose the node
	task = starpu_mpi_task_build(MPI_COMM_WORLD, &mycodelet_2,
				     STARPU_W, handles[0], STARPU_W, handles[1],
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
	starpu_data_unregister(handles[2]);

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
