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

/* Dummy cost function for simgrid */
static double cost_function(struct starpu_task *task STARPU_ATTRIBUTE_UNUSED, unsigned nimpl STARPU_ATTRIBUTE_UNUSED)
{
	return 0.000001;
}
static struct starpu_perfmodel dumb_model =
{
	.type		= STARPU_COMMON,
	.cost_function	= cost_function
};

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 2,
	.modes = {STARPU_W, STARPU_W},
	.model = &dumb_model
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

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED);
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_init(NULL, NULL, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");

	if (size < 2)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 2 processes.\n");

		starpu_mpi_shutdown();
		starpu_shutdown();
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
	starpu_shutdown();
	MPI_Finalize();

	return 0;
}
