/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
	int *x = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int *y = (int *)STARPU_VARIABLE_GET_PTR(descr[1]);

	FPRINTF(stdout, "VALUES: %d %d\n", *x, *y);
}

void func_on_node_cpu(void *descr[], void *_args)
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
};

struct starpu_codelet mycodelet_on_node =
{
	.cpu_funcs = {func_on_node_cpu},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_RW},
	.model = &starpu_perfmodel_nop,
};

int main(int argc, char **argv)
{
	int ret, i, x[2];
	starpu_data_handle_t data_handles[2];
	int barrier_ret;
	int rank;
	struct starpu_task *task;
	struct starpu_mpi_task_exchange_params params;
	struct starpu_data_descr descrs[2];
	struct starpu_conf conf;

	MPI_INIT_THREAD_real(&argc, &argv, MPI_THREAD_SERIALIZED);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_sc = -1;
	conf.ntcpip_sc = -1;

	ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, &conf);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);

	if (starpu_cpu_worker_get_count() == 0)
	{
		// If there is no cpu to execute the codelet, mpi will block trying to do the post-execution communication
		ret = -ENODEV;
		FPRINTF_MPI(stderr, "No CPU is available\n");
		goto nodata;
	}

	for(i=0 ; i<2 ; i++)
	{
		x[i] = rank*2 + (i+1);
		starpu_variable_data_register(&data_handles[i], STARPU_MAIN_RAM, (uintptr_t)&x[i], sizeof(int));
		starpu_mpi_data_register(data_handles[i], i, i);
	}

	task = starpu_task_create();
	task->cl = &mycodelet_on_node;
	task->handles[0] = data_handles[0];
	task->handles[1] = data_handles[1];

	starpu_mpi_task_exchange_data_before_execution(MPI_COMM_WORLD, task, descrs, &params);
	if (params.do_execute)
	{
		ret = starpu_task_submit(task);
		if (ret == -ENODEV)
		{
			task->destroy = 0;
			starpu_task_destroy(task);
			goto xenodev;
		}
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	else
	{
		task->destroy = 0;
		starpu_task_destroy(task);
	}
	starpu_mpi_task_exchange_data_after_execution(MPI_COMM_WORLD, descrs, 2, params);

	struct starpu_mpi_task_exchange_params params_on_node;
	struct starpu_data_descr descrs_on_node[2];
	struct starpu_task *task_on_node;
	
	task_on_node = starpu_task_create();
	task_on_node->cl = &mycodelet_on_node;
	task_on_node->handles[0] = data_handles[0];
	task_on_node->handles[1] = data_handles[1];

	int node = 0;
	struct starpu_codelet_pack_arg_data state;
	starpu_codelet_pack_arg_init(&state);
	starpu_codelet_pack_arg(&state, &node, sizeof(node));
	starpu_codelet_pack_arg_fini(&state, &task_on_node->cl_arg, &task_on_node->cl_arg_size);

	starpu_mpi_task_exchange_data_before_execution_on_node(MPI_COMM_WORLD, task_on_node, descrs_on_node, &params_on_node, node);
	if (params_on_node.do_execute)
	{
		ret = starpu_task_submit(task_on_node);
		if (ret == -ENODEV)
		{
			task_on_node->destroy = 0;
			starpu_task_destroy(task_on_node);
			goto xenodev;
		}
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	else
	{
		task_on_node->destroy = 0;
		starpu_task_destroy(task_on_node);
	}
	starpu_mpi_task_exchange_data_after_execution(MPI_COMM_WORLD, descrs_on_node, 2, params_on_node);

	starpu_task_wait_for_all();

xenodev:
	for(i=0; i<2; i++)
	{
		starpu_data_unregister(data_handles[i]);
	}

nodata:
	barrier_ret = MPI_Barrier(MPI_COMM_WORLD);
	STARPU_ASSERT(barrier_ret == MPI_SUCCESS);
	starpu_mpi_shutdown();

enodev:
	MPI_Finalize();
	return 0;
}
