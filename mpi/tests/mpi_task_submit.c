/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
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

	MPI_INIT_THREAD_real(&argc, &argv, MPI_THREAD_SERIALIZED);
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);

	ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, NULL);
	if (ret == -ENODEV) return rank==0?STARPU_TEST_SKIPPED:0;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

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
	task->cl = &mycodelet;
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
			goto enodev;
		}
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	else
	{
		task->destroy = 0;
		starpu_task_destroy(task);
	}

	starpu_mpi_task_exchange_data_after_execution(MPI_COMM_WORLD, descrs, 2, params);

	starpu_task_wait_for_all();

enodev:
	for(i=0; i<2; i++)
	{
		starpu_data_unregister(data_handles[i]);
	}

nodata:
	barrier_ret = MPI_Barrier(MPI_COMM_WORLD);
	STARPU_ASSERT(barrier_ret == MPI_SUCCESS);
	starpu_mpi_shutdown();

	MPI_Finalize();
	if (rank == 0)
		return ret==-ENODEV?STARPU_TEST_SKIPPED:ret;
	else
		return 0;
}
