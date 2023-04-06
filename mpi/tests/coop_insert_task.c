/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* Test to ensure coop are correctly detected even through the
 * starpu_mpi_task_insert() API.
 *
 * One task put an initial value in a buffer, then each node copies the content
 * of this buffer in a local buffer, within a task. Since each node needs the
 * initial buffer, this triggers a broadcast. */

#include <starpu_mpi.h>
#include "helper.h"

#define TARGET_VALUE 42


static void init_cpu_func(void *descr[], void *args)
{
	int *var = (int*) STARPU_VARIABLE_GET_PTR(descr[0]);
	int val;

	starpu_codelet_unpack_args(args, &val);
	*var = val;

	starpu_sleep(2); // Give time to submit other tasks and detect coop
}

static struct starpu_codelet init_cl =
{
	.cpu_funcs = { init_cpu_func },
	.cpu_funcs_name = { "init_task" },
	.nbuffers = 1,
	.modes = { STARPU_W }
};

static void copy_cpu_func(void* descr[], void* args)
{
	(void) args;

	int *var_src = (int*) STARPU_VARIABLE_GET_PTR(descr[0]);
	int *var_target = (int*) STARPU_VARIABLE_GET_PTR(descr[1]);

	*var_target = *var_src;
}

static struct starpu_codelet copy_cl =
{
	.cpu_funcs = { copy_cpu_func },
	.cpu_funcs_name = { "copy_task" },
	.nbuffers = 2,
	.modes = { STARPU_R, STARPU_W }
};

int main(int argc, char **argv)
{
	int ret, rank, size, mpi_init, i;
	int* data;
	starpu_data_handle_t* handles;
	MPI_Status status;
	struct starpu_conf conf;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, &conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	data = malloc(size*sizeof(int));
	handles = malloc(size*sizeof(starpu_data_handle_t));

	for (i = 0; i < size; i++)
	{
		if (i == rank)
		{
			starpu_variable_data_register(&handles[i], STARPU_MAIN_RAM, (uintptr_t)&data[i], sizeof(int));
		}
		else
		{
			starpu_variable_data_register(&handles[i], -1, (uintptr_t)NULL, sizeof(int));
		}

		STARPU_ASSERT(handles[i] != NULL);
		starpu_mpi_data_register(handles[i], i, i);
	}

	int val = TARGET_VALUE;
	starpu_mpi_task_insert(MPI_COMM_WORLD, &init_cl, STARPU_W, handles[0], STARPU_VALUE, &val, sizeof(val), 0);
	for (i = 1; i < size; i++)
	{
		starpu_mpi_task_insert(MPI_COMM_WORLD, &copy_cl, STARPU_R, handles[0], STARPU_W, handles[i], 0);
	}

	starpu_data_acquire(handles[rank], STARPU_R);
	int* handle_ptr = (int*) starpu_variable_get_local_ptr(handles[rank]);
	printf("[%d] data: %d\n", rank, *handle_ptr);
	STARPU_ASSERT(*handle_ptr == TARGET_VALUE);
	starpu_data_release(handles[rank]);

	for (i = 0; i < size; i++)
	{
		starpu_data_unregister(handles[i]);
	}

	free(handles);
	free(data);

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
