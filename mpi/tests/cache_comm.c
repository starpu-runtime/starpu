/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 20252025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

// This test checks the MPI cache mechanism works when dealing with several MPI communicators
// See https://github.com/starpu-runtime/starpu/pull/72

#include <starpu.h>
#include <starpu_mpi.h>
#include "helper.h"

#define GRID_SIZE 3

void task(void *buffers[], void *cl_arg)
{
	int *a = (int*)STARPU_VARIABLE_GET_PTR(buffers[0]);
	int *b = (int*)STARPU_VARIABLE_GET_PTR(buffers[1]);
	*a = *b;
}

struct starpu_codelet codelet =
{
	.cpu_funcs = {task},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_R}
};

int main(int argc, char** argv)
{
	int rank, size;
	int i, ret;
	MPI_Comm comm_inverse;
	int local_var;
	starpu_data_handle_t handles[GRID_SIZE];
	struct starpu_conf conf;

	MPI_INIT_THREAD_real(&argc, &argv, MPI_THREAD_SERIALIZED);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_sc = -1;
	conf.ntcpip_sc = -1;

	ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, &conf);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	MPI_Comm_split(MPI_COMM_WORLD, 0, size - 1 - rank, &comm_inverse);
	starpu_mpi_comm_register(comm_inverse);

	local_var = rank;
	for(i=0; i<GRID_SIZE; i++)
	{
		if(i == rank)
		{
			starpu_variable_data_register(&handles[i], STARPU_MAIN_RAM, (uintptr_t)&local_var, sizeof(int));
		}
		else
		{
			starpu_variable_data_register(&handles[i], -1, (uintptr_t)NULL, sizeof(int));
		}
		starpu_mpi_data_register(handles[i], i, i);
	}

	for(i=0; i<GRID_SIZE; i++)
	{
		starpu_mpi_task_insert(MPI_COMM_WORLD, &codelet,
				       STARPU_RW, handles[i],
				       STARPU_R, handles[(i+1) % GRID_SIZE],
				       0);
	}

	starpu_mpi_wait_for_all(MPI_COMM_WORLD);
	starpu_mpi_barrier(MPI_COMM_WORLD);

	for(i=0; i<GRID_SIZE; i++)
	{
		starpu_mpi_data_set_rank_comm(handles[i], size - 1 - i, comm_inverse);
	}

	for(i=0; i<GRID_SIZE; i++)
	{
		starpu_mpi_task_insert(comm_inverse, &codelet,
				       STARPU_RW, handles[(i+1) % GRID_SIZE],
				       STARPU_R, handles[i],
				       0);
	}

	starpu_mpi_wait_for_all(comm_inverse);
	starpu_mpi_barrier(MPI_COMM_WORLD);

	for(i=0; i<GRID_SIZE; i++)
	{
		starpu_mpi_data_set_rank_comm(handles[i], i, MPI_COMM_WORLD);
	}

	for(i=0; i<GRID_SIZE; i++)
	{
		starpu_mpi_task_insert(MPI_COMM_WORLD, &codelet,
				       STARPU_RW, handles[i],
				       STARPU_R, handles[(i+1) % GRID_SIZE],
				       0);
	}

	starpu_mpi_shutdown();

enodev:
	MPI_Finalize();
	return 0;
}
