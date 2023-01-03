/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2022-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void cpu_fun(void* buffers[], void* args)
{
	float *ptr = (float*)(STARPU_VECTOR_GET_PTR(buffers[0]));
	ptr[0] = 42;
}

int can_execute(unsigned workerid, struct starpu_task* task, unsigned nimpl)
{
	return 1;
}

static struct starpu_codelet codelet =
{
	.can_execute = can_execute,
	.cpu_funcs = {cpu_fun},
	.nbuffers = 1,
	.modes = {STARPU_W},
	.model = &starpu_perfmodel_nop,
	.flags = STARPU_CODELET_SIMGRID_EXECUTE,
};

int main(int argc, char** argv)
{
	if (starpu_init(NULL))
	{
		fprintf(stderr, "Error initializing StarPU\n");
		return 1;
	}

	if (starpu_mpi_init(&argc, &argv, 1))
	{
		fprintf(stderr, "Error initializing StarPU MPI\n");
		return 1;
	}

	int rank;
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);

	// register a vector of one element
	float *data = malloc(sizeof(float));
	data[0] = 55;
	starpu_data_handle_t handle;
	starpu_vector_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t) data, 1, sizeof(data[0]));
	starpu_mpi_data_register(handle, 0, 0);

	// run the task
	starpu_mpi_task_insert(MPI_COMM_WORLD, &codelet, STARPU_W, handle, NULL);

	// gather the result
	starpu_data_unregister(handle);

	// shutdown starpu
	starpu_mpi_shutdown();
	starpu_shutdown();

	// check results
	int ret = 0;
	if (rank == 0)
	{
		if (data[0] == 42)
		{
			ret = 0;
			fprintf(stderr, "Success!\n");
		}
		else
		{
			ret = 1;
			fprintf(stderr, "Failure!\n");
		}
	}
	free(data);

	return ret;
}
