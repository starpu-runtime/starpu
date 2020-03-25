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
	(void)_args;

	int *data0 = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int *data1 = (int *)STARPU_VARIABLE_GET_PTR(descr[1]);
	int *data2 = (int *)STARPU_VARIABLE_GET_PTR(descr[2]);
	*data1 += *data0;
	*data2 += *data0;
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_W, STARPU_W},
	.model = &starpu_perfmodel_nop,
};

int main(int argc, char **argv)
{
	int ret;
	int i;
	int rank, size;
	int data[3];
	starpu_data_handle_t handles[3];
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);
	(void)mpi_init;

	ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if ((size < 3) || (starpu_cpu_worker_get_count() == 0))
	{
		if (rank == 0)
		{
			if (size < 3)
				FPRINTF(stderr, "We need at least 3 processes.\n");
			else
				FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		}
		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

	data[0] = 12;
	starpu_variable_data_register(&handles[0], STARPU_MAIN_RAM, (uintptr_t)&data[0], sizeof(int));
	starpu_mpi_data_register(handles[0], 10, 0);

	data[1] = 12;
	starpu_variable_data_register(&handles[1], STARPU_MAIN_RAM, (uintptr_t)&data[1], sizeof(int));
	starpu_mpi_data_register(handles[1], 20, 1);

	data[2] = 12;
	starpu_variable_data_register(&handles[2], STARPU_MAIN_RAM, (uintptr_t)&data[2], sizeof(int));
	starpu_mpi_data_register(handles[2], 30, 2);

	starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
			       STARPU_R, handles[2], STARPU_W, handles[0], STARPU_W, handles[1],
			       0);
	for(i=0 ; i<2 ; i++) starpu_data_acquire(handles[i], STARPU_R);
	FPRINTF_MPI(stderr, "data[%d,%d,%d] = %d,%d,%d\n", 0, 1, 2, data[0], data[1], data[2]);
	for(i=0 ; i<2 ; i++) starpu_data_release(handles[i]);
#ifndef STARPU_SIMGRID
	if (rank == 0)
	{
		STARPU_ASSERT_MSG(data[0] == 2*data[2] && data[1] == 2*data[2], "Computation incorrect. data[%d] (%d) != 2*data[%d] (%d) && data[%d] (%d) != 2*data[%d] (%d)\n",
				  0, data[0], 2, data[2], 1, data[1], 2, data[2]);
	}
#endif

	for(i=0 ; i<2 ; i++) starpu_data_acquire(handles[i], STARPU_W);
	for(i=0 ; i<2 ; i++) data[i] = 12;
	for(i=0 ; i<2 ; i++) starpu_data_release(handles[i]);

	// Let StarPU choose the node
	starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet,
			       STARPU_R, handles[2], STARPU_W, handles[0], STARPU_W, handles[1],
			       STARPU_EXECUTE_ON_NODE, 1,
			       0);
	for(i=0 ; i<2 ; i++) starpu_data_acquire(handles[i], STARPU_R);
	FPRINTF_MPI(stderr, "data[%d,%d,%d] = %d,%d,%d\n", 0, 1, 2, data[0], data[1], data[2]);
	for(i=0 ; i<2 ; i++) starpu_data_release(handles[i]);
#ifndef STARPU_SIMGRID
	if (rank == 1)
	{
		STARPU_ASSERT_MSG(data[0] == 2*data[2] && data[1] == 2*data[2], "Computation incorrect. data[%d] (%d) != 2*data[%d] (%d) && data[%d] (%d) != 2*data[%d] (%d)\n",
				  0, data[0], 2, data[2], 1, data[1], 2, data[2]);
	}
#endif

	for(i=0 ; i<3 ; i++) starpu_data_unregister(handles[i]);

	starpu_mpi_shutdown();
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
