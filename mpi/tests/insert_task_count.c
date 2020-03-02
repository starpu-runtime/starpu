/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifdef STARPU_QUICK_CHECK
#  define NITER	32
#elif !defined(STARPU_LONG_CHECK)
#  define NITER	256
#else
#  define NITER	2048
#endif

#ifdef STARPU_USE_CUDA
extern void increment_cuda(void *descr[], void *_args);
#endif

void increment_cpu(void *descr[], void *_args)
{
	(void)_args;
	int *tokenptr = (int *)STARPU_VECTOR_GET_PTR(descr[0]);
	(*tokenptr)++;
}

static struct starpu_codelet increment_cl =
{
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {increment_cuda},
#endif
	.cpu_funcs = {increment_cpu},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &starpu_perfmodel_nop,
};

int main(int argc, char **argv)
{
	int ret, rank, size;
	int token = 0;
	starpu_data_handle_t token_handle;
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size < 2 || (starpu_cpu_worker_get_count() + starpu_cuda_worker_get_count() == 0))
	{
		if (rank == 0)
		{
			if (size < 2)
				FPRINTF(stderr, "We need at least 2 processes.\n");
			else
				FPRINTF(stderr, "We need at least 1 CPU or CUDA worker.\n");
		}
		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

	if (rank == 1)
		starpu_vector_data_register(&token_handle, 0, (uintptr_t)&token, 1, sizeof(token));
	else
		starpu_vector_data_register(&token_handle, -1, (uintptr_t)NULL, 1, sizeof(token));
	starpu_mpi_data_register(token_handle, 12, 1);

	int nloops = NITER;
	int loop;

	FPRINTF_MPI(stderr, "Start with token value %d\n", token);

	for (loop = 0; loop < nloops; loop++)
	{
		if (loop % 2)
			starpu_mpi_task_insert(MPI_COMM_WORLD, &increment_cl,
					       STARPU_RW|STARPU_SSEND, token_handle,
					       STARPU_EXECUTE_ON_NODE, 0,
					       0);
		else
			starpu_mpi_task_insert(MPI_COMM_WORLD, &increment_cl,
					       STARPU_RW, token_handle,
					       STARPU_EXECUTE_ON_NODE, 0,
					       0);
	}

	starpu_task_wait_for_all();
	starpu_data_unregister(token_handle);
	starpu_mpi_shutdown();

	FPRINTF_MPI(stderr, "Final value for token %d\n", token);

	if (!mpi_init)
		MPI_Finalize();

#ifndef STARPU_SIMGRID
	if (rank == 1)
	{
		STARPU_ASSERT_MSG(token == nloops, "token==%d != expected_value==%d\n", token, nloops);
	}
	else
	{
		STARPU_ASSERT_MSG(token == 0, "token==%d != expected_value==0\n", token);

	}
#endif

	return 0;
}
