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

void increment_token(starpu_data_handle_t token_handle)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &increment_cl;
	task->handles[0] = token_handle;
	task->synchronous = 1;

	int ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

int main(int argc, char **argv)
{
	int ret, rank, size;
	int mpi_init;
	int token = 42;
	starpu_data_handle_t token_handle;

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

	starpu_vector_data_register(&token_handle, STARPU_MAIN_RAM, (uintptr_t)&token, 1, sizeof(token));

	int nloops = NITER;
	int loop;

	int last_loop = nloops - 1;
	int last_rank = size - 1;

	for (loop = 0; loop < nloops; loop++)
	{
		starpu_mpi_tag_t tag = loop*size + rank;

		if (loop == 0 && rank == 0)
		{
			starpu_data_acquire(token_handle, STARPU_W);
			token = 0;
			FPRINTF(stdout, "Start with token value %d\n", token);
			starpu_data_release(token_handle);
		}
		else
		{
			MPI_Status status;
			starpu_mpi_recv(token_handle, (rank+size-1)%size, tag, MPI_COMM_WORLD, &status);
		}

		increment_token(token_handle);

		if (loop == last_loop && rank == last_rank)
		{
			starpu_data_acquire(token_handle, STARPU_R);
			FPRINTF(stdout, "Finished : token value %d\n", token);
			starpu_data_release(token_handle);
		}
		else
		{
			starpu_mpi_send(token_handle, (rank+1)%size, tag+1, MPI_COMM_WORLD);
		}
	}

	starpu_data_unregister(token_handle);
	starpu_mpi_shutdown();

	if (!mpi_init)
		MPI_Finalize();

#ifndef STARPU_SIMGRID
	if (rank == last_rank)
	{
		FPRINTF(stderr, "[%d] token = %d == %d * %d ?\n", rank, token, nloops, size);
		STARPU_ASSERT(token == nloops*size);
	}
#endif

	return 0;
}
