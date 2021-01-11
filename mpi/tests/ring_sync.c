/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

int token = 42;
starpu_data_handle_t token_handle;

#ifdef STARPU_USE_CUDA
extern void increment_cuda(void *descr[], STARPU_ATTRIBUTE_UNUSED void *_args);
#endif

void increment_cpu(void *descr[], STARPU_ATTRIBUTE_UNUSED void *_args)
{
	int *tokenptr = (int *)STARPU_VECTOR_GET_PTR(descr[0]);
	(*tokenptr)++;
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

static struct starpu_codelet increment_cl =
{
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {increment_cuda},
#endif
	.cpu_funcs = {increment_cpu},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &dumb_model
};

void increment_token(void)
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

	starpu_vector_data_register(&token_handle, 0, (uintptr_t)&token, 1, sizeof(token));

	int nloops = NITER;
	int loop;

	int last_loop = nloops - 1;
	int last_rank = size - 1;

	for (loop = 0; loop < nloops; loop++)
	{
		int tag = loop*size + rank;

		if (loop == 0 && rank == 0)
		{
			token = 0;
			FPRINTF(stdout, "Start with token value %d\n", token);
		}
		else
		{
			MPI_Status status;
			starpu_mpi_recv(token_handle, (rank+size-1)%size, tag, MPI_COMM_WORLD, &status);
		}

		increment_token();

		if (loop == last_loop && rank == last_rank)
		{
			starpu_data_acquire(token_handle, STARPU_R);
			FPRINTF(stdout, "Finished : token value %d\n", token);
			starpu_data_release(token_handle);
		}
		else
		{
			starpu_mpi_req req;
			MPI_Status status;
			starpu_mpi_issend(token_handle, &req, (rank+1)%size, tag+1, MPI_COMM_WORLD);
			starpu_mpi_wait(&req, &status);
		}
	}

	starpu_data_unregister(token_handle);
	starpu_mpi_shutdown();
	starpu_shutdown();

	MPI_Finalize();

#ifndef STARPU_SIMGRID
	if (rank == last_rank)
	{
		STARPU_ASSERT(token == nloops*size);
	}
#endif

	return 0;
}
