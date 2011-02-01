/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

#define NITER	2048

unsigned token = 42;
starpu_data_handle token_handle;

#ifdef STARPU_USE_CUDA
extern void increment_cuda(void *descr[], __attribute__ ((unused)) void *_args);
#endif

void increment_cpu(void *descr[], __attribute__ ((unused)) void *_args)
{
	unsigned *tokenptr = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);
	(*tokenptr)++;
}

static starpu_codelet increment_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
#ifdef STARPU_USE_CUDA
	.cuda_func = increment_cuda,
#endif
	.cpu_func = increment_cpu,
	.nbuffers = 1
};

void increment_token(void)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &increment_cl;
	task->buffers[0].handle = token_handle;
	task->buffers[0].mode = STARPU_RW;

	starpu_task_submit(task);
}

int main(int argc, char **argv)
{
	int rank, size;

#if 0
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

	starpu_init(NULL);
	starpu_mpi_initialize_extended(1, &rank, &size);

	if (size < 2)
	{
		if (rank == 0)
			fprintf(stderr, "We need at least 2 processes.\n");

		MPI_Finalize();
		return 0;
	}


	starpu_vector_data_register(&token_handle, 0, (uintptr_t)&token, 1, sizeof(unsigned));

	unsigned nloops = NITER;
	unsigned loop;

	unsigned last_loop = nloops - 1;
	unsigned last_rank = size - 1;

	for (loop = 0; loop < nloops; loop++)
	{
		int tag = loop*size + rank;

		if (!((loop == 0) && (rank == 0)))
		{
			starpu_mpi_irecv_detached(token_handle, (rank+size-1)%size, tag, MPI_COMM_WORLD, NULL, NULL);
		}
		else {
			token = 0;
			fprintf(stdout, "Start with token value %d\n", token);
		}

		increment_token();

		if (!((loop == last_loop) && (rank == last_rank)))
		{
			starpu_mpi_isend_detached(token_handle, (rank+1)%size, tag+1, MPI_COMM_WORLD, NULL, NULL);
		}
		else {
			starpu_data_acquire(token_handle, STARPU_R);
			fprintf(stdout, "Finished : token value %d\n", token);
			starpu_data_release(token_handle);
		}
	}

	starpu_task_wait_for_all();

	starpu_mpi_shutdown();
	starpu_shutdown();

        //MPI_Finalize();

	if (rank == last_rank)
	{
                fprintf(stderr, "[%d] token = %d == %d * %d ?\n", rank, token, nloops, size);
                STARPU_ASSERT(token == nloops*size);
	}

	return 0;
}
