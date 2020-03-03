/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <common/thread.h>
#include "helper.h"

#ifdef STARPU_QUICK_CHECK
#  define NITER	16
#elif !defined(STARPU_LONG_CHECK)
#  define NITER	256
#else
#  define NITER	2048
#endif
#define SIZE	16

float *tab;
starpu_data_handle_t tab_handle;

static starpu_pthread_mutex_t mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static starpu_pthread_cond_t cond = STARPU_PTHREAD_COND_INITIALIZER;

void callback(void *arg)
{
	unsigned *received = arg;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	*received = 1;
	STARPU_PTHREAD_COND_SIGNAL(&cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}


int main(int argc, char **argv)
{
	int ret, rank, size;
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size%2 != 0)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need a even number of processes.\n");

		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

	tab = calloc(SIZE, sizeof(float));

	starpu_vector_data_register(&tab_handle, STARPU_MAIN_RAM, (uintptr_t)tab, SIZE, sizeof(float));

	int nloops = NITER;
	int loop;
	int other_rank = rank%2 == 0 ? rank+1 : rank-1;

	for (loop = 0; loop < nloops; loop++)
	{
		if ((loop % 2) == (rank%2))
		{
			starpu_mpi_send(tab_handle, other_rank, loop, MPI_COMM_WORLD);
		}
		else
		{
			int received = 0;
			starpu_mpi_irecv_detached(tab_handle, other_rank, loop, MPI_COMM_WORLD, callback, &received);

			STARPU_PTHREAD_MUTEX_LOCK(&mutex);
			while (!received)
				STARPU_PTHREAD_COND_WAIT(&cond, &mutex);
			STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		}
	}

	starpu_data_unregister(tab_handle);
	free(tab);

	starpu_mpi_shutdown();

	if (!mpi_init)
		MPI_Finalize();

	return 0;
}
