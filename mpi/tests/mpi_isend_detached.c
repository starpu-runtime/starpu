/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu_mpi.h>
#include <pthread.h>

#define NITER	2048
#define SIZE	16

static float *tab;
static starpu_data_handle tab_handle;

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

void callback(void *arg __attribute__((unused)))
{
	unsigned *sent = arg;
	
	pthread_mutex_lock(&mutex);
	*sent = 1;
	pthread_cond_signal(&cond);
	pthread_mutex_unlock(&mutex);
}

int main(int argc, char **argv)
{
	MPI_Init(NULL, NULL);

	int rank, size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size != 2)
	{
		if (rank == 0)
			fprintf(stderr, "We need exactly 2 processes.\n");

		MPI_Finalize();
		return 0;
	}

	starpu_init(NULL);
	starpu_mpi_initialize();

	tab = malloc(SIZE*sizeof(float));

	starpu_vector_data_register(&tab_handle, 0, (uintptr_t)tab, SIZE, sizeof(float));

	unsigned nloops = NITER;
	unsigned loop;

	int other_rank = (rank + 1)%2;

	for (loop = 0; loop < nloops; loop++)
	{
		if (rank == 0)
		{
			MPI_Status status;

			int sent = 0;
			starpu_mpi_isend_detached(tab_handle, other_rank, loop, MPI_COMM_WORLD, callback, &sent);

			pthread_mutex_lock(&mutex);
			while (!sent)
				pthread_cond_wait(&cond, &mutex);
			pthread_mutex_unlock(&mutex);
		}
		else {
			MPI_Status status;
			starpu_mpi_recv(tab_handle, other_rank, loop, MPI_COMM_WORLD, &status);
		}
	}
	
	starpu_mpi_shutdown();
	starpu_shutdown();

	MPI_Finalize();

	return 0;
}
