/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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
#include <unistd.h>
#include <interface/complex_interface.h>

#define NB 10

static starpu_pthread_mutex_t mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static starpu_pthread_cond_t cond = STARPU_PTHREAD_COND_INITIALIZER;

void callback(void *arg)
{
	unsigned *received = arg;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	*received = *received + 1;
	FPRINTF_MPI("Requests %d received\n", *received);
	STARPU_PTHREAD_COND_SIGNAL(&cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

typedef void (*check_func)(starpu_data_handle_t handle, int i, int rank, int *error);

int exchange(int rank, starpu_data_handle_t *handles, check_func func, int detached)
{
	int other_rank = rank%2 == 0 ? rank+1 : rank-1;
	int i;

	if (rank%2)
	{
		starpu_mpi_send(handles[0], other_rank, 0, MPI_COMM_WORLD);
		starpu_mpi_send(handles[NB-1], other_rank, NB-1, MPI_COMM_WORLD);
		for(i=1 ; i<NB-1 ; i++)
		{
			starpu_mpi_send(handles[i], other_rank, i, MPI_COMM_WORLD);
		}
		return 0;
	}
	else
	{
		int ret=0;
		starpu_mpi_req req[NB];
		int received = 0;

		if (detached)
		{
			starpu_mpi_irecv_detached(handles[0], other_rank, 0, MPI_COMM_WORLD, callback, &received);
		}
		else
		{
			memset(req, 0, NB*sizeof(starpu_mpi_req));
			starpu_mpi_irecv(handles[0], &req[0], other_rank, 0, MPI_COMM_WORLD);
			STARPU_ASSERT(req[0] != NULL);
		}

		// We sleep to make sure that the data for the tag 9 will be received before the recv is posted
		usleep(2000000);
		for(i=1 ; i<NB ; i++)
		{
			if (detached)
			{
				starpu_mpi_irecv_detached(handles[i], other_rank, i, MPI_COMM_WORLD, callback, &received);
			}
			else
			{
				starpu_mpi_irecv(handles[i], &req[i], other_rank, i, MPI_COMM_WORLD);
				STARPU_ASSERT(req[i] != NULL);
			}
		}

		if (detached)
		{
			STARPU_PTHREAD_MUTEX_LOCK(&mutex);
			while (received != NB)
			{
			     FPRINTF_MPI("Received %d messages\n", received);
			     STARPU_PTHREAD_COND_WAIT(&cond, &mutex);
			}
			STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		}
		else
		{
			for(i=0 ; i<NB ; i++)
			{
			     starpu_mpi_wait(&req[i], NULL);
			     func(handles[i], i, rank, &ret);
			}
		}
		return ret;
	}
}

void check_variable(starpu_data_handle_t handle, int i, int rank, int *error)
{
	int other_rank = rank%2 == 0 ? rank+1 : rank-1;

	int *rvalue = (int *)starpu_data_get_local_ptr(handle);
	if (*rvalue != i*other_rank)
	{
		FPRINTF_MPI("Incorrect received value: %d != %d\n", *rvalue, i*other_rank);
		*error = 1;
	}
}

int exchange_variable(int rank, int detached)
{
	int ret, i;
	starpu_data_handle_t tab_handle[NB];
	int value[NB];

	FPRINTF_MPI("Exchanging variable data with detached=%d\n", detached);

	for(i=0 ; i<NB ; i++)
	{
		value[i]=i*rank;
		starpu_variable_data_register(&tab_handle[i], STARPU_MAIN_RAM, (uintptr_t)&value[i], sizeof(int));
		starpu_data_set_tag(tab_handle[i], i);
	}
	ret = exchange(rank, tab_handle, check_variable, detached);
	for(i=0 ; i<NB ; i++)
		starpu_data_unregister(tab_handle[i]);

	return ret;
}

void check_complex(starpu_data_handle_t handle, int i, int rank, int *error)
{
	double *real = starpu_complex_get_real(handle);
	double *imaginary = starpu_complex_get_imaginary(handle);

	int other_rank = rank%2 == 0 ? rank+1 : rank-1;

	if ((*real != ((i*other_rank)+12)) || (*imaginary != ((i*other_rank)+45)))
	{
		FPRINTF_MPI("Incorrect received value: %f != %d || %f != %d\n", *real, ((i*other_rank)+12), *imaginary, ((i*other_rank)+45));
		*error = 1;
	}
}

int exchange_complex(int rank, int detached)
{
	int ret, i;
	starpu_data_handle_t handle[NB];
	double real[NB];
	double imaginary[NB];

	FPRINTF_MPI("Exchanging complex data with detached=%d\n", detached);

	for(i=0 ; i<NB ; i++)
	{
		real[i] = (i*rank)+12;
		imaginary[i] = (i*rank)+45;
		starpu_complex_data_register(&handle[i], STARPU_MAIN_RAM, &real[i], &imaginary[i], 1);
		starpu_data_set_tag(handle[i], i);
	}
	ret = exchange(rank, handle, check_complex, detached);
	for(i=0 ; i<NB ; i++)
		starpu_data_unregister(handle[i]);

	return ret;
}

int main(int argc, char **argv)
{
	int ret, rank, size;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size%2 != 0)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need a even number of processes.\n");

		MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_init(NULL, NULL, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");

	ret = exchange_variable(rank, 0);
	if (ret == 0)
		ret = exchange_variable(rank, 1);
	if (ret == 0)
		ret = exchange_complex(rank, 0);
	if (ret == 0)
		ret = exchange_complex(rank, 1);

	starpu_mpi_shutdown();
	starpu_shutdown();

	MPI_Finalize();

	return ret;
}
