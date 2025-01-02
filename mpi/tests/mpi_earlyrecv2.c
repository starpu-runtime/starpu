/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013-2013  Thibaut Lambert
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
	FPRINTF_MPI(stderr, "Requests %u received\n", *received);
	STARPU_PTHREAD_COND_SIGNAL(&cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
}

typedef void (*check_func)(starpu_data_handle_t handle, int i, int rank, int *error);

int exchange(int rank, starpu_data_handle_t *handles, starpu_mpi_tag_t initial_tag, check_func func, int detached)
{
	int other_rank = rank%2 == 0 ? rank+1 : rank-1;
	int i;
	int ret;

	if (rank%2)
	{
		ret = starpu_mpi_send(handles[0], other_rank, initial_tag+0, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
		ret = starpu_mpi_send(handles[NB-1], other_rank, initial_tag+NB-1, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
		for(i=1 ; i<NB-1 ; i++)
		{
			ret = starpu_mpi_send(handles[i], other_rank, initial_tag+i, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
		}
		return 0;
	}
	else
	{
		starpu_mpi_req req[NB];
		int received = 0;

		if (detached)
		{
			ret = starpu_mpi_irecv_detached(handles[0], other_rank, initial_tag+0, MPI_COMM_WORLD, callback, &received);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv_detached");
		}
		else
		{
			memset(req, 0, NB*sizeof(starpu_mpi_req));
			ret = starpu_mpi_irecv(handles[0], &req[0], other_rank, initial_tag+0, MPI_COMM_WORLD);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv");
			STARPU_ASSERT(req[0] != NULL);
		}

		// We sleep to make sure that the data for the tag 9 will be received before the recv is posted
		starpu_sleep(2);
		for(i=1 ; i<NB ; i++)
		{
			if (detached)
			{
				ret = starpu_mpi_irecv_detached(handles[i], other_rank, initial_tag+i, MPI_COMM_WORLD, callback, &received);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv_detached");
			}
			else
			{
				ret = starpu_mpi_irecv(handles[i], &req[i], other_rank, initial_tag+i, MPI_COMM_WORLD);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_irecv");
				STARPU_ASSERT(req[i] != NULL);
			}
		}

		if (detached)
		{
			STARPU_PTHREAD_MUTEX_LOCK(&mutex);
			while (received != NB)
			{
			     FPRINTF_MPI(stderr, "Received %d messages\n", received);
			     STARPU_PTHREAD_COND_WAIT(&cond, &mutex);
			}
			STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		}
		else
		{
			for(i=0 ; i<NB ; i++)
			{
				ret = starpu_mpi_wait(&req[i], MPI_STATUS_IGNORE);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_wait");
				func(handles[i], i, rank, &ret);
			}
		}
		return ret;
	}
}

void check_variable(starpu_data_handle_t handle, int i, int rank, int *error)
{
	int other_rank = rank%2 == 0 ? rank+1 : rank-1;

	starpu_data_acquire_on_node(handle, starpu_worker_get_local_memory_node(), STARPU_R);
	int *rvalue = (int *)starpu_data_get_local_ptr(handle);
	if (*rvalue != i*other_rank)
	{
		FPRINTF_MPI(stderr, "Incorrect received value: %d != %d\n", *rvalue, i*other_rank);
		*error = 1;
	}
	starpu_data_release(handle);
}

int exchange_variable(int rank, starpu_mpi_tag_t initial_tag, int detached)
{
	int ret, i;
	starpu_data_handle_t tab_handle[NB];
	int value[NB];

	FPRINTF_MPI(stderr, "Exchanging variable data with detached=%d\n", detached);

	for(i=0 ; i<NB ; i++)
	{
		value[i]=i*rank;
		starpu_variable_data_register(&tab_handle[i], STARPU_MAIN_RAM, (uintptr_t)&value[i], sizeof(int));
		starpu_mpi_data_register(tab_handle[i], i, rank);
	}
	ret = exchange(rank, tab_handle, initial_tag, check_variable, detached);
	for(i=0 ; i<NB ; i++)
		starpu_data_unregister(tab_handle[i]);

	return ret;
}

void check_void(starpu_data_handle_t handle, int i, int rank, int *error)
{
	(void)handle;
	(void)i;
	(void)rank;
	(void)error;
}

int exchange_void(int rank, starpu_mpi_tag_t initial_tag, int detached)
{
	int ret, i;
	starpu_data_handle_t tab_handle[NB];

	// This test is not run with valgrind as valgrind falsely detects error when exchanging NULL pointers
	STARPU_SKIP_IF_VALGRIND_RETURN_ZERO;

	FPRINTF_MPI(stderr, "Exchanging void data with detached=%d\n", detached);

	for(i=0 ; i<NB ; i++)
	{
		starpu_void_data_register(&tab_handle[i]);
		starpu_mpi_data_register(tab_handle[i], i, rank);
	}
	ret = exchange(rank, tab_handle, initial_tag, check_void, detached);
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
		FPRINTF_MPI(stderr, "Incorrect received value: %f != %d || %f != %d\n", *real, ((i*other_rank)+12), *imaginary, ((i*other_rank)+45));
		*error = 1;
	}
}

int exchange_complex(int rank, starpu_mpi_tag_t initial_tag, int detached)
{
	int ret, i;
	starpu_data_handle_t handle[NB];
	double real[NB];
	double imaginary[NB];

	FPRINTF_MPI(stderr, "Exchanging complex data with detached=%d\n", detached);

	for(i=0 ; i<NB ; i++)
	{
		real[i] = (i*rank)+12;
		imaginary[i] = (i*rank)+45;
		starpu_complex_data_register(&handle[i], STARPU_MAIN_RAM, &real[i], &imaginary[i], 1);
		starpu_mpi_data_register(handle[i], i, rank);
	}
	ret = exchange(rank, handle, initial_tag, check_complex, detached);
	for(i=0 ; i<NB ; i++)
		starpu_data_unregister(handle[i]);

	return ret;
}

int main(int argc, char **argv)
{
	int ret=0, global_ret=0;
	int rank, size;
	int mpi_init;
	struct starpu_conf conf;
	starpu_mpi_tag_t initial_tag = 0;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, &conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size%2 != 0)
	{
		FPRINTF(stderr, "We need a even number of processes.\n");
		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return rank == 0 ? STARPU_TEST_SKIPPED : 0;
	}

	ret = exchange_variable(rank, initial_tag, 0);
	initial_tag += NB;
	if (ret != 0)
		global_ret = ret;

	ret = exchange_variable(rank, initial_tag, 1);
	initial_tag += NB;
	if (ret != 0)
		global_ret = ret;

	ret = exchange_void(rank, initial_tag, 0);
	initial_tag += NB;
	if (ret != 0)
		global_ret = ret;

	ret = exchange_void(rank, initial_tag, 1);
	initial_tag += NB;
	if (ret != 0)
		global_ret = ret;

	ret = exchange_complex(rank, initial_tag, 0);
	initial_tag += NB;
	if (ret != 0)
		global_ret = ret;

	ret = exchange_complex(rank, initial_tag, 1);
	initial_tag += NB;
	if (ret != 0)
		global_ret = ret;

	starpu_mpi_shutdown();

	if (!mpi_init)
		MPI_Finalize();

	return rank == 0 ? global_ret : 0;
}
