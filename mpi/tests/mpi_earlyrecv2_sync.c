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
#include <unistd.h>
#include <interface/complex_interface.h>

#define NB 6

typedef void (*check_func)(starpu_data_handle_t handle, int i, int rank, int *error);

int exchange(int rank, starpu_data_handle_t *handles, check_func func)
{
	int other_rank = rank%2 == 0 ? rank+1 : rank-1;
	int i;
	int ret=0;
	starpu_mpi_req req[NB];

	memset(req, 0, NB*sizeof(starpu_mpi_req));

	if (rank%2)
	{
		starpu_mpi_issend(handles[0], &req[0], other_rank, 0, MPI_COMM_WORLD);
		starpu_mpi_isend(handles[NB-1], &req[NB-1], other_rank, NB-1, MPI_COMM_WORLD);
		starpu_mpi_issend(handles[NB-2], &req[NB-2], other_rank, NB-2, MPI_COMM_WORLD);

		for(i=1 ; i<NB-2 ; i++)
		{
			if (i%2)
			{
				FPRINTF_MPI(stderr, "iSsending value %d\n", i);
				starpu_mpi_issend(handles[i], &req[i], other_rank, i, MPI_COMM_WORLD);
			}
			else
			{
				FPRINTF_MPI(stderr, "isending value %d\n", i);
				starpu_mpi_isend(handles[i], &req[i], other_rank, i, MPI_COMM_WORLD);
			}
		}
		for(i=0 ; i<NB ; i++)
		{
			starpu_mpi_wait(&req[i], MPI_STATUS_IGNORE);
		}
	}
	else
	{
		starpu_mpi_irecv(handles[0], &req[0], other_rank, 0, MPI_COMM_WORLD);
		STARPU_ASSERT(req[0] != NULL);
		starpu_mpi_irecv(handles[1], &req[1], other_rank, 1, MPI_COMM_WORLD);
		STARPU_ASSERT(req[1] != NULL);

		// We sleep to make sure that the data for the tag 8 and the tag 9 will be received before the recv are posted
		starpu_sleep(2);
		for(i=2 ; i<NB ; i++)
		{
			starpu_mpi_irecv(handles[i], &req[i], other_rank, i, MPI_COMM_WORLD);
			STARPU_ASSERT(req[i] != NULL);
		}

		for(i=0 ; i<NB ; i++)
		{
			starpu_mpi_wait(&req[i], MPI_STATUS_IGNORE);
			if (func)
				func(handles[i], i, rank, &ret);
		}
	}
	return ret;
}

void check_variable(starpu_data_handle_t handle, int i, int rank, int *error)
{
	int other_rank = rank%2 == 0 ? rank+1 : rank-1;

	int *rvalue = (int *)starpu_data_get_local_ptr(handle);
	if (*rvalue != i*other_rank)
	{
		FPRINTF_MPI(stderr, "Incorrect received value: %d != %d\n", *rvalue, i*other_rank);
		*error = 1;
	}
	else
	{
		FPRINTF_MPI(stderr, "Correct received value: %d == %d\n", *rvalue, i*other_rank);
	}
}

int exchange_variable(int rank)
{
	int ret, i;
	starpu_data_handle_t tab_handle[NB];
	int value[NB];

	ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	FPRINTF_MPI(stderr, "Exchanging variable data\n");

	for(i=0 ; i<NB ; i++)
	{
		value[i]=i*rank;
		starpu_variable_data_register(&tab_handle[i], STARPU_MAIN_RAM, (uintptr_t)&value[i], sizeof(int));
		starpu_mpi_data_register(tab_handle[i], i, rank);
	}
	ret = exchange(rank, tab_handle, check_variable);
	for(i=0 ; i<NB ; i++)
		starpu_data_unregister(tab_handle[i]);

	starpu_mpi_shutdown();

	return ret;
}

int exchange_void(int rank)
{
	int ret, i;
	starpu_data_handle_t tab_handle[NB];

	// This test is not run with valgrind as valgrind falsely detects error when exchanging NULL pointers
	STARPU_SKIP_IF_VALGRIND_RETURN_ZERO;

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_init(NULL, NULL, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");

	FPRINTF_MPI(stderr, "Exchanging void data\n");

	for(i=0 ; i<NB ; i++)
	{
		starpu_void_data_register(&tab_handle[i]);
		starpu_mpi_data_register(tab_handle[i], i, rank);
	}
	ret = exchange(rank, tab_handle, NULL);
	for(i=0 ; i<NB ; i++)
		starpu_data_unregister(tab_handle[i]);

	starpu_mpi_shutdown();

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
	else
	{
		FPRINTF_MPI(stderr, "Correct received value: %f == %d || %f == %d\n", *real, ((i*other_rank)+12), *imaginary, ((i*other_rank)+45));
	}
}

int exchange_complex(int rank)
{
	int ret, i;
	starpu_data_handle_t handle[NB];
	double real[NB];
	double imaginary[NB];

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_init(NULL, NULL, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");

	FPRINTF_MPI(stderr, "Exchanging complex data\n");

	for(i=0 ; i<NB ; i++)
	{
		real[i] = (i*rank)+12;
		imaginary[i] = (i*rank)+45;
		starpu_complex_data_register(&handle[i], STARPU_MAIN_RAM, &real[i], &imaginary[i], 1);
		starpu_mpi_data_register(handle[i], i, rank);
	}
	ret = exchange(rank, handle, check_complex);
	for(i=0 ; i<NB ; i++)
		starpu_data_unregister(handle[i]);

	starpu_mpi_shutdown();

	return ret;
}

int main(int argc, char **argv)
{
	int ret=0, global_ret=0;
	int rank, size;

	MPI_INIT_THREAD_real(&argc, &argv, MPI_THREAD_SERIALIZED);
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size%2 != 0)
	{
		FPRINTF(stderr, "We need a even number of processes.\n");
		MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

	ret = exchange_variable(rank);
	if (ret != 0)
		global_ret = ret;

	ret = exchange_void(rank);
	if (ret != 0)
		global_ret = ret;

	ret = exchange_complex(rank);
	if (ret != 0)
		global_ret = ret;

	MPI_Finalize();

	return global_ret;
}
