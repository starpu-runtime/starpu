/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2012, 2013  Centre National de la Recherche Scientifique
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

#include <starpu.h>
#include <starpu_mpi.h>
#include <math.h>
#include "helper.h"

#if !defined(STARPU_HAVE_UNSETENV)
#warning unsetenv is not defined. Skipping test
int main(int argc, char **argv)
{
	return STARPU_TEST_SKIPPED;
}
#else

void func_cpu(STARPU_ATTRIBUTE_UNUSED void *descr[], STARPU_ATTRIBUTE_UNUSED void *_args)
{
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu, NULL},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_R}
};

#define N     1000

/* Returns the MPI node number where data indexes index is */
int my_distrib(int x)
{
	return x;
}

void test_cache(int rank, int size, int enabled, size_t *comm_amount)
{
	int i;
	int ret;
	unsigned v[2][N];
	starpu_data_handle_t data_handles[2];
	char string[50];

	sprintf(string, "STARPU_MPI_CACHE=%d", enabled);
	putenv(string);

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_init(NULL, NULL, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");

	for(i = 0; i < 2; i++)
	{
		int mpi_rank = my_distrib(i);
		if (mpi_rank == rank)
		{
			//FPRINTF(stderr, "[%d] Owning data[%d][%d]\n", rank, x, y);
			starpu_vector_data_register(&data_handles[i], 0, (uintptr_t)&(v[i]), N, sizeof(unsigned));
		}
		else
		{
			/* I don't own that index, but will need it for my computations */
			//FPRINTF(stderr, "[%d] Neighbour of data[%d][%d]\n", rank, x, y);
			starpu_vector_data_register(&data_handles[i], -1, (uintptr_t)NULL, N, sizeof(unsigned));
		}
		starpu_data_set_rank(data_handles[i], mpi_rank);
		starpu_data_set_tag(data_handles[i], i);
	}

	for(i = 0; i < 5; i++)
	{
		ret = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[0], STARPU_R, data_handles[1], 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_insert_task");
	}

	for(i = 0; i < 5; i++)
	{
		ret = starpu_mpi_insert_task(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[1], STARPU_R, data_handles[0], 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_insert_task");
	}

	starpu_task_wait_for_all();

	for(i = 0; i < 2; i++)
	{
		starpu_data_unregister(data_handles[i]);
	}

	starpu_mpi_comm_amounts_retrieve(comm_amount);
	starpu_mpi_shutdown();
	starpu_shutdown();
}

int main(int argc, char **argv)
{
	int dst, rank, size;
	int result=0;
	size_t *comm_amount_with_cache;
	size_t *comm_amount_without_cache;
	char *string;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	string = malloc(50);
	sprintf(string, "STARPU_COMM_STATS=1");
	putenv(string);

	comm_amount_with_cache = malloc(size * sizeof(size_t));
	comm_amount_without_cache = malloc(size * sizeof(size_t));

	test_cache(rank, size, 0, comm_amount_with_cache);
	test_cache(rank, size, 1, comm_amount_without_cache);

	if (rank == 0 || rank == 1)
	{
		dst = (rank == 0) ? 1 : 0;
		result = (comm_amount_with_cache[dst] == comm_amount_without_cache[dst] * 5);
		fprintf(stderr, "Communication cache mechanism is %sworking\n", result?"":"NOT ");
	}
	else
		result = 1;

	free(comm_amount_without_cache);
	free(comm_amount_with_cache);
	free(string);

	MPI_Finalize();
	return !result;
}
#endif
