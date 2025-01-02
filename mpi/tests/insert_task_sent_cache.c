/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#if !defined(STARPU_HAVE_SETENV)
#warning setenv is not defined. Skipping test
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else

void func_cpu(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_R},
	.model = &starpu_perfmodel_nop,
};

#define NB_ELEMENTS 1000
#define NB_DATA     2

/* Returns the MPI node number where data indexes index is */
int my_distrib(int x)
{
	return x;
}

void test_cache(int rank, starpu_mpi_tag_t initial_tag, char *enabled, size_t *comm_amount)
{
	int i;
	int ret;
	unsigned *v[NB_DATA];
	starpu_data_handle_t data_handles[NB_DATA];
	struct starpu_conf conf;

	setenv("STARPU_MPI_CACHE", enabled, 1);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, &conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_coop_sends_set_use(0); // disable coop_sends to avoid having wrong results when cache is disabled

	for(i = 0; i < NB_DATA; i++)
	{
		int j;
		starpu_malloc((void **)&v[i], NB_ELEMENTS * sizeof(unsigned));
		for(j=0 ; j<NB_ELEMENTS ; j++)
		{
			v[i][j] = 12;
		}
	}

	for(i = 0; i < NB_DATA; i++)
	{
		int mpi_rank = my_distrib(i);
		if (mpi_rank == rank)
		{
			//FPRINTF(stderr, "[%d] Owning data[%d][%d]\n", rank, x, y);
			starpu_vector_data_register(&data_handles[i], STARPU_MAIN_RAM, (uintptr_t)v[i], NB_ELEMENTS, sizeof(unsigned));
		}
		else
		{
			/* I don't own this index, but will need it for my computations */
			//FPRINTF(stderr, "[%d] Neighbour of data[%d][%d]\n", rank, x, y);
			starpu_vector_data_register(&data_handles[i], -1, (uintptr_t)NULL, NB_ELEMENTS, sizeof(unsigned));
		}
		starpu_mpi_data_register(data_handles[i], initial_tag+i, mpi_rank);
	}

	for(i = 0; i < 5; i++)
	{
		ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[0], STARPU_R, data_handles[1], 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
	}

	for(i = 0; i < 5; i++)
	{
		ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[1], STARPU_R, data_handles[0], 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
	}

	for(i = 0; i < 5; i++)
	{
		starpu_mpi_cache_flush(MPI_COMM_WORLD, data_handles[0]);
	}

	for(i = 0; i < 5; i++)
	{
		ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[1], STARPU_R, data_handles[0], 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
	}

	starpu_task_wait_for_all();

	for(i = 0; i < NB_DATA; i++)
	{
		starpu_data_unregister(data_handles[i]);
		starpu_free_noflag(v[i], NB_DATA * sizeof(unsigned));
	}

	starpu_mpi_comm_stats_retrieve(comm_amount);
	starpu_mpi_shutdown();
}

int main(int argc, char **argv)
{
	int rank, size;
	int result=0;
	size_t *comm_amount_with_cache;
	size_t *comm_amount_without_cache;
	starpu_mpi_tag_t initial_tag = 0;

	MPI_INIT_THREAD_real(&argc, &argv, MPI_THREAD_SERIALIZED);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	setenv("STARPU_MPI_STATS", "1", 1);

	comm_amount_with_cache = malloc(size * sizeof(size_t));
	comm_amount_without_cache = malloc(size * sizeof(size_t));

	test_cache(rank, initial_tag, "0", comm_amount_with_cache);
	initial_tag += NB_DATA;
	test_cache(rank, initial_tag, "1", comm_amount_without_cache);

	if (rank == 0 || rank == 1)
	{
		int dst = (rank == 0) ? 1 : 0;
		result = (comm_amount_with_cache[dst] == comm_amount_without_cache[dst] * 5);
		FPRINTF(stderr, "[%d] Communication cache mechanism is %sworking\n", rank, result?"":"NOT ");
	}
	else
	{
		result = 1;
	}

	free(comm_amount_without_cache);
	free(comm_amount_with_cache);

	MPI_Finalize();
	return rank == 0 ? !result : 0;
}
#endif
