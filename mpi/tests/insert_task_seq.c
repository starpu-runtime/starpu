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

#include <starpu_mpi.h>
#include <math.h>
#include "helper.h"

#if !defined(STARPU_HAVE_SETENV)
#warning setenv is not defined. Skipping test
int main(int argc, char **argv)
{
	return STARPU_TEST_SKIPPED;
}
#else

void func_cpu(void *descr[], void *_args)
{
	(void) descr;
	(void) _args;
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_R},
	.model = &starpu_perfmodel_nop,
};

struct starpu_codelet mycodelet2 =
{
	.cpu_funcs = {func_cpu},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &starpu_perfmodel_nop,
};

#define X     4

/* Returns the MPI node number where data is */
int my_distrib(int x, int nb_nodes)
{
	return x % nb_nodes;
}

void dotest(int rank, int size, starpu_mpi_tag_t initial_tag, char *enabled)
{
	int x, i;
	int ret;
	unsigned values[X];
	starpu_data_handle_t data_handles[X];
	struct starpu_conf conf;

	setenv("STARPU_MPI_CACHE", enabled, 1);

	FPRINTF(stderr, "Testing with cache '%s'\n", enabled);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, &conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	for(x = 0; x < X; x++)
	{
		values[x] = (rank+1)*10;
	}

	for(x = 0; x < X; x++)
	{
		int mpi_rank = my_distrib(x, size);
		if (mpi_rank == rank)
		{
			starpu_variable_data_register(&data_handles[x], STARPU_MAIN_RAM, (uintptr_t)&(values[x]), sizeof(unsigned));
		}
		else
		{
			/* I don't own this index, but will need it for my computations */
			starpu_variable_data_register(&data_handles[x], -1, (uintptr_t)NULL, sizeof(unsigned));
		}
		if (data_handles[x])
		{
			starpu_mpi_data_register(data_handles[x], initial_tag+x, mpi_rank);
		}
	}

	for(i = 0 ; i<size ; i++)
	{
		ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &mycodelet, STARPU_RW, data_handles[i], STARPU_R, data_handles[0], 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
	}

	for(i = 0 ; i<size ; i++)
	{
		// Calling starpu_mpi_get_data_on_all_nodes_detached() is necessary to make sure all nodes have a valid copy of the data
		starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, data_handles[i]);
		ret = starpu_task_insert(&mycodelet2, STARPU_RW, data_handles[i], 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	starpu_task_wait_for_all();

	for(x = 0; x < X; x++)
	{
		STARPU_ASSERT(data_handles[x]);
		starpu_data_unregister(data_handles[x]);
	}

	starpu_mpi_shutdown();
}

int main(int argc, char **argv)
{
	int rank, size;
	int barrier_ret;
	starpu_mpi_tag_t initial_tag = 0;

	MPI_INIT_THREAD_real(&argc, &argv, MPI_THREAD_SERIALIZED);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	dotest(rank, size, initial_tag, "0");
	initial_tag += X;

	/* Be sure all nodes finished the dotest function before repeating it.
	 * This is required by the dynamic broadcasts of StarPU-nmad:
	 * initialization and release of dynamic broadcasts are made when
	 * calling starpu_mpi_init() and starpu_mpi_shutdown() respectively,
	 * but the restart of the dynamic broadcast component can create
	 * deadlocks if the second test starts on a node before the first one
	 * finishes on other nodes. */
	barrier_ret = MPI_Barrier(MPI_COMM_WORLD);
	STARPU_ASSERT(barrier_ret == MPI_SUCCESS);

	dotest(rank, size, initial_tag, "1");

	MPI_Finalize();
	return 0;
}
#endif
