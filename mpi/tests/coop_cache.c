/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * childright (C) 2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * See the GNU Lesser General Public License in childING.LGPL for more details.
 */

/* This test generates a task graph that would lead to duplicate recipients if
 * MPI cache is disabled: output of the "parent" task is required by all
 * "child" tasks, each MPI rank executing two "child" tasks.
 *
 * Duplicates in the list of recipients of a broadcasts can lead to a deadlock.
 * In the NewMadeleine implementation, the following will happen, when cache
 * is disabled:
 * - Rank 0 will trigger a broadcast to ranks {1, 2, 3, 1, 2}
 * - Ranks 1, 2, and 3 will post *one* recv for the data tag 0
 * - In the binomial routing tree, the rank 2 will forward the data to rank 2 (so, itself)
 * - However, on rank 2, the first recv will be finalized only after all
 *   forwards are done. But the forward to 2 can be finished only when the second
 *   recv is posted. Posting the second recv will be done only after the first one
 *   is finalized. Hence the deadlock.
 *  */

#include <starpu_mpi.h>
#include "helper.h"

#if !defined(STARPU_HAVE_SETENV)
#warning setenv is not defined. Skipping test
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else


static void parent_cpu_func(void *descr[], void *args)
{
	starpu_sleep(2); // Give time to submit other tasks and detect coop
}

static struct starpu_codelet parent_cl =
{
	.cpu_funcs = { parent_cpu_func },
	.cpu_funcs_name = { "parent_task" },
	.nbuffers = 1,
	.modes = { STARPU_W }
};

static void child_cpu_func(void* descr[], void* args)
{
	// do nothing
}

static struct starpu_codelet child_cl =
{
	.cpu_funcs = { child_cpu_func },
	.cpu_funcs_name = { "child_task" },
	.nbuffers = 2,
	.modes = { STARPU_R, STARPU_W }
};

static inline int my_distrib(int x, int nb_nodes)
{
	return x % nb_nodes;
}

static inline void do_test(char* cache_enabled)
{
	int ret, rank, worldsize, i;
	int* data;
	starpu_data_handle_t* handles;
	struct starpu_conf conf;

	setenv("STARPU_MPI_CACHE", cache_enabled, 1);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_mpi_init_conf(NULL, NULL, 0, MPI_COMM_WORLD, &conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &worldsize);

	int nblocks = 2 * worldsize;
	int **blocks = malloc(nblocks * sizeof(int*));

	handles = malloc(nblocks*sizeof(starpu_data_handle_t));

	for (i = 0; i < nblocks; i++)
	{
		int mpi_rank = my_distrib(i, worldsize);
		if (mpi_rank == rank)
		{
			blocks[i] = calloc(320*320, sizeof(float));
			starpu_vector_data_register(&handles[i], STARPU_MAIN_RAM, (uintptr_t)blocks[i], 320*320, sizeof(float));
		}
		else
		{
			blocks[i] = NULL;
			starpu_vector_data_register(&handles[i], -1, (uintptr_t)NULL, 320*320, sizeof(float));
		}

		STARPU_ASSERT(handles[i] != NULL);
		starpu_mpi_data_register(handles[i], i, mpi_rank);
	}

	starpu_mpi_task_insert(MPI_COMM_WORLD, &parent_cl, STARPU_W, handles[0], 0);
	for (i = 1; i < nblocks-1; i++)
	{
		starpu_mpi_task_insert(MPI_COMM_WORLD, &child_cl, STARPU_R, handles[0], STARPU_W, handles[i], 0);
	}

	starpu_task_wait_for_all();

	for (i = 0; i < nblocks; i++)
	{
		starpu_data_unregister(handles[i]);

		if (my_distrib(i, worldsize) == rank)
		{
			free(blocks[i]);
		}
	}

	free(handles);
	free(blocks);

	starpu_mpi_shutdown();
}

int main(int argc, char **argv)
{
	MPI_INIT_THREAD_real(&argc, &argv, MPI_THREAD_SERIALIZED);

	do_test(/* disable cache */ "0");
	do_test(/* enable cache */ "1");

	MPI_Finalize();

	return 0;
}
#endif
