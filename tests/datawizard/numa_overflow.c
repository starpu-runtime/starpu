/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>
#include "../helper.h"

#if !defined(STARPU_HAVE_SETENV)
#warning setenv is not defined. Skipping test
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else

#define ITER 10
#define N 10
#define SIZE (10*1024*1024)

/*
 * Check that when overflowing a NUMA node we manage to revert to other nodes.
 */

static void nop(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
}

static struct starpu_codelet cl_r =
{
	.cpu_funcs = { nop },
	.nbuffers = 1,
	.modes = { STARPU_R },
};

static struct starpu_codelet cl_rw =
{
	.cpu_funcs = { nop },
	.nbuffers = 1,
	.modes = { STARPU_RW },
};

int main(int argc, char **argv)
{
	starpu_data_handle_t handles[N];
	uintptr_t data[N];
	int ret;
	unsigned i, j;
	char s[16];
	int worker;

	snprintf(s, sizeof(s), "%u", (N*3/4)*SIZE/(1024*1024));
	/* We make NUMA nodes not big enough for all data */
	setenv("STARPU_LIMIT_CPU_NUMA_MEM", s, 1);

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, &worker, 1) == 0
	    || starpu_memory_nodes_get_numa_count() <= 1)
	{
		/* We need several NUMA nodes */
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	/* We distribute some data on both NUMA nodes */
	for (i = 0; i < N; i++)
	{
		data[i] = starpu_malloc_on_node(i%2, SIZE);
		memset((void*) data[i], 0, SIZE);
		starpu_variable_data_register(&handles[i], i%2, data[i], SIZE);
	}

	/* And now we try to execute all tasks on worker 0, that will fail if
	 * StarPU doesn't manage to evict some memory */
	for (j = 0; j < ITER; j++)
		for (i = 0; i < N; i++)
		{
			if (rand() % 2 == 0)
				ret = starpu_task_insert(&cl_r, STARPU_R, handles[i], STARPU_EXECUTE_ON_WORKER, worker, 0);
			else
				ret = starpu_task_insert(&cl_rw, STARPU_RW, handles[i], STARPU_EXECUTE_ON_WORKER, worker, 0);
			if (ret == ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

	for (i = 0; i < N; i++)
	{
		starpu_data_unregister(handles[i]);
		starpu_free_on_node(i%2, data[i], SIZE);
	}

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	for (i = 0; i < N; i++)
	{
		starpu_data_unregister(handles[i]);
		starpu_free_on_node(i%2, data[i], SIZE);
	}

	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}

#endif
