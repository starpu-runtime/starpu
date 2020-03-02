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

#include <stdio.h>
#include <starpu.h>
#include "../helper.h"

/*
 * Try calling starpu_initialize from different threads in parallel
 */

#define NUM_THREADS 5

int *glob_argc;
char ***glob_argv;

static
void *launch_starpu(void *unused)
{
	int ret;
	(void) unused;
	ret = starpu_initialize(NULL, glob_argc, glob_argv);
	if (ret == -ENODEV)
		exit(STARPU_TEST_SKIPPED);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	return NULL;
}

static
void *shutdown_starpu(void *unused)
{
	(void) unused;
	starpu_shutdown();
	return NULL;
}

int main(int argc, char **argv)
{
	unsigned i;
	double timing;
	double start;
	double end;

	glob_argc = &argc;
	glob_argv = &argv;

	starpu_pthread_t threads[NUM_THREADS];

	start = starpu_timing_now();

	for (i = 0; i < NUM_THREADS; ++i)
	{
		STARPU_PTHREAD_CREATE(&threads[i], NULL, launch_starpu, NULL);
	}

	for (i = 0; i < NUM_THREADS; ++i)
	{
		STARPU_PTHREAD_JOIN(threads[i], NULL);
	}

	end = starpu_timing_now();

	timing = end - start;

	FPRINTF(stderr, "Success : %d threads launching simultaneously starpu_init\n", NUM_THREADS);
	FPRINTF(stderr, "Total: %f secs\n", timing/1000000);
	FPRINTF(stderr, "Per task: %f usecs\n", timing/NUM_THREADS);

	for (i = 0; i < NUM_THREADS; i++)
	{
		STARPU_PTHREAD_CREATE(&threads[i], NULL, shutdown_starpu, NULL);
	}

	for (i = 0; i < NUM_THREADS; i++)
	{
		STARPU_PTHREAD_JOIN(threads[i], NULL);
	}

	return EXIT_SUCCESS;
}
