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

#include <stdio.h>
#include <unistd.h>

#include <starpu.h>
#include "../helper.h"

/*
 * Try submitting tasks from different threads
 */

starpu_pthread_t threads[16];

#ifdef STARPU_QUICK_CHECK
static unsigned ntasks = 64;
#else
static unsigned ntasks = 65536;
#endif
static unsigned nthreads = 2;

void dummy_func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
}

static struct starpu_codelet dummy_codelet =
{
	.cpu_funcs = {dummy_func},
	.cuda_funcs = {dummy_func},
	.opencl_funcs = {dummy_func},
	.cpu_funcs_name = {"dummy_func"},
	.model = NULL,
	.nbuffers = 0
};

static
void *thread_func(void *arg)
{
	int ret;
	unsigned i;
	(void)arg;

	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &dummy_codelet;
		task->cl_arg = NULL;
		task->callback_func = NULL;
		task->callback_arg = NULL;

		ret = starpu_task_submit(task);
		STARPU_ASSERT_MSG(!ret, "task submission failed with error code %d", ret);
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	return NULL;
}

static void usage(char **argv)
{
	FPRINTF(stderr, "%s [-i ntasks] [-t nthreads] [-h]\n", argv[0]);
	exit(-1);
}

static void parse_args(int argc, char **argv)
{
	int c;
	while ((c = getopt(argc, argv, "i:t:h")) != -1)
	switch(c)
	{
		case 'i':
			ntasks = atoi(optarg);
			break;
		case 't':
			nthreads = atoi(optarg);
			break;
		case 'h':
			usage(argv);
			break;
	}
}

int main(int argc, char **argv)
{
	//	unsigned i;
	double timing;
	double start;
	double end;
	int ret;

	parse_args(argc, argv);

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	FPRINTF(stderr, "#tasks : %u\n", ntasks);

	start = starpu_timing_now();

	unsigned t;
	for (t = 0; t < nthreads; t++)
	{
		STARPU_PTHREAD_CREATE(&threads[t], NULL, thread_func, NULL);
	}

	for (t = 0; t < nthreads; t++)
	{
		STARPU_PTHREAD_JOIN(threads[t], NULL);
	}

	end = starpu_timing_now();

	timing = end - start;

	FPRINTF(stderr, "Total: %f secs\n", timing/1000000);
	FPRINTF(stderr, "Per task: %f usecs\n", timing/(nthreads*ntasks));

	starpu_shutdown();

	return EXIT_SUCCESS;
}
