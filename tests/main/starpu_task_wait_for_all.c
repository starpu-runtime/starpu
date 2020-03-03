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

#include <starpu.h>
#include <stdio.h>
#include <unistd.h>
#include "../helper.h"

/*
 * Test waiting for all tasks
 */

#ifdef STARPU_QUICK_CHECK
static unsigned ntasks = 64;
#else
static unsigned ntasks = 65536;
#endif

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

static int inject_one_task(void)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &dummy_codelet;
	task->cl_arg = NULL;
	task->callback_func = NULL;
	task->callback_arg = NULL;

	int ret = starpu_task_submit(task);
	return ret;
}

static void usage(char **argv)
{
	FPRINTF(stderr, "%s [-i ntasks] [-p sched_policy] [-h]\n", argv[0]);
	exit(-1);
}

static void parse_args(int argc, char **argv, struct starpu_conf *conf)
{
	int c;
	while ((c = getopt(argc, argv, "i:p:h")) != -1)
	switch(c)
	{
		case 'i':
			ntasks = atoi(optarg);
			break;
		case 'p':
			conf->sched_policy_name = optarg;
			break;
		case 'h':
			usage(argv);
			break;
	}
}

int main(int argc, char **argv)
{
	unsigned i;
	double timing;
	double start;
	double end;
	int ret;
	struct starpu_conf conf;

	starpu_conf_init(&conf);

	parse_args(argc, argv, &conf);

	ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	FPRINTF(stderr, "#tasks : %u\n", ntasks);

	start = starpu_timing_now();
	for (i = 0; i < ntasks; i++)
	{
		ret = inject_one_task();
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	end = starpu_timing_now();

	timing = end - start;

	FPRINTF(stderr, "Total: %f secs\n", timing/1000000);
	FPRINTF(stderr, "Per task: %f usecs\n", timing/ntasks);

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
