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
#include <unistd.h>
#include <starpu.h>
#include "../helper.h"

/*
 * Test that starpu_task_wait_for_all can work with a regenerating task
 */

#ifdef STARPU_QUICK_CHECK
static unsigned ntasks = 64;
#else
static unsigned ntasks = 1024;
#endif

static void callback(void *arg)
{
	struct starpu_task *task = starpu_task_get_current();

	unsigned *cnt = (unsigned *) arg;

	(*cnt)++;

	if (*cnt == ntasks)
		task->regenerate = 0;
}

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

static void parse_args(int argc, char **argv)
{
	int c;
	while ((c = getopt(argc, argv, "i:")) != -1)
	switch(c)
	{
		case 'i':
			ntasks = atoi(optarg);
			break;
	}
}

#define K	128

int main(int argc, char **argv)
{
	int ret;
	double timing;
	double start;
	double end;

	parse_args(argc, argv);

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	struct starpu_task task[K];
	unsigned cnt[K];

	int i;
	for (i = 0; i < K; i++)
	{
		starpu_task_init(&task[i]);
		cnt[i] = 0;

		task[i].cl = &dummy_codelet;
		task[i].regenerate = 1;
		task[i].detach = 1;

		task[i].callback_func = callback;
		task[i].callback_arg = &cnt[i];
	}

	FPRINTF(stderr, "#tasks : %d x %u tasks\n", K, ntasks);

	start = starpu_timing_now();
	
	for (i = 0; i < K; i++)
	{
		ret = starpu_task_submit(&task[i]);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_wait_for_all();
	for (i = 0; i < K; i++)
		starpu_task_clean(&task[i]);

	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	end = starpu_timing_now();

	/* Check that all the tasks have been properly executed */
	unsigned total_cnt = 0;
	for (i = 0; i < K; i++)
		total_cnt += cnt[i];

	STARPU_ASSERT(total_cnt == K*ntasks);

	timing = end - start;

	FPRINTF(stderr, "Total: %f secs\n", timing/1000000);
	FPRINTF(stderr, "Per task: %f usecs\n", timing/(K*ntasks));

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
