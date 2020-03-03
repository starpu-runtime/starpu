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
 * Try starpu_pause/resume
 */

#ifdef STARPU_QUICK_CHECK
static unsigned ntasks = 64;
#elif !defined(STARPU_LONG_CHECK)
static unsigned ntasks = 1000;
#else
static unsigned ntasks = 50000;
#endif

void dummy_func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
}

static struct starpu_codelet dummy_codelet =
{
	.cpu_funcs = {dummy_func},
	.cpu_funcs_name = {"dummy_func"},
	.cuda_funcs = {dummy_func},
	.opencl_funcs = {dummy_func},
	.model = NULL,
	.nbuffers = 0
};

int main(void)
{
	double timing;
	double start;
	double end;
	int ret;

#ifdef STARPU_HAVE_VALGRIND_H
	if(RUNNING_ON_VALGRIND) ntasks = 5;
#endif

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Check that we can submit tasks to a "paused" StarPU and then have
	 * it run normally.
	 */
	starpu_pause();
	unsigned i;
	for (i = 0; i < ntasks; i++)
	{
		ret = starpu_task_insert(&dummy_codelet, 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	start = starpu_timing_now();
	starpu_resume();
	starpu_task_wait_for_all();
	end = starpu_timing_now();
	timing = end - start;

	FPRINTF(stderr, "Without interruptions:\n\tTotal: %f secs\n", timing/1000000);
	FPRINTF(stderr, "\tPer task: %f usecs\n", timing/ntasks);

	/* Do the same thing, but with a lot of interuptions to see if there
	 * is any overhead associated with the pause/resume calls.
	 */
	starpu_pause();
	for (i = 0; i < ntasks; i++)
	{
		ret = starpu_task_insert(&dummy_codelet, 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
	starpu_resume();

	start = starpu_timing_now();
	for (i = 0; i < 100; i++)
	{
		starpu_pause();
		starpu_resume();
	}
	starpu_task_wait_for_all();
	end = starpu_timing_now();
	timing = end - start;

	FPRINTF(stderr, "With 100 interruptions:\n\tTotal: %f secs\n", timing/1000000);
	FPRINTF(stderr, "\tPer task: %f usecs\n", timing/ntasks);

	/* Finally, check that the nesting of pause/resume calls works. */
	starpu_pause();
	starpu_pause();
	starpu_resume();
	starpu_resume();

	starpu_shutdown();

	return EXIT_SUCCESS;
}
