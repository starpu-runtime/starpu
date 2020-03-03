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
#include <common/thread.h>

/*
 * Run one task with regenerate=1, and thus completes several times
 * before we reset regenerate to 0 in the callback
 */

#ifdef STARPU_QUICK_CHECK
static unsigned ntasks = 64;
#else
static unsigned ntasks = 65536;
#endif
static unsigned cnt = 0;

static unsigned completed = 0;
static starpu_pthread_mutex_t mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static starpu_pthread_cond_t cond = STARPU_PTHREAD_COND_INITIALIZER;

static
void callback(void *arg)
{
	(void)arg;
	struct starpu_task *task = starpu_task_get_current();

	cnt++;

	if (cnt == ntasks)
	{
		task->regenerate = 0;
		FPRINTF(stderr, "Stop !\n");

		STARPU_PTHREAD_MUTEX_LOCK(&mutex);
		completed = 1;
		STARPU_PTHREAD_COND_SIGNAL(&cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
	}
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

	struct starpu_task task;

	starpu_task_init(&task);

	task.cl = &dummy_codelet;
	task.regenerate = 1;
	task.detach = 1;

	task.callback_func = callback;

	FPRINTF(stderr, "#tasks : %u\n", ntasks);

	start = starpu_timing_now();

	ret = starpu_task_submit(&task);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	starpu_do_schedule();
	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	if (!completed)
		STARPU_PTHREAD_COND_WAIT(&cond, &mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	end = starpu_timing_now();

	timing = end - start;

	FPRINTF(stderr, "Total: %f secs\n", timing/1000000);
	FPRINTF(stderr, "Per task: %f usecs\n", timing/ntasks);

	starpu_task_wait_for_all();
	starpu_task_clean(&task);

	starpu_shutdown();

	/* Cleanup the statically allocated tasks after shutdown, as StarPU is still working on it after the callback */
	starpu_task_clean(&task);

	return EXIT_SUCCESS;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
