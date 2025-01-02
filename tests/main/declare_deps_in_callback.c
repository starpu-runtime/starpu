/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Test that we can declare deps from the callback of the task
 */

#ifdef STARPU_QUICK_CHECK
  #define NLOOPS	4
#else
  #define NLOOPS	128
#endif

static void callback(void *arg)
{
	struct starpu_task *taskA, *taskB;
	int ret;

	taskA = starpu_task_get_current();
	taskB = (struct starpu_task *) arg;

	starpu_task_declare_deps_array(taskB, 1, &taskA);
	ret = starpu_task_submit(taskB);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

static struct starpu_task *create_dummy_task(void)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &starpu_codelet_nop;
	task->cl_arg = NULL;

	return task;
}

int main(int argc, char **argv)
{
	int ret;
	unsigned loop;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	struct starpu_task *taskA, *taskB;

	for (loop = 0; loop < NLOOPS; loop++)
	{
		taskA = create_dummy_task();
		taskB = create_dummy_task();

		taskA->callback_func = callback;
		taskA->callback_arg = taskB;

		ret = starpu_task_submit(taskA);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
