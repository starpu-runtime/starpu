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
 * Test task submission
 */

static int i = 0, j;

void dummy_func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
	int old_i = STARPU_ATOMIC_ADD(&i, 1);
	FPRINTF(stdout, "called third task, i = %d\n", old_i+1);
}

static struct starpu_codelet dummy_codelet =
{
	.cpu_funcs = {dummy_func},
	.cuda_funcs = {dummy_func},
	.opencl_funcs = {dummy_func},
	.model = NULL,
	.nbuffers = 0
};

static void callback(void *arg)
{
	(void)arg;
	struct starpu_task *task = starpu_task_create();
	task->cl = &dummy_codelet;
	task->detach = 1;
	if (starpu_task_submit(task) == ENODEV)
		exit(STARPU_TEST_SKIPPED);
	FPRINTF(stdout, "submitted third task, i = %d\n", i);
}

static struct starpu_codelet callback_submit_codelet =
{
	.cpu_funcs = {dummy_func},
	.cuda_funcs = {dummy_func},
	.opencl_funcs = {dummy_func},
	.model = NULL,
	.nbuffers = 0
};

static void task_submit_func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
	struct starpu_task *task = starpu_task_create();
	task->cl = &callback_submit_codelet;
	task->callback_func = callback;
	task->detach = 1;
	if (starpu_task_submit(task) == ENODEV)
		exit(STARPU_TEST_SKIPPED);
	int old_i = STARPU_ATOMIC_ADD(&i, 1);
	FPRINTF(stdout, "submitted second task, i = %d\n", old_i + 1);
}

static struct starpu_codelet task_submit_codelet =
{
	.cpu_funcs = {task_submit_func},
	.cuda_funcs = {task_submit_func},
	.opencl_funcs = {task_submit_func},
	.model = NULL,
	.nbuffers = 0
};

int main(void)
{
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	struct starpu_task *task = starpu_task_create();

	task->cl = &task_submit_codelet;
	task->detach = 1;

	ret = starpu_task_submit(task);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	starpu_task_wait_for_all();
	j = i;

	starpu_shutdown();

	return j == 3 ? EXIT_SUCCESS : EXIT_FAILURE;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
