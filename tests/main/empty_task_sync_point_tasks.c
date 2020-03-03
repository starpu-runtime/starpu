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
#include "../helper.h"

/*
 * Test using a task with NULL codelet as a synchronization task through task dependencies
 */

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

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* {A,B,C} -> D -> {E,F}, D is empty */
	struct starpu_task *taskA = starpu_task_create();
	taskA->cl = &dummy_codelet;

	struct starpu_task *taskB = starpu_task_create();
	taskB->cl = &dummy_codelet;

	struct starpu_task *taskC = starpu_task_create();
	taskC->cl = &dummy_codelet;

	struct starpu_task *taskD = starpu_task_create();
	taskD->cl = NULL;

	struct starpu_task *taskE = starpu_task_create();
	taskE->cl = &dummy_codelet;

	struct starpu_task *taskF = starpu_task_create();
	taskF->cl = &dummy_codelet;

	starpu_task_declare_deps(taskD, 3, taskA, taskB, taskC);
	starpu_task_declare_deps_array(taskE, 1, &taskD);
	starpu_task_declare_deps_array(taskF, 1, &taskD);

	ret = starpu_task_submit(taskA); STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	ret = starpu_task_submit(taskB); STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	ret = starpu_task_submit(taskC); STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	ret = starpu_task_submit(taskD); STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	ret = starpu_task_submit(taskE); STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	ret = starpu_task_submit(taskF); STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	starpu_shutdown();

	return EXIT_SUCCESS;
}
