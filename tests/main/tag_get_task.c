/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Test that starpu_tag_get_task returns the proper task
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

static void callback(void *tag)
{
	fflush(stderr);
	FPRINTF(stderr, "Callback for tag %p\n", tag);
	fflush(stderr);
}

int main(int argc, char **argv)
{
	struct starpu_task *task;
	starpu_tag_t tag = 0x42;
	int ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV)
		return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* create a new dummy task with a tag */
	task = starpu_task_create();
	task->callback_func = callback;
	task->callback_arg = (void *)tag;
	task->cl = &dummy_codelet;
	task->cl_arg = NULL;
	task->destroy = 0; /* tell StarPU to not destroy the task */
	task->use_tag = 1;
	task->tag_id = tag;

	/* execute the task */
	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	/* check that starpu_tag_get_task() returns the correct task */
	ret = (starpu_tag_get_task(task->tag_id) != task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_tag_get_task");

	starpu_task_destroy(task);
	starpu_shutdown();

	return EXIT_SUCCESS;
}
