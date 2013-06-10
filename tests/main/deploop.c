/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2011, 2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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

/*
 * Create task A and B such that
 * - B depends on A by tag dependency.
 * - A would depend on B by data dependency, but we disable that.
 */

#include <stdio.h>
#include <unistd.h>

#include <starpu.h>
#include "../helper.h"

static void dummy_func(void *descr[] STARPU_ATTRIBUTE_UNUSED, void *arg STARPU_ATTRIBUTE_UNUSED)
{
	FPRINTF(stderr,"executing task %p\n", starpu_task_get_current());
}

static struct starpu_codelet dummy_codelet = 
{
	.cpu_funcs = {dummy_func, NULL},
	.cuda_funcs = {dummy_func, NULL},
	.opencl_funcs = {dummy_func, NULL},
	.model = NULL,
	.nbuffers = 1,
	.modes = { STARPU_RW }
};

int main(int argc, char **argv)
{
	int ret;
	starpu_data_handle_t handle;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_void_data_register(&handle);

	struct starpu_task *taskA, *taskB;

	/* Make B depend on A */
	starpu_tag_declare_deps(1, 1, (starpu_tag_t) 0);

	taskA = starpu_task_create();
	taskA->cl = &dummy_codelet;
	taskA->tag_id = 0;
	taskA->use_tag = 1;
	taskA->handles[0] = handle;
	taskA->sequential_consistency = 0;
	FPRINTF(stderr,"A is %p\n", taskA);

	taskB = starpu_task_create();
	taskB->cl = &dummy_codelet;
	taskB->tag_id = 1;
	taskB->use_tag = 1;
	taskB->handles[0] = handle;
	FPRINTF(stderr,"B is %p\n", taskB);

	ret = starpu_task_submit(taskB);
	if (ret == -ENODEV)
		return STARPU_TEST_SKIPPED;
	ret = starpu_task_submit(taskA);
	if (ret == -ENODEV)
		return STARPU_TEST_SKIPPED;

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	starpu_data_unregister(handle);

	starpu_shutdown();

	return EXIT_SUCCESS;
}
