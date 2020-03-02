/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Check that starpu_task_get_task_succs returns the set of children tasks
 */

void func_cpu(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;
}

struct starpu_codelet codelet_w =
{
	.modes = { STARPU_W },
	.cpu_funcs = {func_cpu},
	.cpu_funcs_name = {"func_cpu"},
        .nbuffers = 1
};

struct starpu_codelet codelet_r =
{
	.modes = { STARPU_R },
	.cpu_funcs = {func_cpu},
	.cpu_funcs_name = {"func_cpu"},
        .nbuffers = 1
};

int main(void)
{
        int ret;
	starpu_data_handle_t h;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_void_data_register(&h);

	starpu_tag_t tag_init = 0;

	starpu_tag_declare_deps_array((starpu_tag_t) 1, 1, &tag_init);

	struct starpu_task *task1 = starpu_task_build(&codelet_w, STARPU_W, h, STARPU_TAG, (starpu_tag_t) 1, 0);
	struct starpu_task *task2 = starpu_task_build(&codelet_r, STARPU_R, h, 0);
	struct starpu_task *task3 = starpu_task_build(&codelet_r, STARPU_R, h, 0);
	ret = starpu_task_submit(task1);
	if (ret == -ENODEV) goto enodev;
	ret = starpu_task_submit(task2);
	if (ret == -ENODEV) goto enodev;
	ret = starpu_task_submit(task3);
	if (ret == -ENODEV) goto enodev;

	struct starpu_task *tasks[4];

	ret = starpu_task_get_task_succs(task1, sizeof(tasks)/sizeof(*tasks), tasks);
	STARPU_ASSERT(ret == 2);
	STARPU_ASSERT(tasks[0] == task2 || tasks[1] == task2);
	STARPU_ASSERT(tasks[0] == task3 || tasks[1] == task3);

	starpu_tag_notify_from_apps(0);

	starpu_data_unregister(h);

	starpu_shutdown();

	STARPU_RETURN(ret?0:1);

enodev:
	starpu_shutdown();
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return STARPU_TEST_SKIPPED;
}
