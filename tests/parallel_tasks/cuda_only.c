/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <limits.h>
#include <unistd.h>
#include "../helper.h"

/*
 * Check that pheft works with only GPUs
 */

void codelet_null(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;
}

struct starpu_perfmodel model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "test"
};

static struct starpu_codelet cl =
{
	.cuda_funcs = {codelet_null},
	.model = &model,
	.nbuffers = 1,
	.modes = {STARPU_R}
};

struct starpu_perfmodel model2 =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "test2"
};

static struct starpu_codelet cl2 =
{
	.cuda_funcs = {codelet_null},
	.model = &model2,
	.nbuffers = 1,
	.modes = {STARPU_W}
};

int main(void)
{
	int ret;
	starpu_data_handle_t handle;
	unsigned data;

        struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.sched_policy_name = "pheft";

	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&data, sizeof(data));

	unsigned iter;
	struct starpu_task *task;
	for (iter = 0; iter < 100; iter++)
	{
		task = starpu_task_create();
		task->cl = &cl;
		task->handles[0] = handle;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		task = starpu_task_create();
		task->cl = &cl2;
		task->handles[0] = handle;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	starpu_data_unregister(handle);
	starpu_shutdown();

	STARPU_RETURN(EXIT_SUCCESS);

enodev:
	task->destroy = 0;
	starpu_task_destroy(task);
	starpu_data_unregister(handle);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	STARPU_RETURN(STARPU_TEST_SKIPPED);
}
