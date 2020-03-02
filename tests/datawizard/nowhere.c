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

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>
#include "../helper.h"

/*
 * Try the NOWHERE flag
 */

static int x, y;

static void prod(void *descr[], void *arg)
{
	(void)arg;
	int *v = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);

	*v = 1;
}

static struct starpu_codelet cl_prod =
{
	.cpu_funcs = { prod },
	.nbuffers = 1,
	.modes = { STARPU_W },
};

static void callback0(void *callback_arg)
{
	(void)callback_arg;
	STARPU_ASSERT(x==0);
	STARPU_ASSERT(y==0);
}

static void callback(void *callback_arg)
{
	(void)callback_arg;
	STARPU_ASSERT(x>=1);
	STARPU_ASSERT(y>=1);
}

static struct starpu_codelet cl_nowhere =
{
	.where = STARPU_NOWHERE,
	.nbuffers = 2,
	.modes = { STARPU_R, STARPU_R },
};

static void cons(void *descr[], void *_args)
{
	(void)_args;

	int *v = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);

	STARPU_ASSERT(*v == 1);
	*v = 2;
}

static struct starpu_codelet cl_cons =
{
	.cpu_funcs = { cons },
	.nbuffers = 1,
	.modes = { STARPU_RW },
};

int main(int argc, char **argv)
{
	starpu_data_handle_t handle_x, handle_y;
	int ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_memory_nodes_get_numa_count() > 1)
	{
		/* FIXME: assumes only one RAM node */
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	starpu_variable_data_register(&handle_x, STARPU_MAIN_RAM, (uintptr_t)&x, sizeof(x));
	starpu_variable_data_register(&handle_y, STARPU_MAIN_RAM, (uintptr_t)&y, sizeof(y));

	ret = starpu_task_insert(&cl_nowhere, STARPU_R, handle_x, STARPU_R, handle_y, STARPU_CALLBACK, callback0, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_insert(&cl_prod, STARPU_W, handle_x, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_insert(&cl_prod, STARPU_W, handle_y, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_insert(&cl_nowhere, STARPU_R, handle_x, STARPU_R, handle_y, STARPU_CALLBACK, callback, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_insert(&cl_cons, STARPU_RW, handle_x, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_insert(&cl_cons, STARPU_RW, handle_y, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	starpu_data_unregister(handle_x);
	starpu_data_unregister(handle_y);

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(handle_x);
	starpu_data_unregister(handle_y);

	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
