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
 * Try accessing a variable in read-only mode
 */

#ifdef STARPU_USE_OPENCL
static void codelet(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;
	FPRINTF(stderr, "codelet\n");
}
#endif

static struct starpu_codelet cl =
{
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {codelet},
#endif
	.nbuffers = 1,
	.modes = {STARPU_R}
};

int main(void)
{
	int ret;
	int var = 42;
	starpu_data_handle_t handle;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	int copy = starpu_asynchronous_copy_disabled();
	FPRINTF(stderr, "copy %d\n", copy);

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&var, sizeof(var));

	ret = starpu_task_insert(&cl,
				 STARPU_R, handle,
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();

	starpu_data_unregister(handle);

	starpu_shutdown();

	return 0;

enodev:
	starpu_data_unregister(handle);
	starpu_shutdown();
	/* yes, we do not perform the computation but we did detect that no one
	 * could perform the kernel, so this is not an error from StarPU */
	fprintf(stderr, "WARNING: No one can execute this task\n");
	return STARPU_TEST_SKIPPED;
}
