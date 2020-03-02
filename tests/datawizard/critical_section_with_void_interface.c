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
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>
#include "../helper.h"

/*
 * Use a void interface to protect the access to a variable that is not declared to StarPU
 */

starpu_data_handle_t void_handle;

int critical_var;

void critical_section(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;

	/* We do not protect this variable because it is only accessed when the
	 * "void_handle" piece of data is accessed. */
	critical_var++;
}

static struct starpu_codelet cl =
{
	.cpu_funcs = {critical_section},
	.cuda_funcs = {critical_section},
	.opencl_funcs = {critical_section},
	.nbuffers = 1,
	.modes = {STARPU_RW}
};

int main(void)
{
#ifdef STARPU_QUICK_CHECK
	int ntasks = 10;
#else
	int ntasks = 1000;
#endif

	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	critical_var = 0;

	/* Create a void data which will be used as an exclusion mechanism. */
	starpu_void_data_register(&void_handle);

	int i;
	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &cl;
		task->handles[0] = void_handle;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_data_unregister(void_handle);

	ret = (critical_var == ntasks) ? EXIT_SUCCESS : EXIT_FAILURE;

	starpu_shutdown();

	return ret;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
