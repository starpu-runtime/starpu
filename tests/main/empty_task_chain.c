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
 * Create a chain of tasks with NULL codelet, using manual dependencies
 */

#define N	4

int main(int argc, char **argv)
{
	int i, ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	struct starpu_task **tasks = (struct starpu_task **) malloc(N*sizeof(struct starpu_task *));

	for (i = 0; i < N; i++)
	{
		tasks[i] = starpu_task_create();
		tasks[i]->cl = NULL;

		if (i > 0)
		{
			starpu_task_declare_deps_array(tasks[i], 1, &tasks[i-1]);
		}

		if (i == (N-1))
			tasks[i]->detach = 0;
	}

	for (i = 1; i < N; i++)
	{
		ret = starpu_task_submit(tasks[i]);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_submit(tasks[0]);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_wait(tasks[N-1]);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait");

	starpu_shutdown();
	free(tasks);

	return EXIT_SUCCESS;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
