/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010-2010  Mehdi Juhoor
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
 * Test running a task on a partitioned data, then on the unpartitioned
 * data, etc. in a loop
 */

#ifdef STARPU_QUICK_CHECK
#define NITER		100
#else
#define NITER		1000
#endif
#define VECTORSIZE	1024

float *buffer;

starpu_data_handle_t v_handle;

static struct starpu_task* create_task(starpu_data_handle_t handle)
{
	starpu_codelet_nop.nbuffers = 1;
	starpu_codelet_nop.modes[0] = STARPU_RW;

	struct starpu_task *task;

	task = starpu_task_create();
	task->cl = &starpu_codelet_nop;
	task->handles[0] = handle;
	task->detach = 0;

	return task;
}

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	ret = starpu_malloc((void **)&buffer, VECTORSIZE);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_malloc");

	starpu_vector_data_register(&v_handle, STARPU_MAIN_RAM, (uintptr_t)buffer, VECTORSIZE, sizeof(char));

	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_divide_in_2,
		/* there are only 2 children */
		.nchildren = 2,
		/* the length of the first part */
		.filter_arg = VECTORSIZE/2
	};

	unsigned iter;
	for (iter = 0; iter < NITER; iter++)
	{
		struct starpu_task *tasks[3];

		starpu_data_map_filters(v_handle, 1, &f);

		tasks[0] = create_task(starpu_data_get_sub_data(v_handle, 1, 0));
		ret = starpu_task_submit(tasks[0]);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		tasks[1] = create_task(starpu_data_get_sub_data(v_handle, 1, 1));
		ret = starpu_task_submit(tasks[1]);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		ret = starpu_task_wait_for_all();
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

		starpu_data_unpartition(v_handle, STARPU_MAIN_RAM);

		tasks[2] = create_task(v_handle);
		ret = starpu_task_submit(tasks[2]);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		ret = starpu_task_wait_for_all();
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

		starpu_task_destroy(tasks[0]);
		starpu_task_destroy(tasks[1]);
		starpu_task_destroy(tasks[2]);
	}

	starpu_data_unregister(v_handle);
	starpu_free_noflag(buffer, VECTORSIZE);
	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(v_handle);
	starpu_free_noflag(buffer, VECTORSIZE);
	starpu_shutdown();
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return STARPU_TEST_SKIPPED;
}
