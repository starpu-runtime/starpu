/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013-2013  Thibaut Lambert
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
#include "../vector/memset.h"

/*
 * Try to mix starpu_data_invalidate and starpu_data_invalidate_submit
 * calls with task insertions
 */

#ifdef STARPU_QUICK_CHECK
static unsigned nloops=100;
#else
static unsigned nloops=1000;
#endif
#define VECTORSIZE	1024

static starpu_data_handle_t v_handle;

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if(starpu_cpu_worker_get_count() == 0)
	{
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

#ifdef STARPU_HAVE_VALGRIND_H
	if(RUNNING_ON_VALGRIND) nloops = 2;
#endif

	/* The buffer should never be explicitly allocated */
	starpu_vector_data_register(&v_handle, (uint32_t)-1, (uintptr_t)NULL, VECTORSIZE, sizeof(char));

	unsigned loop;
	for (loop = 0; loop < nloops; loop++)
	{
		struct starpu_task *memset_task;
		struct starpu_task *check_content_task;

		memset_task = starpu_task_create();
		memset_task->cl = &memset_cl;
		memset_task->handles[0] = v_handle;
		memset_task->detach = 0;

		ret = starpu_task_submit(memset_task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		ret = starpu_task_wait(memset_task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait");

		check_content_task = starpu_task_create();
		check_content_task->cl = &memset_check_content_cl;
		check_content_task->handles[0] = v_handle;
		check_content_task->detach = 0;

		ret = starpu_task_submit(check_content_task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		ret = starpu_task_wait(check_content_task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait");

		starpu_data_invalidate(v_handle);
	}

	for (loop = 0; loop < nloops; loop++)
	{
		struct starpu_task *memset_task;
		struct starpu_task *check_content_task;

		memset_task = starpu_task_create();
		memset_task->cl = &memset_cl;
		memset_task->handles[0] = v_handle;

		ret = starpu_task_submit(memset_task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		check_content_task = starpu_task_create();
		check_content_task->cl = &memset_check_content_cl;
		check_content_task->handles[0] = v_handle;

		ret = starpu_task_submit(check_content_task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		starpu_data_invalidate_submit(v_handle);
	}

	/* this should get rid of automatically allocated buffers */
	starpu_data_unregister(v_handle);

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(v_handle);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
