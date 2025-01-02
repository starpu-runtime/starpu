/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../variable/increment.h"

/*
 * Call acquire/release in competition with inserting task working on the same data
 */

#ifdef STARPU_QUICK_CHECK
static unsigned ntasks = 40;
#elif !defined(STARPU_LONG_CHECK)
static unsigned ntasks = 4000;
#else
static unsigned ntasks = 40000;
#endif

unsigned token = 0;
starpu_data_handle_t token_handle;

static int increment_token(int synchronous)
{
	struct starpu_task *task = starpu_task_create();
        task->synchronous = synchronous;
	task->cl = &increment_cl;
	task->handles[0] = token_handle;
	return starpu_task_submit(task);
}

static void callback(void *arg)
{
	(void)arg;
        starpu_data_release(token_handle);
}

#ifdef STARPU_DEVEL
#  warning TODO add threads
#endif

int main(int argc, char **argv)
{
	unsigned i;
	int ret;

        ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	increment_load_opencl();

	starpu_variable_data_register(&token_handle, STARPU_MAIN_RAM, (uintptr_t)&token, sizeof(unsigned));

        FPRINTF(stderr, "Token: %u\n", token);

	for(i=0; i<ntasks; i++)
	{
                ret = starpu_data_acquire_cb(token_handle, STARPU_W, callback, NULL);  // recv
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire_cb");

                ret = increment_token(0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

                ret = starpu_data_acquire_cb(token_handle, STARPU_R, callback, NULL);  // send
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire_cb");
	}

	starpu_data_unregister(token_handle);

	increment_unload_opencl();
	starpu_shutdown();

        FPRINTF(stderr, "Token: %u\n", token);
	if (token == ntasks)
		ret = EXIT_SUCCESS;
	else
		ret = EXIT_FAILURE;
	return ret;

enodev:
	starpu_data_unregister(token_handle);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	increment_unload_opencl();
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
