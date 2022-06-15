/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
static unsigned ntasks = 10;
#else
static unsigned ntasks = 10000;
#endif

unsigned token = 0;
starpu_data_handle_t token_handle;

static int increment_token(void)
{
	int ret;
	struct starpu_task *task = starpu_task_create();
        task->synchronous = 1;
	task->cl = &increment_cl;
	task->handles[0] = token_handle;
	ret = starpu_task_submit(task);
	return ret;
}

static void callback(void *arg)
{
	(void)arg;
	token++;
	starpu_data_release(token_handle);
}

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
		/* synchronize data in RAM */
                ret = starpu_data_acquire(token_handle, STARPU_RW);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");

                token ++;
                starpu_data_release(token_handle);

                ret = increment_token();
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

                ret = starpu_data_acquire_cb(token_handle, STARPU_RW, callback, NULL);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire_cb");
	}

	starpu_data_unregister(token_handle);

	increment_unload_opencl();
	starpu_shutdown();

        FPRINTF(stderr, "Token: %u\n", token);
	if (token == ntasks * 3)
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
