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
 * Check that the initializer passed to starpu_data_set_reduction_methods
 * is used to initialize a handle when it is registered from NULL, and when
 * starpu_data_invalidate is called
 */

int main(void)
{
	unsigned *pvar = NULL;
	int ret;
	starpu_data_handle_t handle;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	increment_load_opencl();

	starpu_variable_data_register(&handle, -1, 0, sizeof(unsigned));

	starpu_data_set_reduction_methods(handle, NULL, &neutral_cl);

#ifdef STARPU_QUICK_CHECK
	unsigned ntasks = 32;
	unsigned nloops = 4;
#else
	unsigned ntasks = 1024;
	unsigned nloops = 16;
#endif

	unsigned loop;
	unsigned t;

	for (loop = 0; loop < nloops; loop++)
	{
		for (t = 0; t < ntasks; t++)
		{
			struct starpu_task *task = starpu_task_create();

			task->cl = &increment_cl;
			task->handles[0] = handle;

			ret = starpu_task_submit(task);
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		ret = starpu_data_acquire(handle, STARPU_R);
		pvar = starpu_data_handle_to_pointer(handle, 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
		if (*pvar != ntasks)
		{
			FPRINTF(stderr, "[end of loop] Value %u != Expected value %u\n", *pvar, ntasks);
			starpu_data_release(handle);
			starpu_data_unregister(handle);
			goto err;
		}
		starpu_data_release(handle);
		starpu_data_invalidate(handle);
	}

	starpu_data_unregister(handle);
	increment_unload_opencl();
	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(handle);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;

err:
	starpu_shutdown();
	return EXIT_FAILURE;
}
