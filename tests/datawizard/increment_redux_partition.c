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
 * Check that STARPU_REDUX works with a mere incrementation and on partitioned data
 */
#define N 8
int main(int argc, char **argv)
{
	int ret;
	unsigned vec[N] = {};
	unsigned i;
	int status;
	starpu_data_handle_t handle;

	/* Not supported yet */
	if (starpu_getenv_number_default("STARPU_GLOBAL_ARBITER", 0) > 0)
		return STARPU_TEST_SKIPPED;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	increment_load_opencl();

	for (i = 0; i < N; i++)
		vec[i] = i;
	starpu_vector_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&vec, N, sizeof(unsigned));

	/* Partition the vector in PARTS sub-variables */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_pick_variable,
		.filter_arg_ptr = (void*)(uintptr_t) 0,
		.nchildren = N,
		/* the children use a variable interface*/
		.get_child_ops = starpu_vector_filter_pick_variable_child_ops
	};
	starpu_data_partition(handle, &f);

	for (i = 0; i < N; i++)
	{
		starpu_data_handle_t sub_handle = starpu_data_get_sub_data(handle, 1, i);
		starpu_data_set_reduction_methods(sub_handle, &redux_cl, &neutral_cl);
	}

#ifdef STARPU_QUICK_CHECK
	unsigned ntasks = 32;
#else
	unsigned ntasks = 1024;
#endif

	unsigned t;

	for (i = 0; i < N; i++)
	{
		starpu_data_handle_t sub_handle = starpu_data_get_sub_data(handle, 1, i);

		for (t = 0; t < ntasks; t++)
		{
			struct starpu_task *task = starpu_task_create();

			task->cl = &increment_redux_cl;
			task->handles[0] = sub_handle;

			ret = starpu_task_submit(task);
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}
	}

	status = EXIT_SUCCESS;
	for (i = 0; i < N; i++)
	{
		starpu_data_handle_t sub_handle = starpu_data_get_sub_data(handle, 1, i);

		ret = starpu_data_acquire(sub_handle, STARPU_R);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
		if (vec[i] != i + ntasks)
		{
			FPRINTF(stderr, "[end of loop] Value %u != Expected value %u\n", vec[i], ntasks);
			status = EXIT_FAILURE;
		}
		starpu_data_release(sub_handle);
	}

	starpu_data_unpartition(handle, STARPU_MAIN_RAM);

	starpu_data_acquire(handle, STARPU_R);
	for (i = 0; i < N; i++)
	{
		if (vec[i] != i + ntasks)
		{
			FPRINTF(stderr, "[end of loop] Value %u != Expected value %u\n", vec[i], ntasks);
			status = EXIT_FAILURE;
		}
	}
	starpu_data_release(handle);

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
	STARPU_RETURN(EXIT_FAILURE);

}
