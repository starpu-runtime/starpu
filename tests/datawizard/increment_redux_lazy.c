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
 * Check that STARPU_REDUX works with a mere incrementation, but without
 * initializing the variable
 */

int main(int argc, char **argv)
{
	int ret;
	unsigned *var;
	starpu_data_handle_t handle;

	/* Not supported yet */
	if (starpu_getenv_number_default("STARPU_GLOBAL_ARBITER", 0) > 0)
		return STARPU_TEST_SKIPPED;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	if (starpu_cpu_worker_get_count() + starpu_cuda_worker_get_count() + starpu_opencl_worker_get_count() + starpu_hip_worker_get_count() == 0)
	{
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	increment_load_opencl();

	starpu_variable_data_register(&handle, -1, (uintptr_t)NULL, sizeof(unsigned));

	starpu_data_set_reduction_methods(handle, &redux_cl, &neutral_cl);

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

			task->cl = &increment_redux_cl;
			task->handles[0] = handle;

			ret = starpu_task_submit(task);
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		ret = starpu_data_acquire(handle, STARPU_R);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
		var = (unsigned*) starpu_variable_get_local_ptr(handle);
		starpu_data_release(handle);

		if (*var != ntasks*(loop + 1))
		{
			ret = EXIT_FAILURE;
			FPRINTF(stderr, "[end of loop] Value %u != Expected value %u\n", *var, ntasks * (loop+1));
			goto err;
		}
	}

	ret = starpu_data_acquire(handle, STARPU_R);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");
	var = (unsigned*) starpu_variable_get_local_ptr(handle);

	if (*var != ntasks*nloops)
	{
		ret = EXIT_FAILURE;
		FPRINTF(stderr, "Value %u != Expected value %u\n", *var, ntasks * (loop+1));
		goto err;
	}

	starpu_data_release(handle);
	starpu_data_unregister(handle);

	increment_unload_opencl();

err:
	starpu_shutdown();
	STARPU_RETURN(ret);

enodev:
	starpu_data_unregister(handle);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	STARPU_RETURN(STARPU_TEST_SKIPPED);
}
