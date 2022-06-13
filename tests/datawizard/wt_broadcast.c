/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Test using starpu_data_set_wt_mask(handle, ~0);, i.e. broadcasting the
 * result on all devices as soon as it is available.
 */

static unsigned var = 0;
static starpu_data_handle_t handle;

int main(void)
{
	int ret;

	struct starpu_conf conf;
	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);

        conf.ncpus = -1;
        conf.ncuda = -1;
        conf.nopencl = -1;
        conf.nmpi_ms = -1;
        conf.ntcpip_ms = -1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	increment_load_opencl();

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&var, sizeof(unsigned));

	/* Create a mask with all the memory nodes, so that we can ask StarPU
	 * to broadcast the handle whenever it is modified. */
	starpu_data_set_wt_mask(handle, ~0);

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
	}

	starpu_data_unregister(handle);

	ret = EXIT_SUCCESS;
	if (var != ntasks*nloops)
	{
		FPRINTF(stderr, "VAR is %u should be %u\n", var, ntasks);
		ret = EXIT_FAILURE;
	}

	increment_unload_opencl();

	starpu_shutdown();

	return ret;

enodev:
	starpu_data_unregister(handle);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
