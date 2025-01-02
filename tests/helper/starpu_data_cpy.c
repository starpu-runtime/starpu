/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Test starpu_data_cpy
 */

int main(int argc, char **argv)
{
	int ret;
	int var1, var2;
	starpu_data_handle_t var1_handle, var2_handle;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	if (starpu_cpu_worker_get_count() + starpu_cuda_worker_get_count() + starpu_opencl_worker_get_count() == 0)
	{
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	var1 = 42;
	var2 = 12;

	starpu_variable_data_register(&var1_handle, STARPU_MAIN_RAM, (uintptr_t)&var1, sizeof(var1));
	starpu_variable_data_register(&var2_handle, STARPU_MAIN_RAM, (uintptr_t)&var2, sizeof(var2));

	ret = starpu_data_cpy(var2_handle, var1_handle, 0, NULL, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_cpy");

	starpu_data_acquire(var2_handle, STARPU_R);
	ret = EXIT_SUCCESS;
	if (var2 != var1)
	{
	     FPRINTF(stderr, "var2 is %d but it should be %d\n", var2, var1);
	     ret = EXIT_FAILURE;
	}
	starpu_data_release(var2_handle);

	starpu_data_unregister(var1_handle);
	starpu_data_unregister(var2_handle);
	starpu_shutdown();

	STARPU_RETURN(ret);
}
