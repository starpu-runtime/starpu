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
#include "../variable/increment.h"
#include "../helper.h"

/*
 * Test starpu_data_dup_ro
 */

int main(int argc, char **argv)
{
	int ret;
	unsigned var1, *var;
	starpu_data_handle_t var1_handle, var2_handle, var3_handle, var4_handle, var5_handle;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	if (starpu_cpu_worker_get_count() + starpu_cuda_worker_get_count() + starpu_opencl_worker_get_count() == 0)
	{
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	increment_load_opencl();

	var1 = 42;
	starpu_variable_data_register(&var1_handle, STARPU_MAIN_RAM, (uintptr_t)&var1, sizeof(var1));

	/* Make a duplicate of the original data */
	ret = starpu_data_dup_ro(&var2_handle, var1_handle, 1);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_dup_ro");

	/* Free it */
	starpu_data_unregister(var2_handle);

	/* Make another duplicate of the original data */
	ret = starpu_data_dup_ro(&var2_handle, var1_handle, 1);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_dup_ro");

	/* Free it through submit */
	starpu_data_unregister_submit(var2_handle);

	/* Make another duplicate of the original data */
	ret = starpu_data_dup_ro(&var2_handle, var1_handle, 1);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_dup_ro");

	/* Make a second duplicate of the original data */
	ret = starpu_data_dup_ro(&var3_handle, var1_handle, 1);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_dup_ro");
	STARPU_ASSERT(var3_handle == var2_handle);

	/* Make a duplicate of a duplicate */
	ret = starpu_data_dup_ro(&var4_handle, var2_handle, 1);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_dup_ro");
	STARPU_ASSERT(var4_handle == var2_handle);

	ret = starpu_task_insert(&increment_cl, STARPU_RW, var1_handle, 0);
	if (ret == -ENODEV)
	{
		starpu_data_unregister(var1_handle);
		starpu_data_unregister(var2_handle);
		starpu_data_unregister(var3_handle);
		starpu_data_unregister(var4_handle);
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	/* Make a duplicate of the new value */
	ret = starpu_data_dup_ro(&var5_handle, var1_handle, 1);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_dup_ro");

	starpu_data_acquire(var2_handle, STARPU_R);
	var = starpu_data_get_local_ptr(var2_handle);
	ret = EXIT_SUCCESS;
	if (*var != 42)
	{
	     FPRINTF(stderr, "var2 is %u but it should be %d\n", *var, 42);
	     ret = EXIT_FAILURE;
	}
	starpu_data_release(var2_handle);

	starpu_data_acquire(var3_handle, STARPU_R);
	var = starpu_data_get_local_ptr(var3_handle);
	if (*var != 42)
	{
	     FPRINTF(stderr, "var3 is %u but it should be %d\n", *var, 42);
	     ret = EXIT_FAILURE;
	}
	starpu_data_release(var3_handle);

	starpu_data_acquire(var4_handle, STARPU_R);
	var = starpu_data_get_local_ptr(var4_handle);
	if (*var != 42)
	{
	     FPRINTF(stderr, "var4 is %u but it should be %d\n", *var, 42);
	     ret = EXIT_FAILURE;
	}
	starpu_data_release(var4_handle);

	starpu_data_acquire(var5_handle, STARPU_R);
	var = starpu_data_get_local_ptr(var5_handle);
	if (*var != 43)
	{
	     FPRINTF(stderr, "var5 is %u but it should be %d\n", *var, 43);
	     ret = EXIT_FAILURE;
	}
	starpu_data_release(var5_handle);

	starpu_data_unregister(var1_handle);
	starpu_data_unregister(var2_handle);
	starpu_data_unregister(var3_handle);
	starpu_data_unregister(var4_handle);
	starpu_data_unregister(var5_handle);

	increment_unload_opencl();

	starpu_shutdown();

	STARPU_RETURN(ret);
}
