/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Test starpu_data_move
 */

static int ret;

static void check_value(const char *name, starpu_data_handle_t handle, unsigned expected)
{
	starpu_data_acquire(handle, STARPU_R);
	unsigned *var = starpu_data_get_local_ptr(handle);
	ret = EXIT_SUCCESS;
	if (*var != expected)
	{
	     FPRINTF(stderr, "%s is %u but it should be %d\n", name, *var, expected);
	     ret = EXIT_FAILURE;
	}
	starpu_data_release(handle);
}

int main(int argc, char **argv)
{
	starpu_data_handle_t var1_handle, var2_handle, var3_handle, var4_handle, var5_handle, var6_handle;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	if (starpu_cpu_worker_get_count() + starpu_cuda_worker_get_count() + starpu_opencl_worker_get_count() == 0)
	{
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	increment_load_opencl();

	starpu_variable_data_register(&var1_handle, -1, 0, sizeof(unsigned));


	/* Check that tasks work */
	ret = starpu_task_insert(&neutral_cl, STARPU_W, var1_handle, 0);
	if (ret == -ENODEV)
	{
		starpu_data_unregister(var1_handle);
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	ret = starpu_task_insert(&increment_cl, STARPU_RW, var1_handle, 0);
	if (ret == -ENODEV)
	{
		starpu_data_unregister(var1_handle);
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	/* Take note of the original pointer */
	starpu_data_acquire(var1_handle, STARPU_R);
	unsigned *var = starpu_data_get_local_ptr(var1_handle);
	starpu_data_release(var1_handle);

	/* Move the original data */
	starpu_variable_data_register(&var2_handle, -1, 0, sizeof(unsigned));
	starpu_data_move(var2_handle, var1_handle);
	ret = starpu_task_insert(&increment_cl, STARPU_RW, var2_handle, 0);
	STARPU_ASSERT(ret == 0);

	/* Check that it is correct */
	check_value("var2", var2_handle, 2);

	/* Check that we now have the original pointer in var2 */
	starpu_data_acquire(var2_handle, STARPU_R);
	STARPU_ASSERT(var == starpu_data_get_local_ptr(var2_handle));
	starpu_data_release(var2_handle);

	/* Check that var1 now has a different pointer */
	starpu_data_acquire(var1_handle, STARPU_W);
	STARPU_ASSERT(var != starpu_data_get_local_ptr(var1_handle));
	starpu_data_release(var1_handle);

	/* Free it */
	starpu_data_unregister(var2_handle);


	/* Make another value of the original data */
	ret = starpu_task_insert(&neutral_cl, STARPU_W, var1_handle, 0);
	STARPU_ASSERT(ret == 0);
	ret = starpu_task_insert(&increment_cl, STARPU_RW, var1_handle, 0);
	STARPU_ASSERT(ret == 0);
	/* Check that it is correct */
	check_value("var1", var1_handle, 1);

	starpu_variable_data_register(&var2_handle, -1, 0, sizeof(unsigned));
	starpu_data_move(var2_handle, var1_handle);

	ret = starpu_task_insert(&increment_cl, STARPU_RW, var2_handle, 0);
	STARPU_ASSERT(ret == 0);

	/* Check that it is correct */
	check_value("var2", var2_handle, 2);

	/* Free it through submit */
	starpu_data_unregister_submit(var2_handle);

	/* Make another duplicate of the original data */
	ret = starpu_task_insert(&neutral_cl, STARPU_W, var1_handle, 0);
	STARPU_ASSERT(ret == 0);
	ret = starpu_task_insert(&increment_cl, STARPU_RW, var1_handle, 0);
	STARPU_ASSERT(ret == 0);

	/* Move twice the original data */
	starpu_variable_data_register(&var2_handle, -1, 0, sizeof(unsigned));
	starpu_data_move(var2_handle, var1_handle);
	ret = starpu_task_insert(&increment_cl, STARPU_RW, var2_handle, 0);
	STARPU_ASSERT(ret == 0);
	check_value("var2", var2_handle, 2);

	starpu_variable_data_register(&var3_handle, -1, 0, sizeof(unsigned));
	starpu_data_move(var3_handle, var2_handle);
	ret = starpu_task_insert(&increment_cl, STARPU_RW, var3_handle, 0);
	STARPU_ASSERT(ret == 0);

	/* Check that it is correct */
	check_value("var3", var3_handle, 3);

	starpu_data_unregister(var1_handle);
	starpu_data_unregister(var2_handle);
	starpu_data_unregister(var3_handle);

	increment_unload_opencl();

	starpu_shutdown();

	STARPU_RETURN(ret);
}
