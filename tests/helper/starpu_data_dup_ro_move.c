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
#include "../helper.h"

/*
 * Test starpu_data_dup_ro_move
 */

static int ret;

static void check_value(const char *name, starpu_data_handle_t handle, unsigned expected)
{
	handle = starpu_data_dup_ro_get(handle);
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
	starpu_data_handle_t var1_handle, var2_handle, var3_handle, var4_handle;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	if (starpu_cpu_worker_get_count() + starpu_cuda_worker_get_count() + starpu_opencl_worker_get_count() == 0)
	{
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	starpu_variable_data_register(&var1_handle, -1, 0, sizeof(unsigned));

	/* Initialize the source through acquisition */
	starpu_data_acquire(var1_handle, STARPU_W);
	unsigned *var_orig = starpu_data_get_local_ptr(var1_handle);
	*var_orig = 42;
	starpu_data_release(var1_handle);

	/* Make a duplicate of the original data */
	ret = starpu_data_dup_ro(&var2_handle, var1_handle);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_dup_ro");

	/* Make a second duplicate of the original data */
	ret = starpu_data_dup_ro(&var3_handle, var1_handle);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_dup_ro");
	STARPU_ASSERT(var3_handle == var2_handle);

	/* Make a duplicate of a duplicate */
	ret = starpu_data_dup_ro(&var4_handle, var2_handle);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_dup_ro");
	STARPU_ASSERT(var4_handle == var2_handle);

	/* Check that they are correct */
	check_value("var2", var2_handle, 42);
	check_value("var3", var3_handle, 42);
	check_value("var4", var4_handle, 42);

	/* Detach var1 from the others, should not trigger an actual copy. */
	starpu_data_invalidate(var1_handle);

	/* Check that they are correct */
	check_value("var2", var2_handle, 42);
	check_value("var3", var3_handle, 42);
	check_value("var4", var4_handle, 42);

	/* Reinitialize the source through acquisition */
	starpu_data_acquire(var1_handle, STARPU_W);
	unsigned *var_new = starpu_data_get_local_ptr(var1_handle);
	/* It should be a different one */
	STARPU_ASSERT(var_new != var_orig);
	*var_new = 43;
	starpu_data_release(var1_handle);

	/* Check that everything remains correct */
	check_value("var1", var1_handle, 43);
	check_value("var2", var2_handle, 42);
	check_value("var3", var3_handle, 42);
	check_value("var4", var4_handle, 42);

	/* Check that var2 indeed got the var1 buffer */
	starpu_data_acquire(var2_handle, STARPU_R);
	STARPU_ASSERT(starpu_data_get_local_ptr(var2_handle) == var_orig);
	starpu_data_release(var2_handle);

	starpu_data_unregister(var1_handle);
	starpu_data_unregister(var2_handle);
	starpu_data_unregister(var3_handle);
	starpu_data_unregister(var4_handle);

	starpu_shutdown();

	STARPU_RETURN(ret);
}
