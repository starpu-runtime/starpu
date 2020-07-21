/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Test starpu_data_dup_ro
 */

int main(int argc, char **argv)
{
	int ret;
	int var1, *var2, *var3, *var4;
	starpu_data_handle_t var1_handle, var2_handle, var3_handle, var4_handle;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	var1 = 42;

	starpu_variable_data_register(&var1_handle, STARPU_MAIN_RAM, (uintptr_t)&var1, sizeof(var1));

	/* Make a duplicate of the original data */
	ret = starpu_data_dup_ro(&var2_handle, var1_handle, 1, NULL, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_dup_ro");

	/* Make a second duplicate of the original data */
	ret = starpu_data_dup_ro(&var3_handle, var1_handle, 1, NULL, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_dup_ro");

	/* Make a duplicate of a duplicate */
	ret = starpu_data_dup_ro(&var4_handle, var2_handle, 1, NULL, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_dup_ro");

	starpu_data_acquire(var2_handle, STARPU_R);
	var2 = starpu_data_get_local_ptr(var2_handle);
	ret = EXIT_SUCCESS;
	if (*var2 != var1)
	{
	     FPRINTF(stderr, "var2 is %d but it should be %d\n", *var2, var1);
	     ret = EXIT_FAILURE;
	}
	starpu_data_release(var2_handle);

	starpu_data_acquire(var3_handle, STARPU_R);
	var3 = starpu_data_get_local_ptr(var3_handle);
	ret = EXIT_SUCCESS;
	if (*var3 != var1)
	{
	     FPRINTF(stderr, "var3 is %d but it should be %d\n", *var3, var1);
	     ret = EXIT_FAILURE;
	}
	starpu_data_release(var3_handle);

	starpu_data_acquire(var4_handle, STARPU_R);
	var4 = starpu_data_get_local_ptr(var4_handle);
	ret = EXIT_SUCCESS;
	if (*var4 != var1)
	{
	     FPRINTF(stderr, "var4 is %d but it should be %d\n", *var4, var1);
	     ret = EXIT_FAILURE;
	}
	starpu_data_release(var4_handle);

	starpu_data_unregister(var1_handle);
	starpu_data_unregister(var2_handle);
	starpu_data_unregister(var3_handle);
	starpu_data_unregister(var4_handle);
	starpu_shutdown();

	STARPU_RETURN(ret);
}
