/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void redux_with_args_cpu(void *descr[], void *arg)
{
	int *value = (int *)arg;
	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *src = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);
	*dst = *dst + *src + *value;
}

struct starpu_codelet redux_with_args_cl =
{
	.modes = {STARPU_RW|STARPU_COMMUTE, STARPU_R},
	.nbuffers = 2,
	.cpu_funcs = {redux_with_args_cpu},
};

int main(int argc, char **argv)
{
	int ret;
	unsigned var = 0;
	starpu_data_handle_t handle;
	unsigned value = 42;

	/* Not supported yet */
	if (starpu_getenv_number_default("STARPU_GLOBAL_ARBITER", 0) > 0)
		return STARPU_TEST_SKIPPED;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "we need 1 cpu worker\n");
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	increment_load_opencl();

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&var, sizeof(unsigned));
	starpu_data_set_reduction_methods_with_args(handle, &redux_with_args_cl, &value, &neutral_cl, NULL);
	ret = starpu_task_insert(&increment_redux_cl, STARPU_REDUX, handle, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	starpu_data_unregister(handle);

	if (var != value+1)
	{
		FPRINTF(stderr, "Value %u != Expected value %u\n", var, value+1);
		goto err;
	}

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
