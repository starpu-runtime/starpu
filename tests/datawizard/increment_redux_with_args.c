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

void init_with_args_cpu(void *descr[], void *arg)
{
	int *value = (int *)arg;
	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*dst = *value;
}

struct starpu_codelet init_with_args_cl =
{
	.modes = { STARPU_W },
	.nbuffers = 1,
	.cpu_funcs = {init_with_args_cpu},
};

void redux_with_args_cpu(void *descr[], void *arg)
{
	int value_1;
	int value_2;
	starpu_codelet_unpack_args(arg, &value_1, &value_2);
	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *src = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);
	*dst = *dst + *src + value_1 + value_2;
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
	unsigned init_value = 42;
	int redux_value_1 = 5;
	int redux_value_2 = 7;
	void* redux_cl_args;
	size_t redux_cl_args_size;
	struct starpu_conf conf;

	/* Not supported yet */
	if (starpu_getenv_number_default("STARPU_GLOBAL_ARBITER", 0) > 0)
		return STARPU_TEST_SKIPPED;

	/* This test uses a redux codelet with a value on the heap, hence it cannot work with server client */
	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.nmpi_sc = -1;
	conf.ntcpip_sc = -1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "we need 1 cpu worker\n");
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	starpu_codelet_pack_args(&redux_cl_args, &redux_cl_args_size,
				 STARPU_VALUE, &redux_value_1, sizeof(redux_value_1),
				 STARPU_VALUE, &redux_value_2, sizeof(redux_value_2),
				 0);

	increment_load_opencl();

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&var, sizeof(unsigned));
	starpu_data_set_reduction_methods_with_args(handle, &redux_with_args_cl, redux_cl_args, &init_with_args_cl, &init_value);
	ret = starpu_task_insert(&increment_redux_cl, STARPU_REDUX, handle, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	starpu_data_unregister(handle);

	int expected_value = init_value + redux_value_1 + redux_value_2 + 1;
	if (var != expected_value)
	{
		FPRINTF(stderr, "Value %u != Expected value %u\n", var, expected_value);
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
