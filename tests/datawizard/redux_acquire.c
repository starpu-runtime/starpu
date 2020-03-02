/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <math.h>
#include "helper.h"

void init_cpu_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	long int *dot = (long int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*dot = 42;
}

void redux_cpu_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	long int *dota = (long int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	long int *dotb = (long int *)STARPU_VARIABLE_GET_PTR(descr[1]);

	*dota = *dota + *dotb;
}

static struct starpu_codelet init_codelet =
{
	.cpu_funcs = {init_cpu_func},
	.nbuffers = 1,
	.modes = {STARPU_W},
	.name = "init_codelet"
};

static struct starpu_codelet redux_codelet =
{
	.cpu_funcs = {redux_cpu_func},
	.modes = {STARPU_RW, STARPU_R},
	.nbuffers = 2,
	.name = "redux_codelet"
};

static void check_dot(void *dot_handle)
{
	long int *x = starpu_data_get_local_ptr(dot_handle);
	STARPU_ASSERT_MSG(*x == 42, "Incorrect value %ld", *x);
	starpu_data_release(dot_handle);
}

int main(void)
{
	starpu_data_handle_t dot_handle;

	int ret = starpu_init(NULL);
	if (ret == -ENODEV)
		goto skip;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_cpu_worker_get_count() == 0)
		goto enodev;

	starpu_variable_data_register(&dot_handle, -1, (uintptr_t)NULL, sizeof(long int));
	starpu_data_set_reduction_methods(dot_handle, &redux_codelet, &init_codelet);
	starpu_data_acquire(dot_handle, STARPU_R);
	long int *x = starpu_data_get_local_ptr(dot_handle);
	STARPU_ASSERT_MSG(*x == 42, "Incorrect value %ld", *x);
	starpu_data_release(dot_handle);
	starpu_data_unregister(dot_handle);

	starpu_variable_data_register(&dot_handle, -1, (uintptr_t)NULL, sizeof(long int));
	starpu_data_set_reduction_methods(dot_handle, &redux_codelet, &init_codelet);
	starpu_data_acquire_cb(dot_handle, STARPU_R, check_dot, dot_handle);
	starpu_data_unregister(dot_handle);

	starpu_shutdown();
	return 0;

enodev:
	starpu_shutdown();
skip:
	return STARPU_TEST_SKIPPED;
}
