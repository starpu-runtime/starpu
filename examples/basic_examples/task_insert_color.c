/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void func(void *descr[], void *_args)
{
	int *x = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	(void)_args;

	*x *= 2;
}

struct starpu_codelet mycodelet =
{
	.modes = { STARPU_RW },
	.cpu_funcs = {func},
	.cpu_funcs_name = {"func"},
        .nbuffers = 1
};

struct starpu_codelet mycodelet_color =
{
	.modes = { STARPU_RW },
	.cpu_funcs = {func},
	.cpu_funcs_name = {"func"},
        .nbuffers = 1,
	.color = 0x0000FF,
};

int main(void)
{
	unsigned i;
	int value=42;
	starpu_data_handle_t handle;
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));

	// In the trace file, the following task should be green (executed on CPU)
	ret = starpu_task_insert(&mycodelet, STARPU_RW, handle, STARPU_NAME, "mytask",
				 0);
	if (STARPU_UNLIKELY(ret == -ENODEV))
	{
		starpu_data_unregister(handle);
		goto enodev;
	}
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	// In the trace file, the following task will be red as specified by STARPU_TASK_COLOR
	ret = starpu_task_insert(&mycodelet, STARPU_RW, handle, STARPU_NAME, "mytask",
				 STARPU_TASK_COLOR, 0xFF0000,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	// In the trace file, the following task will be blue as specified by the field color of mycodelet_color
	ret = starpu_task_insert(&mycodelet_color, STARPU_RW, handle, STARPU_NAME, "mytask",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();
	starpu_data_unregister(handle);

	starpu_shutdown();

	return 0;

 enodev:
	return 77;
}
