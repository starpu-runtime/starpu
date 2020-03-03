/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This is an example of using a callback. We submit a task, whose callback
 * submits another task (without any callback).
 */

#include <starpu.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

starpu_data_handle_t handle;

void cpu_codelet(void *descr[], void *_args)
{
	(void)_args;
	int *val = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);

	*val += 1;
}

struct starpu_codelet cl =
{
	.modes = { STARPU_RW },
	.cpu_funcs = {cpu_codelet},
	.cpu_funcs_name = {"cpu_codelet"},
	.nbuffers = 1,
	.name = "callback"
};

void callback_func(void *callback_arg)
{
	int ret;

	(void)callback_arg;

	struct starpu_task *task = starpu_task_create();
	task->cl = &cl;
	task->handles[0] = handle;

	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

int main(void)
{
	int v=40;
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&v, sizeof(int));

	struct starpu_task *task = starpu_task_create();
	task->cl = &cl;
	task->callback_func = callback_func;
	task->callback_arg = NULL;
	task->handles[0] = handle;

	ret = starpu_task_submit(task);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	starpu_task_wait_for_all();
	starpu_data_unregister(handle);

	FPRINTF(stderr, "v -> %d\n", v);

	starpu_shutdown();

	return (v == 42) ? 0 : 1;

enodev:
	starpu_shutdown();
	return 77;
}
