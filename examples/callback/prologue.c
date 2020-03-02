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
 * This is an example of using a prologue callback. We submit a task, whose
 * prologue callback (i.e. before task gets scheduled) prints a value, and
 * whose pop_prologue callback (i.e. after task gets scheduled, but before task
 * execution) prints another value.
 */

#include <starpu.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

starpu_data_handle_t handle;

void cpu_codelet(void *descr[], void *_args)
{
	(void)_args;
	int *val = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);

	*val += 1;
	printf("task executing \n");
}

struct starpu_codelet cl =
{
	.modes = { STARPU_RW },
	.cpu_funcs = {cpu_codelet},
	.cpu_funcs_name = {"cpu_codelet"},
	.nbuffers = 1,
	.name = "callback"
};

void prologue_callback_func(void *callback_arg)
{
	double *x = (double*)callback_arg;
	printf("x = %lf\n", *x);
	STARPU_ASSERT(*x == -999.0);
}

void pop_prologue_callback_func(void *args)
{
	unsigned val = (uintptr_t) args;
	printf("pop_prologue_callback val %u \n", val);
	STARPU_ASSERT(val == 5);
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
	double x = -999.0;

	struct starpu_task *task = starpu_task_create();
	task->cl = &cl;
	task->prologue_callback_func = prologue_callback_func;
	task->prologue_callback_arg = &x;

	task->prologue_callback_pop_func = pop_prologue_callback_func;
	task->prologue_callback_pop_arg = (void*) 5;

	task->handles[0] = handle;

	ret = starpu_task_submit(task);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_task_insert(&cl,
				 STARPU_RW, handle,
				 STARPU_PROLOGUE_CALLBACK, prologue_callback_func,
				 STARPU_PROLOGUE_CALLBACK_ARG_NFREE, &x,
				 STARPU_PROLOGUE_CALLBACK_POP, pop_prologue_callback_func,
				 STARPU_PROLOGUE_CALLBACK_POP_ARG_NFREE, 5,
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();

enodev:
	starpu_data_unregister(handle);
	FPRINTF(stderr, "v -> %d\n", v);
	starpu_shutdown();
	return (ret == -ENODEV) ? 77 : 0;
}
