/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Test starpu_codelet_pack_args and starpu_codelet_unpack_args
 */

void func_cpu(void *descr[], void *_args)
{
	int factor;
	char c;
	int x;

	(void)descr;

	starpu_codelet_unpack_args(_args, &factor, &c, &x);

        FPRINTF(stderr, "[codelet] values: %d %c %d\n", factor, c, x);
	assert(factor == 12 && c == 'n' && x == 42);
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
	.cpu_funcs_name = {"func_cpu"},
        .nbuffers = 0
};

int main(void)
{
        int ret;
        int x=42;
	int factor=12;
	char c='n';
	struct starpu_task *task, *task2;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

        FPRINTF(stderr, "[init] values: %d %c %d\n", factor, c, x);

	task = starpu_task_create();
	task->synchronous = 1;
	task->cl = &mycodelet;
	task->cl_arg_free = 1;
	starpu_codelet_pack_args(&task->cl_arg, &task->cl_arg_size,
				 STARPU_VALUE, &factor, sizeof(factor),
				 STARPU_VALUE, &c, sizeof(c),
				 STARPU_VALUE, &x, sizeof(x),
				 0);
	ret = starpu_task_submit(task);
	if (ret != -ENODEV)
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	task2 = starpu_task_create();
	task2->synchronous = 1;
	task2->cl = &mycodelet;
	task2->cl_arg_free = 1;

	{
		struct starpu_codelet_pack_arg_data state;
		starpu_codelet_pack_arg_init(&state);
		starpu_codelet_pack_arg(&state, &factor, sizeof(factor));
		starpu_codelet_pack_arg(&state, &c, sizeof(c));
		starpu_codelet_pack_arg(&state, &x, sizeof(x));
		starpu_codelet_pack_arg_fini(&state, &task2->cl_arg, &task2->cl_arg_size);
	}

	ret = starpu_task_submit(task2);
	if (ret != -ENODEV)
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	starpu_shutdown();
	if (ret == -ENODEV)
	{
		fprintf(stderr, "WARNING: No one can execute this task\n");
		/* yes, we do not perform the computation but we did detect that no one
		 * could perform the kernel, so this is not an error from StarPU */
		return STARPU_TEST_SKIPPED;
	}
	else
		return 0;
}
