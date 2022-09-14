/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void func_unpack_args(void *descr[], void *_args)
{
	int factor;
	char c;
	int x;

	(void)descr;

	starpu_codelet_unpack_args(_args, &factor, &c, &x);

	FPRINTF(stderr, "[codelet unpack_args] values: %d %c %d\n", factor, c, x);
	assert(factor == 12 && c == 'n' && x == 42);
}

struct starpu_codelet mycodelet_unpack_args =
{
	.cpu_funcs = {func_unpack_args},
	.cpu_funcs_name = {"func_unpack_args"},
        .nbuffers = 0
};

void func_unpack_arg(void *descr[], void *_args)
{
	int factor;
	char c;
	int x;

	(void)descr;

	size_t size = sizeof(int) + 3*sizeof(size_t) + sizeof(int) + sizeof(char) + sizeof(int);
	struct starpu_codelet_pack_arg_data state;
	starpu_codelet_unpack_arg_init(&state, _args, size);
	starpu_codelet_unpack_arg(&state, (void**)&factor, sizeof(factor));
	starpu_codelet_unpack_arg(&state, (void**)&c, sizeof(c));
	starpu_codelet_unpack_arg(&state, (void**)&x, sizeof(x));
	starpu_codelet_unpack_arg_fini(&state);

	FPRINTF(stderr, "[codelet unpack_arg] values: %d %c %d\n", factor, c, x);
	assert(factor == 12 && c == 'n' && x == 42);
}

struct starpu_codelet mycodelet_unpack_arg =
{
	.cpu_funcs = {func_unpack_arg},
	.cpu_funcs_name = {"func_unpack_arg"},
        .nbuffers = 0
};

void func_dup_arg(void *descr[], void *_args)
{
	int *factor;
	char *c;
	int *x;
	size_t size;

	(void)descr;

	size_t psize = sizeof(int) + 3*sizeof(size_t) + sizeof(int) + sizeof(char) + sizeof(int);
	struct starpu_codelet_pack_arg_data state;
	starpu_codelet_unpack_arg_init(&state, _args, psize);
	starpu_codelet_dup_arg(&state, (void**)&factor, &size);
	assert(size == sizeof(*factor));
	starpu_codelet_dup_arg(&state, (void**)&c, &size);
	assert(size == sizeof(*c));
	starpu_codelet_dup_arg(&state, (void**)&x, &size);
	assert(size == sizeof(*x));
	starpu_codelet_unpack_arg_fini(&state);

	FPRINTF(stderr, "[codelet dup_arg] values: %d %c %d\n", *factor, *c, *x);
	assert(*factor == 12 && *c == 'n' && *x == 42);
	free(factor);
	free(c);
	free(x);
}

struct starpu_codelet mycodelet_dup_arg =
{
	.cpu_funcs = {func_dup_arg},
	.cpu_funcs_name = {"func_dup_arg"},
        .nbuffers = 0
};

void func_pick_arg(void *descr[], void *_args)
{
	int *factor;
	char *c;
	int *x;
	size_t size;

	(void)descr;

	size_t psize = sizeof(int) + 6*sizeof(size_t) + sizeof(int) + 4*sizeof(char) + sizeof(int);
	struct starpu_codelet_pack_arg_data state;
	starpu_codelet_unpack_arg_init(&state, _args, psize);
	starpu_codelet_pick_arg(&state, (void**)&factor, &size);
	assert(size == sizeof(*factor));
	starpu_codelet_pick_arg(&state, (void**)&c, &size);
	assert(size == sizeof(*c));
	starpu_codelet_pick_arg(&state, (void**)&c, &size);
	assert(size == sizeof(*c));
	starpu_codelet_pick_arg(&state, (void**)&c, &size);
	assert(size == sizeof(*c));
	starpu_codelet_pick_arg(&state, (void**)&c, &size);
	assert(size == sizeof(*c));
	starpu_codelet_pick_arg(&state, (void**)&x, &size);
	assert(size == sizeof(*x));
	starpu_codelet_unpack_arg_fini(&state);

	FPRINTF(stderr, "[codelet pick_arg] values: %d %c %d\n", *factor, *c, *x);
	assert(*factor == 12 && *c == 'n' && *x == 42);
}

struct starpu_codelet mycodelet_pick_arg =
{
	.cpu_funcs = {func_pick_arg},
	.cpu_funcs_name = {"func_pick_arg"},
        .nbuffers = 0
};

int main(void)
{
	int ret;
	int x=42;
	int factor=12;
	char c='n';

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	FPRINTF(stderr, "[init] values: %d %c %d\n", factor, c, x);

	{
		struct starpu_task *task = starpu_task_build(&mycodelet_unpack_args, STARPU_TASK_SYNCHRONOUS, 1, 0);
		task->cl_arg_free = 1;

		starpu_codelet_pack_args(&task->cl_arg, &task->cl_arg_size,
					 STARPU_VALUE, &factor, sizeof(factor),
					 STARPU_VALUE, &c, sizeof(c),
					 STARPU_VALUE, &x, sizeof(x),
					 0);
		ret = starpu_task_submit(task);
		if (ret != -ENODEV)
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/* Test with starpu_codelet_unpack_args */
 	{
		struct starpu_task *task = starpu_task_build(&mycodelet_unpack_args, STARPU_TASK_SYNCHRONOUS, 1, 0);
		task->cl_arg_free = 1;

		struct starpu_codelet_pack_arg_data state;
		starpu_codelet_pack_arg_init(&state);
		starpu_codelet_pack_arg(&state, &factor, sizeof(factor));
		starpu_codelet_pack_arg(&state, &c, sizeof(c));
		starpu_codelet_pack_arg(&state, &x, sizeof(x));
		starpu_codelet_pack_arg_fini(&state, &task->cl_arg, &task->cl_arg_size);

		ret = starpu_task_submit(task);
		if (ret != -ENODEV)
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/* Test with starpu_codelet_unpack_arg */
 	{
		struct starpu_task *task = starpu_task_build(&mycodelet_unpack_arg, STARPU_TASK_SYNCHRONOUS, 1, 0);
		task->cl_arg_free = 1;

		struct starpu_codelet_pack_arg_data state;
		starpu_codelet_pack_arg_init(&state);
		starpu_codelet_pack_arg(&state, &factor, sizeof(factor));
		starpu_codelet_pack_arg(&state, &c, sizeof(c));
		starpu_codelet_pack_arg(&state, &x, sizeof(x));
		starpu_codelet_pack_arg_fini(&state, &task->cl_arg, &task->cl_arg_size);

		ret = starpu_task_submit(task);
		if (ret != -ENODEV)
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/* Test with starpu_codelet_dup_arg */
 	{
		struct starpu_task *task = starpu_task_build(&mycodelet_dup_arg, STARPU_TASK_SYNCHRONOUS, 1, 0);
		task->cl_arg_free = 1;

		struct starpu_codelet_pack_arg_data state;
		starpu_codelet_pack_arg_init(&state);
		starpu_codelet_pack_arg(&state, &factor, sizeof(factor));
		starpu_codelet_pack_arg(&state, &c, sizeof(c));
		starpu_codelet_pack_arg(&state, &x, sizeof(x));
		starpu_codelet_pack_arg_fini(&state, &task->cl_arg, &task->cl_arg_size);

		ret = starpu_task_submit(task);
		if (ret != -ENODEV)
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/* Test with starpu_codelet_pick_arg */
 	{
		struct starpu_task *task = starpu_task_build(&mycodelet_pick_arg, STARPU_TASK_SYNCHRONOUS, 1, 0);
		task->cl_arg_free = 1;

		struct starpu_codelet_pack_arg_data state;
		starpu_codelet_pack_arg_init(&state);
		starpu_codelet_pack_arg(&state, &factor, sizeof(factor));
		starpu_codelet_pack_arg(&state, &c, sizeof(c));
		starpu_codelet_pack_arg(&state, &c, sizeof(c));
		starpu_codelet_pack_arg(&state, &c, sizeof(c));
		starpu_codelet_pack_arg(&state, &c, sizeof(c));
		starpu_codelet_pack_arg(&state, &x, sizeof(x));
		starpu_codelet_pack_arg_fini(&state, &task->cl_arg, &task->cl_arg_size);

		ret = starpu_task_submit(task);
		if (ret != -ENODEV)
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

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
