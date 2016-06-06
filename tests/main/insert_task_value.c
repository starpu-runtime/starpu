/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015, 2016  CNRS
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

#include <config.h>
#include <starpu.h>
#include "../helper.h"

/*
 * Test passing values to tasks in different ways
 */

#define IFACTOR 42
#define FFACTOR 12.00

void func_cpu_int_float(void *descr[], void *_args)
{
	int ifactor[2048];
	float ffactor;
	(void) descr;

	starpu_codelet_unpack_args(_args, ifactor, &ffactor);

	FPRINTF(stderr, "[func_cpu_int_float                ] Values %d - %3.2f\n", ifactor[0], ffactor);
	assert(ifactor[0] == IFACTOR && ffactor == FFACTOR);
}

void func_cpu_int_float_multiple_unpack(void *descr[], void *_args)
{
	int ifactor[2048];
	float ffactor;
	(void) descr;

	starpu_codelet_unpack_args(_args, ifactor, NULL);
	starpu_codelet_unpack_args(_args, ifactor, &ffactor);

	FPRINTF(stderr, "[func_cpu_int_float_multiple_unpack] Values %d - %3.2f\n", ifactor[0], ffactor);
	assert(ifactor[0] == IFACTOR && ffactor == FFACTOR);
}

void func_cpu_int_float_unpack_copyleft(void *descr[], void *_args)
{
	int ifactor[2048];
	float ffactor;
	char buffer[1024];
	(void) descr;

	starpu_codelet_unpack_args_and_copyleft(_args, buffer, 1024, ifactor, NULL);
	starpu_codelet_unpack_args(buffer, &ffactor);

	FPRINTF(stderr, "[func_cpu_int_float_unpack_copyleft] Values %d - %3.2f\n", ifactor[0], ffactor);
	assert(ifactor[0] == IFACTOR && ffactor == FFACTOR);
}

void func_cpu_float_int(void *descr[], void *_args)
{
	int ifactor[2048];
	float ffactor;
	(void) descr;

	starpu_codelet_unpack_args(_args, &ffactor, ifactor);

	FPRINTF(stderr, "[func_cpu_float_int                ] Values %d - %3.2f\n", ifactor[0], ffactor);
	assert(ifactor[0] == IFACTOR && ffactor == FFACTOR);
}

void func_cpu_float_int_multiple_unpack(void *descr[], void *_args)
{
	int ifactor[2048];
	float ffactor;
	(void) descr;

	starpu_codelet_unpack_args(_args, &ffactor, NULL);
	starpu_codelet_unpack_args(_args, &ffactor, ifactor);

	FPRINTF(stderr, "[func_cpu_float_int_multiple_unpack] Values %d - %3.2f\n", ifactor[0], ffactor);
	assert(ifactor[0] == IFACTOR && ffactor == FFACTOR);
}

void func_cpu_float_int_unpack_copyleft(void *descr[], void *_args)
{
	int ifactor[2048];
	float ffactor;
	char buffer[10240];
	(void) descr;

	starpu_codelet_unpack_args_and_copyleft(_args, buffer, 1024, &ffactor, NULL);
	starpu_codelet_unpack_args(buffer, ifactor);

	FPRINTF(stderr, "[func_cpu_float_int_multiple_unpack] Values %d - %3.2f\n", ifactor[0], ffactor);
	assert(ifactor[0] == IFACTOR && ffactor == FFACTOR);
}

int do_test_int_float_task_insert(starpu_cpu_func_t func, char* func_name)
{
	int ifactor[2048];
	float ffactor=FFACTOR;
	int ret;
	struct starpu_codelet codelet;

	starpu_codelet_init(&codelet);
	codelet.cpu_funcs[0] = func;
	codelet.cpu_funcs_name[0] = func_name;

	ifactor[0] = IFACTOR;

	ret = starpu_task_insert(&codelet,
				 STARPU_VALUE, ifactor, 2048*sizeof(ifactor[0]),
				 STARPU_VALUE, &ffactor, sizeof(ffactor),
				 0);
	if (ret == -ENODEV) return ret;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	starpu_task_wait_for_all();
	return 0;
}

int do_test_float_int_task_insert(starpu_cpu_func_t func, char* func_name)
{
	int ifactor[2048];
	float ffactor=FFACTOR;
	int ret;
	struct starpu_codelet codelet;

	starpu_codelet_init(&codelet);
	codelet.cpu_funcs[0] = func;
	codelet.cpu_funcs_name[0] = func_name;

	ifactor[0] = IFACTOR;

	ret = starpu_task_insert(&codelet,
				 STARPU_VALUE, &ffactor, sizeof(ffactor),
				 STARPU_VALUE, ifactor, 2048*sizeof(ifactor[0]),
				 0);
	if (ret == -ENODEV) return ret;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	starpu_task_wait_for_all();
	return 0;
}

int do_test_int_float_pack(starpu_cpu_func_t func, char* func_name)
{
	struct starpu_task *task;
	struct starpu_codelet codelet;
	int ret;
	int ifactor[2048];
	float ffactor=FFACTOR;

	starpu_codelet_init(&codelet);
	codelet.cpu_funcs[0] = func;
	codelet.cpu_funcs_name[0] = func_name;

	task = starpu_task_create();
	task->synchronous = 1;
	task->cl = &codelet;
	task->cl_arg_free = 1;
	starpu_codelet_pack_args(&task->cl_arg, &task->cl_arg_size,
				 STARPU_VALUE, ifactor, 2048*sizeof(ifactor[0]),
				 STARPU_VALUE, &ffactor, sizeof(ffactor),
				 0);
	ret = starpu_task_submit(task);
	if (ret == -ENODEV) return ret;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	starpu_task_wait_for_all();
	return 0;
}

int do_test_float_int_pack(starpu_cpu_func_t func, char* func_name)
{
	struct starpu_task *task;
	struct starpu_codelet codelet;
	int ret;
	int ifactor[2048];
	float ffactor=FFACTOR;

	starpu_codelet_init(&codelet);
	codelet.cpu_funcs[0] = func;
	codelet.cpu_funcs_name[0] = func_name;

	task = starpu_task_create();
	task->synchronous = 1;
	task->cl = &codelet;
	task->cl_arg_free = 1;
	starpu_codelet_pack_args(&task->cl_arg, &task->cl_arg_size,
				 STARPU_VALUE, &ffactor, sizeof(ffactor),
				 STARPU_VALUE, ifactor, 2048*sizeof(ifactor[0]),
				 0);
	ret = starpu_task_submit(task);
	if (ret == -ENODEV) return ret;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	starpu_task_wait_for_all();
	return 0;
}

int main(int argc, char **argv)
{
        int ret;
	int ifactor[2048];
	float ffactor=FFACTOR;
	struct starpu_task *task;
	(void) argc;
	(void) argv;

	ifactor[0] = IFACTOR;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	ret = do_test_int_float_task_insert(func_cpu_int_float, "func_cpu_int_float_name");
	if (ret == -ENODEV) goto enodev;
	ret = do_test_int_float_task_insert(func_cpu_int_float_multiple_unpack, "func_cpu_int_float_multiple_unpack");
	if (ret == -ENODEV) goto enodev;
	ret = do_test_int_float_task_insert(func_cpu_int_float_unpack_copyleft, "func_cpu_int_float_unpack_copyleft");
	if (ret == -ENODEV) goto enodev;

	ret = do_test_float_int_task_insert(func_cpu_float_int, "func_cpu_float_int");
	if (ret == -ENODEV) goto enodev;
	ret = do_test_float_int_task_insert(func_cpu_float_int_multiple_unpack, "func_cpu_float_int_multiple_unpack");
	if (ret == -ENODEV) goto enodev;
	ret = do_test_float_int_task_insert(func_cpu_float_int_unpack_copyleft, "func_cpu_float_int_unpack_copyleft");
	if (ret == -ENODEV) goto enodev;

	ret = do_test_int_float_pack(func_cpu_int_float, "func_cpu_int_float_name");
	if (ret == -ENODEV) goto enodev;
	ret = do_test_int_float_pack(func_cpu_int_float_multiple_unpack, "func_cpu_int_float_multiple_unpack");
	if (ret == -ENODEV) goto enodev;
	ret = do_test_int_float_pack(func_cpu_int_float_unpack_copyleft, "func_cpu_int_float_unpack_copyleft");
	if (ret == -ENODEV) goto enodev;

	ret = do_test_float_int_pack(func_cpu_float_int, "func_cpu_float_int");
	if (ret == -ENODEV) goto enodev;
	ret = do_test_float_int_pack(func_cpu_float_int_multiple_unpack, "func_cpu_float_int_multiple_unpack");
	if (ret == -ENODEV) goto enodev;
	ret = do_test_float_int_pack(func_cpu_float_int_unpack_copyleft, "func_cpu_float_int_unpack_copyleft");
	if (ret == -ENODEV) goto enodev;

	starpu_shutdown();

	return 0;

enodev:
	starpu_shutdown();
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return STARPU_TEST_SKIPPED;
}
