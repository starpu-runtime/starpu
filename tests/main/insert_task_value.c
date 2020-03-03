/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

	starpu_codelet_unpack_args(_args, ifactor, 0);
	starpu_codelet_unpack_args(_args, ifactor, &ffactor);

	FPRINTF(stderr, "[func_cpu_int_float_multiple_unpack] Values %d - %3.2f\n", ifactor[0], ffactor);
	assert(ifactor[0] == IFACTOR && ffactor == FFACTOR);
}

void func_cpu_int_float_unpack_copyleft(void *descr[], void *_args)
{
	int ifactor[2048];
	float ffactor;
	void *buffer;
	size_t buffer_size;
	(void) descr;

	buffer_size = sizeof(int)+sizeof(float)+sizeof(size_t);
	buffer = calloc(buffer_size, 1);
	starpu_codelet_unpack_args_and_copyleft(_args, buffer, buffer_size, ifactor, 0);
	starpu_codelet_unpack_args(buffer, &ffactor);

	FPRINTF(stderr, "[func_cpu_int_float_unpack_copyleft] Values %d - %3.2f\n", ifactor[0], ffactor);
	assert(ifactor[0] == IFACTOR && ffactor == FFACTOR);
	free(buffer);
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

	starpu_codelet_unpack_args(_args, &ffactor, 0);
	starpu_codelet_unpack_args(_args, &ffactor, ifactor);

	FPRINTF(stderr, "[func_cpu_float_int_multiple_unpack] Values %d - %3.2f\n", ifactor[0], ffactor);
	assert(ifactor[0] == IFACTOR && ffactor == FFACTOR);
}

void func_cpu_float_int_unpack_copyleft(void *descr[], void *_args)
{
	int ifactor[2048];
	float ffactor;
	void *buffer;
	size_t buffer_size;
	(void) descr;

	buffer_size = sizeof(int)+2048*sizeof(int)+sizeof(size_t);
	buffer = calloc(buffer_size, 1);
	starpu_codelet_unpack_args_and_copyleft(_args, buffer, buffer_size, &ffactor, 0);
	starpu_codelet_unpack_args(buffer, ifactor);

	FPRINTF(stderr, "[func_cpu_float_int_multiple_unpack] Values %d - %3.2f\n", ifactor[0], ffactor);
	assert(ifactor[0] == IFACTOR && ffactor == FFACTOR);
	free(buffer);
}

void do_test_int_float_task_insert(starpu_cpu_func_t func, char* func_name)
{
	int *ifactor;
	float ffactor=FFACTOR;
	int ret;
	struct starpu_codelet codelet;

	FPRINTF(stderr, "\nTesting %s\n", __func__);

	starpu_codelet_init(&codelet);
	codelet.cpu_funcs[0] = func;
	codelet.cpu_funcs_name[0] = func_name;

	ifactor = calloc(2048, sizeof(int));
	ifactor[0] = IFACTOR;

	ret = starpu_task_insert(&codelet,
				 STARPU_VALUE, ifactor, 2048*sizeof(ifactor[0]),
				 STARPU_VALUE, &ffactor, sizeof(ffactor),
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	starpu_task_wait_for_all();
	free(ifactor);
}

void do_test_int_float_task_insert_pack(starpu_cpu_func_t func, char* func_name)
{
	int *ifactor;
	float ffactor=FFACTOR;
	int ret;
	struct starpu_codelet codelet;
	void *cl_arg = NULL;
	size_t cl_arg_size = 0;

	FPRINTF(stderr, "\nTesting %s\n", __func__);

	ifactor = calloc(2048, sizeof(int));
	ifactor[0] = IFACTOR;

	starpu_codelet_pack_args(&cl_arg, &cl_arg_size,
				 STARPU_VALUE, ifactor, 2048*sizeof(ifactor[0]),
				 STARPU_VALUE, &ffactor, sizeof(ffactor),
				 0);

	starpu_codelet_init(&codelet);
	codelet.cpu_funcs[0] = func;
	codelet.cpu_funcs_name[0] = func_name;

	ret = starpu_task_insert(&codelet,
				 STARPU_CL_ARGS, cl_arg, cl_arg_size,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	starpu_task_wait_for_all();
	free(ifactor);
}

void do_test_float_int_task_insert(starpu_cpu_func_t func, char* func_name)
{
	int *ifactor;
	float ffactor=FFACTOR;
	int ret;
	struct starpu_codelet codelet;

	FPRINTF(stderr, "\nTesting %s\n", __func__);

	starpu_codelet_init(&codelet);
	codelet.cpu_funcs[0] = func;
	codelet.cpu_funcs_name[0] = func_name;

	ifactor = calloc(2048, sizeof(int));
	ifactor[0] = IFACTOR;

	ret = starpu_task_insert(&codelet,
				 STARPU_VALUE, &ffactor, sizeof(ffactor),
				 STARPU_VALUE, ifactor, 2048*sizeof(ifactor[0]),
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	starpu_task_wait_for_all();
	free(ifactor);
}

void do_test_float_int_task_insert_pack(starpu_cpu_func_t func, char* func_name)
{
	int *ifactor;
	float ffactor=FFACTOR;
	int ret;
	struct starpu_codelet codelet;
	void *cl_arg = NULL;
	size_t cl_arg_size = 0;

	FPRINTF(stderr, "\nTesting %s\n", __func__);

	ifactor = calloc(2048, sizeof(int));
	ifactor[0] = IFACTOR;

	starpu_codelet_pack_args(&cl_arg, &cl_arg_size,
				 STARPU_VALUE, &ffactor, sizeof(ffactor),
				 STARPU_VALUE, ifactor, 2048*sizeof(ifactor[0]),
				 0);

	starpu_codelet_init(&codelet);
	codelet.cpu_funcs[0] = func;
	codelet.cpu_funcs_name[0] = func_name;

	ret = starpu_task_insert(&codelet,
				 STARPU_CL_ARGS, cl_arg, cl_arg_size,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	starpu_task_wait_for_all();
	free(ifactor);
}

void do_test_int_float_pack(starpu_cpu_func_t func, char* func_name)
{
	struct starpu_task *task;
	struct starpu_codelet codelet;
	int ret;
	int *ifactor;
	float ffactor=FFACTOR;

	FPRINTF(stderr, "\nTesting %s\n", __func__);

	ifactor = calloc(2048, sizeof(int));
	ifactor[0] = IFACTOR;

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
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	starpu_task_wait_for_all();
	free(ifactor);
}

void do_test_float_int_pack(starpu_cpu_func_t func, char* func_name)
{
	struct starpu_task *task;
	struct starpu_codelet codelet;
	int ret;
	int *ifactor;
	float ffactor=FFACTOR;

	FPRINTF(stderr, "\nTesting %s\n", __func__);

	ifactor = calloc(2048, sizeof(int));
	ifactor[0] = IFACTOR;

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
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	starpu_task_wait_for_all();

	free(ifactor);
}

int main(void)
{
        int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	if (starpu_worker_get_count_by_type(STARPU_CPU_WORKER) == 0)
		goto enodev;

	do_test_int_float_task_insert(func_cpu_int_float, "func_cpu_int_float");
	do_test_int_float_task_insert(func_cpu_int_float_multiple_unpack, "func_cpu_int_float_multiple_unpack");
	do_test_int_float_task_insert(func_cpu_int_float_unpack_copyleft, "func_cpu_int_float_unpack_copyleft");

	do_test_int_float_task_insert_pack(func_cpu_int_float, "func_cpu_int_float");
	do_test_int_float_task_insert_pack(func_cpu_int_float_multiple_unpack, "func_cpu_int_float_multiple_unpack");
	do_test_int_float_task_insert_pack(func_cpu_int_float_unpack_copyleft, "func_cpu_int_float_unpack_copyleft");

	do_test_float_int_task_insert(func_cpu_float_int, "func_cpu_float_int");
	do_test_float_int_task_insert(func_cpu_float_int_multiple_unpack, "func_cpu_float_int_multiple_unpack");
	do_test_float_int_task_insert(func_cpu_float_int_unpack_copyleft, "func_cpu_float_int_unpack_copyleft");

	do_test_float_int_task_insert_pack(func_cpu_float_int, "func_cpu_float_int");
	do_test_float_int_task_insert_pack(func_cpu_float_int_multiple_unpack, "func_cpu_float_int_multiple_unpack");
	do_test_float_int_task_insert_pack(func_cpu_float_int_unpack_copyleft, "func_cpu_float_int_unpack_copyleft");

	do_test_int_float_pack(func_cpu_int_float, "func_cpu_int_float");
	do_test_int_float_pack(func_cpu_int_float_multiple_unpack, "func_cpu_int_float_multiple_unpack");
	do_test_int_float_pack(func_cpu_int_float_unpack_copyleft, "func_cpu_int_float_unpack_copyleft");

	do_test_float_int_pack(func_cpu_float_int, "func_cpu_float_int");
	do_test_float_int_pack(func_cpu_float_int_multiple_unpack, "func_cpu_float_int_multiple_unpack");
	do_test_float_int_pack(func_cpu_float_int_unpack_copyleft, "func_cpu_float_int_unpack_copyleft");

	starpu_shutdown();

	return 0;

enodev:
	starpu_shutdown();
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return STARPU_TEST_SKIPPED;
}
