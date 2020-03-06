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

#include <config.h>
#include <starpu.h>
#include "../helper.h"

#define IFACTOR 42
#define FFACTOR 12.0

void func_cpu(void *descr[], void *_args)
{
	int ifactor[2048];
	float ffactor;

	starpu_codelet_unpack_args(_args, ifactor, &ffactor);

	FPRINTF(stderr, "Values %d - %3.2f\n", ifactor[0], ffactor);
	assert(ifactor[0] == IFACTOR && ffactor == FFACTOR);
}

void func_cpu2(void *descr[], void *_args)
{
	int ifactor[2048];
	float ffactor;

	starpu_codelet_unpack_args(_args, &ffactor, ifactor);

	FPRINTF(stderr, "Values %d - %3.2f\n", ifactor[0], ffactor);
	assert(ifactor[0] == IFACTOR && ffactor == FFACTOR);
}

struct starpu_codelet mycodelet =
{
	.cpu_funcs = {func_cpu},
};

struct starpu_codelet mycodelet2 =
{
	.cpu_funcs = {func_cpu2},
};

int main(int argc, char **argv)
{
        int ret;
	int ifactor[2048];
	float ffactor=FFACTOR;
	struct starpu_task *task;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	ifactor[0] = IFACTOR;

	ret = starpu_insert_task(&mycodelet,
				 STARPU_VALUE, ifactor, 2048*sizeof(ifactor[0]),
				 STARPU_VALUE, &ffactor, sizeof(ffactor),
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_insert_task(&mycodelet2,
				 STARPU_VALUE, &ffactor, sizeof(ffactor),
				 STARPU_VALUE, ifactor, 2048*sizeof(ifactor[0]),
				 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	task = starpu_task_create();
	task->synchronous = 1;
	task->cl = &mycodelet;
	task->cl_arg_free = 1;
	starpu_codelet_pack_args(&task->cl_arg, &task->cl_arg_size,
				 STARPU_VALUE, ifactor, 2048*sizeof(ifactor[0]),
				 STARPU_VALUE, &ffactor, sizeof(ffactor),
				 0);
	ret = starpu_task_submit(task);
	if (ret != -ENODEV)
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	task = starpu_task_create();
	task->synchronous = 1;
	task->cl = &mycodelet2;
	task->cl_arg_free = 1;
	starpu_codelet_pack_args(&task->cl_arg, &task->cl_arg_size,
				 STARPU_VALUE, &ffactor, sizeof(ffactor),
				 STARPU_VALUE, ifactor, 2048*sizeof(ifactor[0]),
				 0);
	ret = starpu_task_submit(task);
	if (ret != -ENODEV)
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	starpu_shutdown();

	return 0;

enodev:
	starpu_shutdown();
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return STARPU_TEST_SKIPPED;
}
