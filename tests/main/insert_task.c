/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Try the starpu_task_insert interface in various ways
 */

static int _ifactor = 12;
static float _ffactor = 10.0;

void func_cpu_args(void *descr[], void *_args)
{
	int *x0 = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	float *x1 = (float *)STARPU_VARIABLE_GET_PTR(descr[1]);
	int ifactor;
	float ffactor;

	starpu_codelet_unpack_args(_args, &ifactor, &ffactor);

        *x0 = *x0 * ifactor;
        *x1 = *x1 * ffactor;
}

void func_cpu_noargs(void *descr[], void *_args)
{
	int *x0 = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	float *x1 = (float *)STARPU_VARIABLE_GET_PTR(descr[1]);

	(void)_args;

	*x0 = *x0 * _ifactor;
        *x1 = *x1 * _ffactor;
}

struct starpu_codelet mycodelet_args =
{
	.modes = { STARPU_RW, STARPU_RW },
	.cpu_funcs = {func_cpu_args},
	.cpu_funcs_name = {"func_cpu_args"},
        .nbuffers = 2
};

struct starpu_codelet mycodelet_noargs =
{
	.modes = { STARPU_RW, STARPU_RW },
	.cpu_funcs = {func_cpu_noargs},
        .nbuffers = 2
};

static
int test_codelet(struct starpu_codelet *codelet, int task_insert, int args, int x, float f)
{
        starpu_data_handle_t data_handles[2];
	int xx = x;
	float ff = f;
	int i, ret;

	starpu_variable_data_register(&data_handles[0], STARPU_MAIN_RAM, (uintptr_t)&xx, sizeof(xx));
	starpu_variable_data_register(&data_handles[1], STARPU_MAIN_RAM, (uintptr_t)&ff, sizeof(ff));

        FPRINTF(stderr, "values: %d (%d) %f (%f)\n", xx, _ifactor, ff, _ffactor);

	if (task_insert)
	{
		if (args)
			ret = starpu_task_insert(codelet,
						 STARPU_VALUE, &_ifactor, sizeof(_ifactor),
						 STARPU_VALUE, &_ffactor, sizeof(_ffactor),
						 STARPU_RW, data_handles[0], STARPU_RW, data_handles[1],
						 0);
		else
			ret = starpu_task_insert(codelet,
						 STARPU_RW, data_handles[0], STARPU_RW, data_handles[1],
						 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
	else
	{
		struct starpu_task *task;
		if (args)
			task = starpu_task_build(codelet,
						 STARPU_VALUE, &_ifactor, sizeof(_ifactor),
						 STARPU_VALUE, &_ffactor, sizeof(_ffactor),
						 STARPU_RW, data_handles[0], STARPU_RW, data_handles[1],
						 0);
		else
			task = starpu_task_build(codelet,
						 STARPU_RW, data_handles[0], STARPU_RW, data_handles[1],
						 0);
		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

enodev:
        for(i=0 ; i<2 ; i++)
	{
                starpu_data_unregister(data_handles[i]);
        }

        FPRINTF(stderr, "values: %d (should be %d) %f (should be %f)\n\n", xx, x*_ifactor, ff, f*_ffactor);
	return ret == -ENODEV ? ret : xx == x*_ifactor && ff == f*_ffactor;
}

int main(void)
{
        int x; float f;
        int i, ret;
	int ifactor=12;
	float ffactor=10.0;
        starpu_data_handle_t data_handles[2];

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	FPRINTF(stderr, "Testing codelet with task_insert and with arguments\n");
	ret = test_codelet(&mycodelet_args, 1, 1, 4, 2.0);
	if (ret == -ENODEV) goto enodev;
	if (ret)
	{
		FPRINTF(stderr, "Testing codelet with task_insert and without arguments\n");
		ret = test_codelet(&mycodelet_noargs, 1, 0, 9, 7.0);
	}
	if (ret == -ENODEV) goto enodev;
	if (ret)
	{
		FPRINTF(stderr, "Testing codelet with task_build and with arguments\n");
		ret = test_codelet(&mycodelet_args, 0, 1, 5, 3.0);
	}
	if (ret == -ENODEV) goto enodev;
	if (ret)
	{
		FPRINTF(stderr, "Testing codelet with task_build and without arguments\n");
		ret = test_codelet(&mycodelet_noargs, 0, 0, 7, 5.0);
	}
	if (ret == -ENODEV) goto enodev;

	starpu_shutdown();

	STARPU_RETURN(ret?0:1);

enodev:
	starpu_shutdown();
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return STARPU_TEST_SKIPPED;
}
