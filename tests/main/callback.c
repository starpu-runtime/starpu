/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void codelet_callback_func(void *arg)
{
	if (arg)
	{
		int *x = (int *)arg;
		FPRINTF(stderr, "calling callback codelet arg %d\n", *x);
	}
	else
		FPRINTF(stderr, "calling callback codelet arg %p\n", arg);
}

void task_callback_func(void *arg)
{
	FPRINTF(stderr, "\ncalling callback task arg %p\n", arg);
	if (starpu_task_get_current()->cl->callback_func)
		starpu_task_get_current()->cl->callback_func(arg);
}

struct starpu_codelet mycodelet =
{
	.where = STARPU_NOWHERE,
	.callback_func = codelet_callback_func
};

struct starpu_codelet mycodelet2 =
{
	.where = STARPU_NOWHERE,
};

int main(void)
{
        int ret;
	int value=12;
	int value2=24;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	ret = starpu_task_insert(&mycodelet,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&mycodelet,
				 STARPU_CALLBACK_ARG_NFREE, &value,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&mycodelet,
				 STARPU_CALLBACK, &task_callback_func,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&mycodelet,
				 STARPU_CALLBACK, &task_callback_func,
				 STARPU_CALLBACK_ARG_NFREE, &value2,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&mycodelet2,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&mycodelet2,
				 STARPU_CALLBACK, &task_callback_func,
				 STARPU_CALLBACK_ARG_NFREE, &value,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_shutdown();
	return 0;
}
