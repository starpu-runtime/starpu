/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Test passing a NULL codelet, but callbacks
 */

static
void callback(void *ptr)
{
     int *x = (int *)ptr;
     FPRINTF(stderr, "x=%d\n", *x);
     STARPU_ASSERT_MSG(*x == 40, "%d != %d\n", *x, 40);
     (*x)++;
}

static
void callback2(void *ptr)
{
     int *x2 = (int *)ptr;
     FPRINTF(stderr, "x2=%d\n", *x2);
     STARPU_ASSERT_MSG(*x2 == 41, "%d != %d\n", *x2, 41);
     (*x2)++;
}

static
void prologue_callback(void *ptr)
{
     int *y = (int *)ptr;
     FPRINTF(stderr, "y=%d\n", *y);
     STARPU_ASSERT_MSG(*y == 12, "%d != %d\n", *y, 12);
     (*y)++;
}

static
void prologue_callback_pop(void *ptr)
{
     int *z = (int *)ptr;
     FPRINTF(stderr, "z=%d\n", *z);
     STARPU_ASSERT_MSG(*z == 32, "%d != %d\n", *z, 32);
     (*z)++;
}

int main(int argc, char **argv)
{
	int ret;
	int x=40;
	int x2=41;
	int y=12;
	int z=32;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	ret = starpu_task_insert(NULL,
				 STARPU_CALLBACK_WITH_ARG_NFREE, callback, &x,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(NULL,
				 STARPU_CALLBACK, callback2,
				 STARPU_CALLBACK_ARG_NFREE, &x2,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(NULL,
				 STARPU_PROLOGUE_CALLBACK, prologue_callback,
				 STARPU_PROLOGUE_CALLBACK_ARG_NFREE, &y,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(NULL,
				 STARPU_PROLOGUE_CALLBACK_POP, prologue_callback_pop,
				 STARPU_PROLOGUE_CALLBACK_POP_ARG_NFREE, &z,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();

	STARPU_ASSERT_MSG(x == 41, "x should be equal to %d and not %d\n", 41, x);
	STARPU_ASSERT_MSG(x2 == 42, "x2 should be equal to %d and not %d\n", 42, x2);
	STARPU_ASSERT_MSG(y == 13, "y should be equal to %d and not %d\n", 13, y);
	STARPU_ASSERT_MSG(z == 33, "z should be equal to %d and not %d\n", 33, z);

	starpu_shutdown();

	return EXIT_SUCCESS;
}

