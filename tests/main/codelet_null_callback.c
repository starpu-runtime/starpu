/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013, 2014  Centre National de la Recherche Scientifique
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

static
int expected_x=40;
static
int expected_y=12;

static
void callback(void *ptr)
{
     int *x = (int *)ptr;
     FPRINTF(stderr, "x=%d\n", *x);
     STARPU_ASSERT_MSG(*x == expected_x, "%d != %d\n", *x, expected_x);
     (*x)++;
}

static
void prologue_callback(void *ptr)
{
     int *y = (int *)ptr;
     FPRINTF(stderr, "y=%d\n", *y);
     STARPU_ASSERT_MSG(*y == expected_y, "%d != %d\n", *y, expected_y);
     (*y)++;
}

int main(int argc, char **argv)
{
	int ret;
	int x=40;
	int y=12;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	ret = starpu_task_insert(NULL,
				 STARPU_CALLBACK_WITH_ARG, callback, &x,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	expected_x ++;
	ret = starpu_task_insert(NULL,
				 STARPU_CALLBACK, callback,
				 STARPU_CALLBACK_ARG, &x,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	expected_x ++;
	STARPU_ASSERT_MSG(x == expected_x, "x should be equal to %d and not %d\n", expected_x, x);

	ret = starpu_task_insert(NULL,
				 STARPU_PROLOGUE_CALLBACK, prologue_callback,
				 STARPU_PROLOGUE_CALLBACK_ARG, &y,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

#ifdef STARPU_DEVEL
#warning the following code should work
#if 0
	expected_y ++;
	ret = starpu_task_insert(NULL,
				 STARPU_PROLOGUE_CALLBACK_POP, prologue_callback,
				 STARPU_PROLOGUE_CALLBACK_ARG, &y,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
#endif
#endif

	expected_y ++;
	STARPU_ASSERT_MSG(y == expected_y, "y should be equal to %d and not %d\n", expected_y, y);

	starpu_shutdown();

	return EXIT_SUCCESS;
}

