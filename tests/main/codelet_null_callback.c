/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Centre National de la Recherche Scientifique
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

void callback(void *ptr)
{
     int *x = (int *)ptr;
     FPRINTF(stderr, "x=%d\n", *x);
     STARPU_ASSERT(*x == 42);
}

int main(int argc, char **argv)
{
	int ret;
	int x=42;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	ret = starpu_insert_task(NULL,
				 STARPU_CALLBACK_WITH_ARG, callback, &x,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_insert_task");

	starpu_task_wait_for_all();
	starpu_shutdown();

	return EXIT_SUCCESS;
}

