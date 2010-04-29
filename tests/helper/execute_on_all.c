/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>

void func(void *arg)
{
	int *ptr = arg;
	STARPU_ASSERT(*ptr == 0x42);
}

int main(int argc, char **argv)
{
	starpu_init(NULL);

	int arg = 0x42;

	starpu_execute_on_each_worker(func, &arg, STARPU_CPU|STARPU_CUDA|STARPU_OPENCL);

	starpu_execute_on_each_worker(func, &arg, STARPU_CPU);
	
	starpu_execute_on_each_worker(func, &arg, STARPU_CUDA);

        starpu_execute_on_each_worker(func, &arg, STARPU_OPENCL);

	starpu_shutdown();

	return 0;
}
