/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* dumb kernel to fill a vector */

#include <starpu.h>

void vector_cpu_func(void *buffers[], void *cl_arg)
{
	int i;
	int *factor = (int *) cl_arg;

	/* length of the vector */
	int n = (int)STARPU_VECTOR_GET_NX(buffers[0]);
	/* local copy of the vector pointer */
	int *vector = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);

	for (i = 0; i < n; i++)
		vector[i] *= *factor;
}

