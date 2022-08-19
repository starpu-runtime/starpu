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

/* dumb kernel to fill a 2D matrix */

#include <starpu.h>

void matrix_cpu_func(void *buffers[], void *cl_arg)
{
	int i, j;
	int *factor = (int *) cl_arg;

	/* length of the matrix */
	int nx = (int)STARPU_MATRIX_GET_NX(buffers[0]);
	int ny = (int)STARPU_MATRIX_GET_NY(buffers[0]);
	unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
	/* local copy of the matrix pointer */
	int *matrix = (int *)STARPU_MATRIX_GET_PTR(buffers[0]);

	for(j=0; j<ny ; j++)
	{
		for(i=0; i<nx ; i++)
			matrix[(j*ld)+i] *= *factor;
	}
}

