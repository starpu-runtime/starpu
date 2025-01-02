/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void f3d_cpu_func(void *buffers[], void *cl_arg)
{
	size_t i, j, k;
	int *factor = (int *) cl_arg;
	int *arr3d = (int *)STARPU_NDIM_GET_PTR(buffers[0]);
	size_t *nn = STARPU_NDIM_GET_NN(buffers[0]);
	size_t *ldn = STARPU_NDIM_GET_LDN(buffers[0]);
	size_t nx = nn[0];
	size_t ny = nn[1];
	size_t nz = nn[2];
	size_t ldy = ldn[1];
	size_t ldz = ldn[2];

	for(k=0; k<nz ; k++)
	{
		for(j=0; j<ny ; j++)
		{
			for(i=0; i<nx ; i++)
				arr3d[(k*ldz)+(j*ldy)+i] *= *factor;
		}
	}
}

