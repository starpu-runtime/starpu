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

/* dumb kernel to fill a 3D matrix */

#include <starpu.h>

void block_cpu_func(void *buffers[], void *cl_arg)
{
	int i, j, k;
	int *factor = (int *) cl_arg;
	int *block = (int *)STARPU_BLOCK_GET_PTR(buffers[0]);
	int nx = (int)STARPU_BLOCK_GET_NX(buffers[0]);
	int ny = (int)STARPU_BLOCK_GET_NY(buffers[0]);
	int nz = (int)STARPU_BLOCK_GET_NZ(buffers[0]);
	unsigned ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
	unsigned ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);

	for(k=0; k<nz ; k++)
	{
		for(j=0; j<ny ; j++)
		{
			for(i=0; i<nx ; i++)
				block[(k*ldz)+(j*ldy)+i] *= *factor;
		}
	}
}

