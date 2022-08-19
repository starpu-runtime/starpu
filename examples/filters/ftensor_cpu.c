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

/* dumb kernel to fill a 4D matrix */

#include <starpu.h>

void tensor_cpu_func(void *buffers[], void *cl_arg)
{
	int i, j, k, l;
	int *factor = (int *) cl_arg;
	int *tensor = (int *)STARPU_TENSOR_GET_PTR(buffers[0]);
	int nx = (int)STARPU_TENSOR_GET_NX(buffers[0]);
	int ny = (int)STARPU_TENSOR_GET_NY(buffers[0]);
	int nz = (int)STARPU_TENSOR_GET_NZ(buffers[0]);
	int nt = (int)STARPU_TENSOR_GET_NT(buffers[0]);
	unsigned ldy = STARPU_TENSOR_GET_LDY(buffers[0]);
	unsigned ldz = STARPU_TENSOR_GET_LDZ(buffers[0]);
	unsigned ldt = STARPU_TENSOR_GET_LDT(buffers[0]);

	for(l=0; l<nt ; l++)
	{
		for(k=0; k<nz ; k++)
		{
			for(j=0; j<ny ; j++)
			{
				for(i=0; i<nx ; i++)
					tensor[(l*ldt)+(k*ldz)+(j*ldy)+i] *= *factor;
			}
		}
	}
}

