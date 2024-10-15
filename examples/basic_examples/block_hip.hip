/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static __global__ void hip_block(float *block, size_t nx, size_t ny, size_t nz, size_t ldy, size_t ldz, float multiplier)
{
        size_t i, j, k;
        for(k=0; k<nz ; k++)
	{
                for(j=0; j<ny ; j++)
		{
                        for(i=0; i<nx ; i++)
                                block[(k*ldz)+(j*ldy)+i] *= multiplier;
                }
        }
}

extern "C" void hip_codelet(void *descr[], void *_args)
{
        float *block = (float *)STARPU_BLOCK_GET_PTR(descr[0]);
	size_t nx = STARPU_BLOCK_GET_NX(descr[0]);
	size_t ny = STARPU_BLOCK_GET_NY(descr[0]);
	size_t nz = STARPU_BLOCK_GET_NZ(descr[0]);
        size_t ldy = STARPU_BLOCK_GET_LDY(descr[0]);
        size_t ldz = STARPU_BLOCK_GET_LDZ(descr[0]);
        float *multiplier = (float *)_args;

        hipLaunchKernelGGL(hip_block, 1, 1, 0, 0,
			   block, nx, ny, nz,
			   ldy, ldz, *multiplier);
        hipError_t status = hipGetLastError();
        if (status != hipSuccess) STARPU_HIP_REPORT_ERROR(status);
}
