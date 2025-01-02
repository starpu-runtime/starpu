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

/* dumb HIP kernel to fill a 3D matrix */

#include <starpu.h>

static __global__ void f3d_hip(int *arr3d, size_t nx, size_t ny, size_t nz, size_t ldy, size_t ldz, float factor)
{
        size_t i, j, k;

        for(k=0; k<nz ; k++)
        {
                for(j=0; j<ny ; j++)
                {
                        for(i=0; i<nx ; i++)
                                arr3d[(k*ldz)+(j*ldy)+i] *= factor;
                }
        }
}

extern "C" void f3d_hip_func(void *buffers[], void *_args)
{
        int *factor = (int *)_args;
        int *arr3d = (int *)STARPU_NDIM_GET_PTR(buffers[0]);
        size_t *nn = STARPU_NDIM_GET_NN(buffers[0]);
        size_t *ldn = STARPU_NDIM_GET_LDN(buffers[0]);
        size_t nx = nn[0];
        size_t ny = nn[1];
        size_t nz = nn[2];
        size_t ldy = ldn[1];
        size_t ldz = ldn[2];

        hipLaunchKernelGGL(f3d_hip, 1, 1, 0, starpu_hip_get_local_stream(), arr3d, nx, ny, nz, ldy, ldz, *factor);
        hipError_t status = hipGetLastError();
        if (status != hipSuccess) STARPU_HIP_REPORT_ERROR(status);
}
