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

/* dumb HIP kernel to fill a 4D matrix */

#include <starpu.h>

static __global__ void f4d_hip(int *arr4d, int nx, int ny, int nz, int nt, unsigned ldy, unsigned ldz, unsigned ldt, float factor)
{
        int i, j, k, l;

        for(l=0; l<nt ; l++)
        {
                for(k=0; k<nz ; k++)
                {
                        for(j=0; j<ny ; j++)
                        {
                                for(i=0; i<nx ; i++)
                                        arr4d[(l*ldt)+(k*ldz)+(j*ldy)+i] *= factor;
                        }
                }
        }
}

extern "C" void f4d_hip_func(void *buffers[], void *_args)
{
        int *factor = (int *)_args;
        int *arr4d = (int *)STARPU_NDIM_GET_PTR(buffers[0]);
        int *nn = (int *)STARPU_NDIM_GET_NN(buffers[0]);
        unsigned *ldn = STARPU_NDIM_GET_LDN(buffers[0]);
        int nx = nn[0];
        int ny = nn[1];
        int nz = nn[2];
        int nt = nn[3];
        unsigned ldy = ldn[1];
        unsigned ldz = ldn[2];
        unsigned ldt = ldn[3];

        hipLaunchKernelGGL(f4d_hip, 1, 1, 0, starpu_hip_get_local_stream(), arr4d, nx, ny, nz, nt, ldy, ldz, ldt, *factor);
        hipError_t status = hipGetLastError();
        if (status != hipSuccess) STARPU_HIP_REPORT_ERROR(status);
}
