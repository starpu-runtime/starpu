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

/* dumb HIP kernel to fill a 4D matrix */

#include <starpu.h>

static __global__ void ftensor_hip(int *tensor, size_t nx, size_t ny, size_t nz, size_t nt, size_t ldy, size_t ldz, size_t ldt, float factor)
{
        size_t i, j, k, l;

        for(l=0; l<nt ; l++)
        {
                for(k=0; k<nz ; k++)
                {
                        for(j=0; j<ny ; j++)
                        {
                                for(i=0; i<nx ; i++)
                                        tensor[(l*ldt)+(k*ldz)+(j*ldy)+i] *= factor;
                        }
                }
        }
}

extern "C" void tensor_hip_func(void *buffers[], void *_args)
{
        int *factor = (int *)_args;
        int *tensor = (int *)STARPU_TENSOR_GET_PTR(buffers[0]);
        size_t nx = STARPU_TENSOR_GET_NX(buffers[0]);
        size_t ny = STARPU_TENSOR_GET_NY(buffers[0]);
        size_t nz = STARPU_TENSOR_GET_NZ(buffers[0]);
        size_t nt = STARPU_TENSOR_GET_NT(buffers[0]);
        size_t ldy = STARPU_TENSOR_GET_LDY(buffers[0]);
        size_t ldz = STARPU_TENSOR_GET_LDZ(buffers[0]);
        size_t ldt = STARPU_TENSOR_GET_LDT(buffers[0]);

        hipLaunchKernelGGL(ftensor_hip, 1, 1, 0, starpu_hip_get_local_stream(), tensor, nx, ny, nz, nt, ldy, ldz, ldt, *factor);
        hipError_t status = hipGetLastError();
        if (status != hipSuccess) STARPU_HIP_REPORT_ERROR(status);
}
