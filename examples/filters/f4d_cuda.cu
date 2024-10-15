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

/* dumb CUDA kernel to fill a 4D matrix */

#include <starpu.h>

static __global__ void f4d_cuda(int *arr4d, size_t nx, size_t ny, size_t nz, size_t nt, size_t ldy, size_t ldz, size_t ldt, float factor)
{
        size_t i, j, k, l;

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

extern "C" void f4d_cuda_func(void *buffers[], void *_args)
{
        int *factor = (int *)_args;
        int *arr4d = (int *)STARPU_NDIM_GET_PTR(buffers[0]);
        size_t *nn = STARPU_NDIM_GET_NN(buffers[0]);
        size_t *ldn = STARPU_NDIM_GET_LDN(buffers[0]);
        size_t nx = nn[0];
        size_t ny = nn[1];
        size_t nz = nn[2];
        size_t nt = nn[3];
        size_t ldy = ldn[1];
        size_t ldz = ldn[2];
        size_t ldt = ldn[3];

        f4d_cuda<<<1,1, 0, starpu_cuda_get_local_stream()>>>(arr4d, nx, ny, nz, nt, ldy, ldz, ldt, *factor);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}
