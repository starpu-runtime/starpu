/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* dumb CUDA kernel to fill a 1D matrix */

#include <starpu.h>

static __global__ void fvariable_cuda(int *val, int factor)
{
        *val *= factor;
}

extern "C" void variable_cuda_func(void *buffers[], void *_args)
{
        int *factor = (int *)_args;
        int *val = (int *) STARPU_VARIABLE_GET_PTR(buffers[0]);

        fvariable_cuda<<<1,1, 0, starpu_cuda_get_local_stream()>>>(val, *factor);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}
