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

/* dumb HIP kernel to fill a 1D matrix */

#include <starpu.h>

static __global__ void fvector_hip(int *vector, int n, float factor)
{
        int i;
        for (i = 0; i < n; i++)
                vector[i] *= factor;
}

extern "C" void vector_hip_func(void *buffers[], void *_args)
{
        int *factor = (int *)_args;
        int *vector = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
        int n = (int)STARPU_VECTOR_GET_NX(buffers[0]);

        hipLaunchKernelGGL(fvector_hip, 1, 1, 0, starpu_hip_get_local_stream(), vector, n, *factor);
        hipError_t status = hipGetLastError();
        if (status != hipSuccess) STARPU_HIP_REPORT_ERROR(status);
}
