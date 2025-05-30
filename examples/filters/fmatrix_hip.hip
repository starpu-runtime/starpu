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

/* dumb HIP kernel to fill a 2D matrix */

#include <starpu.h>

static __global__ void fmatrix_hip(int *matrix, size_t nx, size_t ny, size_t ld, float factor)
{
        size_t i, j;
        for(j=0; j<ny ; j++)
        {
                for(i=0; i<nx ; i++)
                        matrix[(j*ld)+i] *= factor;
        }
}

extern "C" void matrix_hip_func(void *buffers[], void *_args)
{
        int *factor = (int *)_args;
        int *matrix = (int *)STARPU_MATRIX_GET_PTR(buffers[0]);
        size_t nx = STARPU_MATRIX_GET_NX(buffers[0]);
        size_t ny = STARPU_MATRIX_GET_NY(buffers[0]);
        size_t ld = STARPU_MATRIX_GET_LD(buffers[0]);

        hipLaunchKernelGGL(fmatrix_hip, 1, 1, 0, starpu_hip_get_local_stream(), matrix, nx, ny, ld, *factor);
        hipError_t status = hipGetLastError();
        if (status != hipSuccess) STARPU_HIP_REPORT_ERROR(status);
}
