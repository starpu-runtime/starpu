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

/*
 * This example complements vector_scal.c: here we implement a HIP version.
 */


#include <starpu.h>

static __global__ void vector_mult_hip(size_t n, float *val, float factor)
{
        unsigned i =  blockIdx.x*blockDim.x + threadIdx.x;

	if (i < n)
               val[i] *= factor;
}

extern "C" void scal_hip_func(void *buffers[], void *_args)
{
        float *factor = (float *)_args;
        /* length of the vector */
        size_t n = STARPU_VECTOR_GET_NX(buffers[0]);
        /* local copy of the vector pointer */
        float *val = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
	unsigned threads_per_block = 64;
	unsigned nblocks = (n + threads_per_block-1) / threads_per_block;

        hipLaunchKernelGGL(vector_mult_hip, nblocks, threads_per_block, 0, starpu_hip_get_local_stream(), n, val, *factor);
        hipError_t status = hipGetLastError();
        if (status != hipSuccess) STARPU_HIP_REPORT_ERROR(status);
}
