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

#include <stdio.h>
#include <starpu.h>

#define MAXNBLOCKS		32
#define MAXTHREADSPERBLOCK	128

static __global__ void cpy_vector(double *dst, double *src, size_t nx)
{
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;
	const int nthreads = gridDim.x * blockDim.x;

	int i;
	for (i = tid; i < nx; i += nthreads)
	{
		dst[i] = src[i];
	}
}

extern "C" void cpy_cuda(void *descr[], void *_args)
{
	double *dst = (double *)STARPU_VECTOR_GET_PTR(descr[0]);
	double *src = (double *)STARPU_VECTOR_GET_PTR(descr[1]);

	size_t nx = STARPU_VECTOR_GET_NX(descr[0]);

	unsigned nblocks = 128;
	unsigned nthread_per_block = STARPU_MIN(MAXTHREADSPERBLOCK, ((nx + nblocks-1) / nblocks));

	cpy_vector<<<nblocks, nthread_per_block, 0, starpu_cuda_get_local_stream()>>>(dst, src, nx);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}
