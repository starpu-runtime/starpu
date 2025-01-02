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

#include <starpu.h>
#include "multiformat_types.h"

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

static __global__ void multiformat_cuda(struct struct_of_arrays *soa, unsigned n)
{
        unsigned i =  blockIdx.x*blockDim.x + threadIdx.x;

	if (i < n)
		soa->x[i] *= soa->y[i];
}

extern "C" void multiformat_scal_cuda_func(void *buffers[], void *_args)
{
	(void) _args;

	FPRINTF(stderr, "Running the cuda kernel (%s)\n", __starpu_func__);
	unsigned int n = STARPU_MULTIFORMAT_GET_NX(buffers[0]);
	struct struct_of_arrays *soa;

	soa = (struct struct_of_arrays *) STARPU_MULTIFORMAT_GET_CUDA_PTR(buffers[0]);
	unsigned threads_per_block = 64;
	unsigned nblocks = (n + threads_per_block-1) / threads_per_block;
        multiformat_cuda<<<nblocks,threads_per_block,2,starpu_cuda_get_local_stream()>>>(soa, n);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);

	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
