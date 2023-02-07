/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2021, 2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "complex_interface.h"

static __global__ void complex_copy_cuda(double *output, double *input, unsigned n)
{
        unsigned i =  blockIdx.x*blockDim.x + threadIdx.x;

	if (i < n)
	{
		output[i] = input[i];
	}
}

extern "C" void copy_complex_codelet_cuda(void *descr[], void *_args)
{
	(void)_args;

	int nx = STARPU_COMPLEX_GET_NX(descr[0]);

	double *input = STARPU_COMPLEX_GET_PTR(descr[0]);
	double *output = STARPU_COMPLEX_GET_PTR(descr[1]);

	unsigned threads_per_block = 64;
	unsigned nblocks = (2*nx + threads_per_block-1) / threads_per_block;

        complex_copy_cuda<<<nblocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>(o_real, o_imaginary, i_real, i_imaginary, 2*nx);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}
