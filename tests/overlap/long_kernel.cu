/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

extern "C" __global__
void long_kernel(unsigned long niters)
{
	unsigned long i;
	for (i = 0; i < niters; i++)
		__syncthreads();
}

extern "C" void long_kernel_cuda(unsigned long niters)
{
	dim3 dimBlock(1,1);
	dim3 dimGrid(1,1);
	long_kernel<<<dimGrid, dimBlock, 0, starpu_cuda_get_local_stream()>>>(niters);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}
