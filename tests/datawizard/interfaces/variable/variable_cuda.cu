/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../test_interfaces.h"

extern struct test_config variable_config;

static __global__ void variable_cuda(int *val, int *err, int factor)
{
        unsigned i =  blockIdx.x*blockDim.x + threadIdx.x;

	if (i > 0)
		return;

	if (*val != 42 * factor)
		*err = 1;
	else
		*val *= -1;
}

extern "C" void test_variable_cuda_func(void *buffers[], void *args)
{
	cudaError_t error;
	int *ret;

	error = cudaMalloc(&ret, sizeof(int));
	if (error != cudaSuccess)
		STARPU_CUDA_REPORT_ERROR(error);

	error = cudaMemcpyAsync(ret,
			   &variable_config.copy_failed,
			   sizeof(int),
			   cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());
	if (error != cudaSuccess)
		STARPU_CUDA_REPORT_ERROR(error);

        int *val = (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);
	int factor = *(int*) args;

	unsigned threads_per_block = 64;
	unsigned nblocks = 1;

        variable_cuda<<<nblocks,threads_per_block,0,starpu_cuda_get_local_stream()>>>(val, ret, factor);
	error = cudaGetLastError();
	if (error != cudaSuccess) STARPU_CUDA_REPORT_ERROR(error);
	error = cudaMemcpyAsync(&variable_config.copy_failed,
			   ret,
			   sizeof(int),
			   cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
	if (error != cudaSuccess)
		STARPU_CUDA_REPORT_ERROR(error);

	cudaFree(ret);
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
