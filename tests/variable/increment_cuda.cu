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

#include <starpu.h>
#include "../helper.h"

static __global__ void _increment_cuda(unsigned *val)
{
	val[0]++;
}

extern "C" void increment_cuda(void *descr[], void *cl_arg)
{
	unsigned *val = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);

	STARPU_SKIP_IF_VALGRIND;

	_increment_cuda<<<1,1, 0, starpu_cuda_get_local_stream()>>>(val);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}

static __global__ void _redux_cuda(unsigned *dst, unsigned *src)
{
	dst[0] += src[0];
}

extern "C" void redux_cuda(void *descr[], void *arg)
{
	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *src = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);

	STARPU_SKIP_IF_VALGRIND;

	_redux_cuda<<<1,1, 0, starpu_cuda_get_local_stream()>>>(dst, src);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}

static __global__ void _neutral_cuda(unsigned *dst)
{
	dst[0] = 0;
}

extern "C" void neutral_cuda(void *descr[], void *arg)
{
	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);

	STARPU_SKIP_IF_VALGRIND;

	_neutral_cuda<<<1,1, 0, starpu_cuda_get_local_stream()>>>(dst);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}
