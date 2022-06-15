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

static __global__ void _increment_cuda_codelet(unsigned *val)
{
	val[0]++;
}

extern "C" void increment_cuda(void *descr[], void *cl_arg)
{
	unsigned *val = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);

	_increment_cuda_codelet<<<1,1, 0, starpu_cuda_get_local_stream()>>>(val);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}

extern "C" void redux_cuda_kernel(void *descr[], void *arg)
{
	(void)arg;

	STARPU_SKIP_IF_VALGRIND;

	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *src = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);

	unsigned host_dst, host_src;

	/* This is a dummy technique of course */
	cudaMemcpyAsync(&host_src, src, sizeof(unsigned), cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
	cudaMemcpyAsync(&host_dst, dst, sizeof(unsigned), cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
	cudaStreamSynchronize(starpu_cuda_get_local_stream());

	host_dst += host_src;

	cudaMemcpyAsync(dst, &host_dst, sizeof(unsigned), cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());
}

extern "C" void neutral_cuda_kernel(void *descr[], void *arg)
{
	(void)arg;

	STARPU_SKIP_IF_VALGRIND;

	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);

	/* This is a dummy technique of course */
	unsigned host_dst = 0;
	cudaMemcpyAsync(dst, &host_dst, sizeof(unsigned), cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());
}
