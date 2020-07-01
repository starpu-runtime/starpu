/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 *	increment a (val[0])
 */

static __global__ void _cuda_incA(unsigned *v)
{
	v[0]++;
}

extern "C" void cuda_codelet_incA(void *descr[], void *_args)
{
	unsigned *v = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);

	_cuda_incA<<<1,1, 0, starpu_cuda_get_local_stream()>>>(v);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}

/*
 *	increment c (val[2])
 */

static __global__ void _cuda_incC(unsigned *v)
{
	v[2]++;
}

extern "C" void cuda_codelet_incC(void *descr[], void *_args)
{
	unsigned *v = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);

	_cuda_incC<<<1,1, 0, starpu_cuda_get_local_stream()>>>(v);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}
