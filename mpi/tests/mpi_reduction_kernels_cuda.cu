/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static __global__ void cuda_init(long int *dot)
{
	(*dot) = 0;
}

static __global__ void cuda_redux(long int *dota, long int* dotb)
{
	(*dota) = (*dota) + (*dotb);
}

static __global__ void cuda_dot(long int *local_x, size_t n, long int* dot)
{
	size_t i;
	for (i = 0; i < n; i++)
	{
		(*dot) += local_x[i];
	}
}

extern "C" {

/*
 *	Codelet to create a neutral element
 */
void init_cuda_func(void *descr[], void *cl_arg)
{
	(void) cl_arg;
	long int *dotptr = (long int *)STARPU_VECTOR_GET_PTR(descr[0]);

	cuda_init<<<1,1, 0, starpu_cuda_get_local_stream()>>>(dotptr);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}

/*
 *	Codelet to perform the reduction of two elements
 */
void redux_cuda_func(void *descr[], void *cl_arg)
{
	(void) cl_arg;
	long int *dota = (long int *)STARPU_VECTOR_GET_PTR(descr[0]);
	long int *dotb = (long int *)STARPU_VECTOR_GET_PTR(descr[1]);

	long int dota_host = 0;
	long int dotb_host = 0;

	cuda_redux<<<1,1, 0, starpu_cuda_get_local_stream()>>>(dota, dotb);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}

/*
 *	Dot product codelet
 */
void dot_cuda_func(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	long int *local_x = (long int *)STARPU_VECTOR_GET_PTR(descr[0]);
	size_t n = STARPU_VECTOR_GET_NX(descr[0]);

	long int *dot = (long int *)STARPU_VARIABLE_GET_PTR(descr[1]);

	cuda_dot<<<1,1, 0, starpu_cuda_get_local_stream()>>>(local_x, n, dot);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}

}
