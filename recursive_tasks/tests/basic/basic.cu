/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2023-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static __global__ void print_vector_cuda(unsigned n, int *v, int prefix)
{
	unsigned i;
	printf(prefix == 0 ? "task" : (prefix == 1 ? "subtask" : (prefix == 2 ? "task_ro" : "subtask_ro")));

	for (i=0; i<n; i++)
	{
		printf(" %d", v[i]);
	}
	printf("\n");
}

static __global__ void scal_cuda(unsigned n, int *v, int factor)
{
	unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		v[i] *= factor;
	}
}

extern "C" void sub_data_cuda_func(void *buffers[], void *arg)
{
        int *v = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
        int nx = STARPU_VECTOR_GET_NX(buffers[0]);

	unsigned threads_per_block = 64;
        unsigned nblocks = (nx + threads_per_block-1) / threads_per_block;
        scal_cuda<<<nblocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>(nx, v, 2);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
        cudaStreamSynchronize(starpu_cuda_get_local_stream());

	if (!getenv("STARPU_SSILENT"))
	{
		print_vector_cuda<<<1, 1, 0, starpu_cuda_get_local_stream()>>>(nx, v, 1);
	        cudaError_t status = cudaGetLastError();
	        if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
	        cudaStreamSynchronize(starpu_cuda_get_local_stream());
	}
}

extern "C" void sub_data_RO_cuda_func(void *buffers[], void *arg)
{
        int *v = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
        int nx = STARPU_VECTOR_GET_NX(buffers[0]);
        if (!getenv("STARPU_SSILENT"))
        {
                print_vector_cuda<<<1, 1, 0, starpu_cuda_get_local_stream()>>>(nx, v, 3);
	        cudaError_t status = cudaGetLastError();
	        if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
	        cudaStreamSynchronize(starpu_cuda_get_local_stream());
	}
}

static __global__ void add_cuda(unsigned n, int *v, int term)
{
	unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		v[i] += term;
	}

}

extern "C" void task_cuda_func(void *buffers[], void *arg)
{
	int *v = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	int nx = STARPU_VECTOR_GET_NX(buffers[0]);
        if (!getenv("STARPU_SSILENT"))
        {
                print_vector_cuda<<<1, 1, 0, starpu_cuda_get_local_stream()>>>(nx, v, 0);
                cudaError_t status = cudaGetLastError();
                if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
                cudaStreamSynchronize(starpu_cuda_get_local_stream());
        }

        unsigned threads_per_block = 64;
        unsigned nblocks = (nx + threads_per_block-1) / threads_per_block;
        add_cuda<<<nblocks, threads_per_block, 0, starpu_cuda_get_local_stream()>>>(nx, v, 10);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
        cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

extern "C" void task_RO_cuda_func(void *buffers[], void *arg)
{
        int *v = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
        int nx = STARPU_VECTOR_GET_NX(buffers[0]);

        if (!getenv("STARPU_SSILENT"))
        {
                print_vector_cuda<<<1, 1, 0, starpu_cuda_get_local_stream()>>>(nx, v, 2);
                cudaError_t status = cudaGetLastError();
                if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
                cudaStreamSynchronize(starpu_cuda_get_local_stream());
        }
}
