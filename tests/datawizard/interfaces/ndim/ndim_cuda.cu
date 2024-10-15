/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

extern struct test_config arr4d_config;

static __global__ void arr4d_cuda(int *arr4d,
				  size_t nx, size_t ny, size_t nz, size_t nt,
				  size_t ldy, size_t ldz, size_t ldt,
				  int factor, int *err)
{
        int i, j, k, l;
	int val = 0;

        for (l = 0; l < nt ;l++)
	{
	    for (k = 0; k < nz ;k++)
	    {
                for (j = 0; j < ny ;j++)
		{
                        for(i = 0; i < nx ;i++)
			{
				if (arr4d[(l*ldt)+(k*ldz)+(j*ldy)+i] != factor * val)
				{
					*err = 1;
					return;
				}
				else
				{
					arr4d[(l*ldt)+(k*ldz)+(j*ldy)+i] *= -1;
					val++;
				}
			}
                }
	    }
        }
}

extern "C" void test_arr4d_cuda_func(void *buffers[], void *args)
{
	cudaError_t error;
	int *ret;

	error = cudaMalloc(&ret, sizeof(int));
	if (error != cudaSuccess)
		STARPU_CUDA_REPORT_ERROR(error);

	error = cudaMemcpyAsync(ret, &arr4d_config.copy_failed, sizeof(int), cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());
	if (error != cudaSuccess)
		STARPU_CUDA_REPORT_ERROR(error);

	size_t *nn = STARPU_NDIM_GET_NN(buffers[0]);
	size_t *ldn = STARPU_NDIM_GET_LDN(buffers[0]);
	size_t nx = nn[0];
	size_t ny = nn[1];
	size_t nz = nn[2];
	size_t nt = nn[3];
	size_t ldy = ldn[1];
	size_t ldz = ldn[2];
	size_t ldt = ldn[3];
	int *arr4d = (int *) STARPU_NDIM_GET_PTR(buffers[0]);
	int factor = *(int*) args;

        arr4d_cuda<<<1,1, 0, starpu_cuda_get_local_stream()>>> (arr4d, nx, ny, nz, nt, ldy, ldz, ldt, factor, ret);
	error = cudaGetLastError();
	if (error != cudaSuccess) STARPU_CUDA_REPORT_ERROR(error);
	error = cudaMemcpyAsync(&arr4d_config.copy_failed, ret, sizeof(int), cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
	if (error != cudaSuccess)
		STARPU_CUDA_REPORT_ERROR(error);

	cudaFree(ret);
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
