/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

extern struct test_config tensor_config;

static __global__ void tensor_cuda(int *tensor,
				  int nx, int ny, int nz, int nt,
				  unsigned ldy, unsigned ldz, unsigned ldt,
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
				if (tensor[(l*ldt)+(k*ldz)+(j*ldy)+i] != factor * val)
				{
					*err = 1;
					return;
				}
				else
				{
					tensor[(l*ldt)+(k*ldz)+(j*ldy)+i] *= -1;
					val++;
				}
			}
                }
	    }
        }
}

extern "C" void test_tensor_cuda_func(void *buffers[], void *args)
{
	cudaError_t error;
	int *ret;

	error = cudaMalloc(&ret, sizeof(int));
	if (error != cudaSuccess)
		STARPU_CUDA_REPORT_ERROR(error);

	error = cudaMemcpyAsync(ret, &tensor_config.copy_failed, sizeof(int), cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());
	if (error != cudaSuccess)
		STARPU_CUDA_REPORT_ERROR(error);

	int nx = STARPU_TENSOR_GET_NX(buffers[0]);
	int ny = STARPU_TENSOR_GET_NY(buffers[0]);
	int nz = STARPU_TENSOR_GET_NZ(buffers[0]);
	int nt = STARPU_TENSOR_GET_NT(buffers[0]);
        unsigned ldy = STARPU_TENSOR_GET_LDY(buffers[0]);
        unsigned ldz = STARPU_TENSOR_GET_LDZ(buffers[0]);
        unsigned ldt = STARPU_TENSOR_GET_LDT(buffers[0]);
	int *tensor = (int *) STARPU_TENSOR_GET_PTR(buffers[0]);
	int factor = *(int*) args;

        tensor_cuda<<<1,1, 0, starpu_cuda_get_local_stream()>>>
		(tensor, nx, ny, nz, nt, ldy, ldz, ldt, factor, ret);
	error = cudaGetLastError();
	if (error != cudaSuccess) STARPU_CUDA_REPORT_ERROR(error);
	error = cudaMemcpyAsync(&tensor_config.copy_failed, ret, sizeof(int), cudaMemcpyDeviceToHost, starpu_cuda_get_local_stream());
	if (error != cudaSuccess)
		STARPU_CUDA_REPORT_ERROR(error);

	cudaFree(ret);
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
