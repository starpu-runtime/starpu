/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* dumb CUDA kernel to check the matrix values and scale it up */

#include <starpu.h>

static __global__ void _fmultiple_check_scale_cuda(int *val, int nx, int ny, unsigned ld, int start, int factor)
{
        int i, j;
	for(j=0; j<ny ; j++)
	{
		for(i=0; i<nx ; i++)
		{
			if (val[(j*ld)+i] != start + factor*(i+100*j))
				asm("trap;");
			val[(j*ld)+i] *= 2;
		}
        }
}

extern "C" void fmultiple_check_scale_cuda(void *buffers[], void *cl_arg)
{
	int start, factor;
	int nx = (int)STARPU_MATRIX_GET_NX(buffers[0]);
	int ny = (int)STARPU_MATRIX_GET_NY(buffers[0]);
        unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
	int *val = (int *)STARPU_MATRIX_GET_PTR(buffers[0]);

	starpu_codelet_unpack_args(cl_arg, &start, &factor);

        /* TODO: use more vals and threads in vals */
	_fmultiple_check_scale_cuda<<<1,1, 0, starpu_cuda_get_local_stream()>>>(val, nx, ny, ld, start, factor);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}

static __global__ void _fmultiple_check_cuda(int *val, int nx, int ny, unsigned ld, int start, int factor)
{
        int i, j;
	for(j=0; j<ny ; j++)
	{
		for(i=0; i<nx ; i++)
		{
			if (val[(j*ld)+i] != start + factor*(i+100*j))
				asm("trap;");
		}
        }
}

extern "C" void fmultiple_check_cuda(void *buffers[], void *cl_arg)
{
	int start, factor;
	int nx = (int)STARPU_MATRIX_GET_NX(buffers[0]);
	int ny = (int)STARPU_MATRIX_GET_NY(buffers[0]);
        unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
	int *val = (int *)STARPU_MATRIX_GET_PTR(buffers[0]);

	starpu_codelet_unpack_args(cl_arg, &start, &factor);

        /* TODO: use more vals and threads in vals */
	_fmultiple_check_cuda<<<1,1, 0, starpu_cuda_get_local_stream()>>>(val, nx, ny, ld, start, factor);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}
