/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2018-2018  Alexis Juven
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
#include <stdint.h>
#include <stdio.h>

__global__ void gpuMultKernel
(
	size_t nxC, size_t nyC, size_t nyA,
	size_t ldA, size_t ldB, size_t ldC,
	float * subA, float * subB, float * subC
)
{
	size_t id, i, j, k;
	float sum;

	id = blockIdx.x * blockDim.x + threadIdx.x;
	i = id % nxC;
	j = id / nxC;

	if (j >= nyC){
		return;
	}

	sum = 0.;

	for (k = 0 ; k < nyA ; k++){
		sum += subA[i + k*ldA] * subB[k + j*ldB];
	}

	subC[i + j*ldC] = sum;
}

#define THREADS_PER_BLOCK 64

extern "C" void gpu_mult(void * descr[], void * args)
{
	float * d_subA, * d_subB, * d_subC;
	size_t nxC, nyC, nyA;
	size_t ldA, ldB, ldC;
	size_t nblocks;

	d_subA = (float *) STARPU_MATRIX_GET_PTR(descr[0]);
	d_subB = (float *) STARPU_MATRIX_GET_PTR(descr[1]);
	d_subC = (float *) STARPU_MATRIX_GET_PTR(descr[2]);

	nxC = STARPU_MATRIX_GET_NX(descr[2]);
	nyC = STARPU_MATRIX_GET_NY(descr[2]);
	nyA = STARPU_MATRIX_GET_NY(descr[0]);

	ldA = STARPU_MATRIX_GET_LD(descr[0]);
	ldB = STARPU_MATRIX_GET_LD(descr[1]);
	ldC = STARPU_MATRIX_GET_LD(descr[2]);

	nblocks = (nxC * nyC + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

	gpuMultKernel
		<<< nblocks, THREADS_PER_BLOCK, 0, starpu_cuda_get_local_stream()
		>>> (nxC, nyC, nyA, ldA, ldB, ldC, d_subA, d_subB, d_subC);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);

	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
