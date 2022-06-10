/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
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

/*
 * This example shows a simple implementation of a blocked matrix
 * multiplication. Note that this is NOT intended to be an efficient
 * implementation of sgemm! In this example, we show:
 *  - how to declare dense matrices (starpu_matrix_data_register)
 *  - how to manipulate matrices within codelets (eg. descr[0].blas.ld)
 *  - how to use filters to partition the matrices into blocks
 *    (starpu_data_partition and starpu_data_map_filters)
 *  - how to unpartition data (starpu_data_unpartition) and how to stop
 *    monitoring data (starpu_data_unregister)
 *  - how to manipulate subsets of data (starpu_data_get_sub_data)
 *  - how to construct an autocalibrated performance model (starpu_perfmodel)
 *  - how to submit asynchronous tasks
 */

#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <signal.h>
#include <starpu.h>

#define THREADS_PER_BLOCK 64

/*
 * That program should compute C = A * B
 *
 *   A of size (z,y)
 *   B of size (x,z)
 *   C of size (x,y)

              |---------------|
            z |       B       |
              |---------------|
       z              x
     |----|   |---------------|
     |    |   |               |
     |    |   |               |
     | A  | y |       C       |
     |    |   |               |
     |    |   |               |
     |----|   |---------------|

 * Note: we use FORTRAN ordering.
 */

/*
 * The codelet is passed 3 matrices, the "descr" union-type field gives a
 * description of the layout of those 3 matrices in the local memory (ie. RAM
 * in the case of CPU, GPU frame buffer in the case of GPU etc.). Since we have
 * registered data with the "matrix" data interface, we use the matrix macros.
 */
static __global__ void cuda_mult_kernel(uint32_t nxC, uint32_t nyC, uint32_t nyA,
				       uint32_t ldA, uint32_t ldB, uint32_t ldC,
				       float * subA, float * subB, float * subC )
{
	uint32_t id, i, j, k;
	float sum;
	id = blockIdx.x * blockDim.x + threadIdx.x;
	i = id % nxC;
	j = id / nxC;
	if (j >= nyC)
	{
		return;
	}
	sum = 0.;
	for (k = 0 ; k < nyA ; k++)
	{
		sum += subA[i + k*ldA] * subB[k + j*ldB];
	}
	subC[i + j*ldC] = sum;
}

extern "C" void cuda_mult(void *descr[], void *arg)
{
	(void)arg;
	float *subA, *subB, *subC;
	uint32_t nxC, nyC, nyA;
	uint32_t ldA, ldB, ldC;
	uint32_t nblocks;

	/* ptr gives a pointer to the first element of the local copy */
	subA = (float *)STARPU_MATRIX_GET_PTR(descr[0]);
	subB = (float *)STARPU_MATRIX_GET_PTR(descr[1]);
	subC = (float *)STARPU_MATRIX_GET_PTR(descr[2]);

	/*
	 * Note: STARPU_MATRIX_GET_NX/NY is different from X/Y of the FORTRAN
	 * ordering:
	 * - nx is the number of consecutive elements (thus the number of rows
	 *   in FORTRAN order)
	 * - ny is the number of series that are separated by ld elements (thus
	 *   the number of columns in FORTRAN order)
	 * - ld stands for leading dimension
	 *
	 * NB: in case some filters were used, the leading dimension is not
	 * guaranteed to be the same in main memory (on the original matrix)
	 * and on the accelerator! */

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
}
