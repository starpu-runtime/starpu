/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Standard GEMV kernel (on one matrix block of the sparse matrix)
 */
#include "dw_block_spmv.h"

/*
 *   U22 
 */

#ifdef STARPU_USE_CUDA
#include <starpu_cublas_v2.h>
static const float p1 =  1.0;
static const float m1 = -1.0;
#endif

static inline void common_block_spmv(void *descr[], int s, void *_args)
{
	/* printf("22\n"); */
	float *block 	= (float *)STARPU_MATRIX_GET_PTR(descr[0]);
	float *in 	= (float *)STARPU_VECTOR_GET_PTR(descr[1]);
	float *out 	= (float *)STARPU_VECTOR_GET_PTR(descr[2]);

	unsigned dx = STARPU_MATRIX_GET_NX(descr[0]);
	unsigned dy = STARPU_MATRIX_GET_NY(descr[0]);

	unsigned ld = STARPU_MATRIX_GET_LD(descr[0]);

	switch (s)
	{
		case 0:
			cblas_sgemv(CblasRowMajor, CblasNoTrans, dx, dy, 1.0f, block, ld, in, 1, 1.0f, out, 1);
			break;
#ifdef STARPU_USE_CUDA
		case 1:
		{
			cublasStatus_t status = cublasSgemv (starpu_cublas_get_local_handle(),
					CUBLAS_OP_T, dx, dy, &p1, block, ld, in, 1, &p1, out, 1);
			if (status != CUBLAS_STATUS_SUCCESS)
				STARPU_CUBLAS_REPORT_ERROR(status);
			break;
		}
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

void cpu_block_spmv(void *descr[], void *_args)
{
/*	printf("CPU CODELET \n"); */

	common_block_spmv(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void cublas_block_spmv(void *descr[], void *_args)
{
/*	printf("CUBLAS CODELET \n"); */

	common_block_spmv(descr, 1, _args);
}
#endif /* STARPU_USE_CUDA */
