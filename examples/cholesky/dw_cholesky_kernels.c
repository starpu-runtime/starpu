/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu_config.h>
#include "dw_cholesky.h"
#include "../common/blas.h"
#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>
#endif

/*
 *   U22 
 */

static inline void chol_common_cpu_codelet_update_u22(void *descr[], int s, __attribute__((unused)) void *_args)
{
	//printf("22\n");
	float *left 	= (float *)STARPU_GET_BLAS_PTR(descr[0]);
	float *right 	= (float *)STARPU_GET_BLAS_PTR(descr[1]);
	float *center 	= (float *)STARPU_GET_BLAS_PTR(descr[2]);

	unsigned dx = STARPU_GET_BLAS_NY(descr[2]);
	unsigned dy = STARPU_GET_BLAS_NX(descr[2]);
	unsigned dz = STARPU_GET_BLAS_NY(descr[0]);

	unsigned ld21 = STARPU_GET_BLAS_LD(descr[0]);
	unsigned ld12 = STARPU_GET_BLAS_LD(descr[1]);
	unsigned ld22 = STARPU_GET_BLAS_LD(descr[2]);

#ifdef USE_CUDA
	cublasStatus st;
#endif

	switch (s) {
		case 0:
			SGEMM("N", "T", dy, dx, dz, -1.0f, left, ld21, 
				right, ld12, 1.0f, center, ld22);
			break;
#ifdef USE_CUDA
		case 1:
			cublasSgemm('n', 't', dy, dx, dz, 
					-1.0f, left, ld21, right, ld12, 
					 1.0f, center, ld22);
			st = cublasGetError();
			STARPU_ASSERT(!st);

			cudaThreadSynchronize();

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

void chol_cpu_codelet_update_u22(void *descr[], void *_args)
{
	chol_common_cpu_codelet_update_u22(descr, 0, _args);
}

#ifdef USE_CUDA
void chol_cublas_codelet_update_u22(void *descr[], void *_args)
{
	chol_common_cpu_codelet_update_u22(descr, 1, _args);
}
#endif// USE_CUDA

/* 
 * U21
 */

static inline void chol_common_codelet_update_u21(void *descr[], int s, __attribute__((unused)) void *_args)
{
//	printf("21\n");
	float *sub11;
	float *sub21;

	sub11 = (float *)STARPU_GET_BLAS_PTR(descr[0]);
	sub21 = (float *)STARPU_GET_BLAS_PTR(descr[1]);

	unsigned ld11 = STARPU_GET_BLAS_LD(descr[0]);
	unsigned ld21 = STARPU_GET_BLAS_LD(descr[1]);

	unsigned nx21 = STARPU_GET_BLAS_NY(descr[1]);
	unsigned ny21 = STARPU_GET_BLAS_NX(descr[1]);

	switch (s) {
		case 0:
			STRSM("R", "L", "T", "N", nx21, ny21, 1.0f, sub11, ld11, sub21, ld21);
			break;
#ifdef USE_CUDA
		case 1:
			cublasStrsm('R', 'L', 'T', 'N', nx21, ny21, 1.0f, sub11, ld11, sub21, ld21);
			cudaThreadSynchronize();
			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

void chol_cpu_codelet_update_u21(void *descr[], void *_args)
{
	 chol_common_codelet_update_u21(descr, 0, _args);
}

#ifdef USE_CUDA
void chol_cublas_codelet_update_u21(void *descr[], void *_args)
{
	chol_common_codelet_update_u21(descr, 1, _args);
}
#endif 

/*
 *	U11
 */

static inline void chol_common_codelet_update_u11(void *descr[], int s, __attribute__((unused)) void *_args) 
{
//	printf("11\n");
	float *sub11;

	sub11 = (float *)STARPU_GET_BLAS_PTR(descr[0]); 

	unsigned nx = STARPU_GET_BLAS_NY(descr[0]);
	unsigned ld = STARPU_GET_BLAS_LD(descr[0]);

	unsigned z;

	switch (s) {
		case 0:

			/*
			 *	- alpha 11 <- lambda 11 = sqrt(alpha11)
			 *	- alpha 21 <- l 21	= alpha 21 / lambda 11
			 *	- A22 <- A22 - l21 trans(l21)
			 */

			for (z = 0; z < nx; z++)
			{
				float lambda11;
				lambda11 = sqrt(sub11[z+z*ld]);
				sub11[z+z*ld] = lambda11;

				STARPU_ASSERT(lambda11 != 0.0f);
		
				SSCAL(nx - z - 1, 1.0f/lambda11, &sub11[(z+1)+z*ld], 1);
		
				SSYR("L", nx - z - 1, -1.0f, 
							&sub11[(z+1)+z*ld], 1,
							&sub11[(z+1)+(z+1)*ld], ld);
			}
			break;
#ifdef USE_CUDA
		case 1:
			for (z = 0; z < nx; z++)
			{
				float lambda11;
				cudaMemcpy(&lambda11, &sub11[z+z*ld], sizeof(float), cudaMemcpyDeviceToHost);
				cudaStreamSynchronize(0);

				STARPU_ASSERT(lambda11 != 0.0f);
				
				lambda11 = sqrt(lambda11);

				cublasSetVector(1, sizeof(float), &lambda11, sizeof(float), &sub11[z+z*ld], sizeof(float));

				cublasSscal(nx - z - 1, 1.0f/lambda11, &sub11[(z+1)+z*ld], 1);

				cublasSsyr('U', nx - z - 1, -1.0f,
							&sub11[(z+1)+z*ld], 1,
							&sub11[(z+1)+(z+1)*ld], ld);
			}
		
			cudaThreadSynchronize();

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}


void chol_cpu_codelet_update_u11(void *descr[], void *_args)
{
	chol_common_codelet_update_u11(descr, 0, _args);
}

#ifdef USE_CUDA
void chol_cublas_codelet_update_u11(void *descr[], void *_args)
{
	chol_common_codelet_update_u11(descr, 1, _args);
}
#endif// USE_CUDA
