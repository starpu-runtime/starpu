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

#include "dw_cholesky.h"
#include "../common/blas.h"

/*
 *   U22 
 */

static inline void chol_common_core_codelet_update_u22(starpu_data_interface_t *buffers, int s, __attribute__((unused)) void *_args)
{
	//printf("22\n");
	float *left 	= (float *)buffers[0].blas.ptr;
	float *right 	= (float *)buffers[1].blas.ptr;
	float *center 	= (float *)buffers[2].blas.ptr;

	unsigned dx = buffers[2].blas.ny;
	unsigned dy = buffers[2].blas.nx;
	unsigned dz = buffers[0].blas.ny;

	unsigned ld21 = buffers[0].blas.ld;
	unsigned ld12 = buffers[1].blas.ld;
	unsigned ld22 = buffers[2].blas.ld;

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
			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}
}

void chol_core_codelet_update_u22(starpu_data_interface_t *descr, void *_args)
{
	chol_common_core_codelet_update_u22(descr, 0, _args);
}

#ifdef USE_CUDA
void chol_cublas_codelet_update_u22(starpu_data_interface_t *descr, void *_args)
{
	chol_common_core_codelet_update_u22(descr, 1, _args);
}
#endif// USE_CUDA

/* 
 * U21
 */

static inline void chol_common_codelet_update_u21(starpu_data_interface_t *buffers, int s, __attribute__((unused)) void *_args)
{
//	printf("21\n");
	float *sub11;
	float *sub21;

	sub11 = (float *)buffers[0].blas.ptr;
	sub21 = (float *)buffers[1].blas.ptr;

	unsigned ld11 = buffers[0].blas.ld;
	unsigned ld21 = buffers[1].blas.ld;

	unsigned nx21 = buffers[1].blas.ny;
	unsigned ny21 = buffers[1].blas.nx;

	switch (s) {
		case 0:
			STRSM("R", "L", "T", "N", nx21, ny21, 1.0f, sub11, ld11, sub21, ld21);
			break;
#ifdef USE_CUDA
		case 1:
			cublasStrsm('R', 'L', 'T', 'N', nx21, ny21, 1.0f, sub11, ld11, sub21, ld21);
			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}
}

void chol_core_codelet_update_u21(starpu_data_interface_t *descr, void *_args)
{
	 chol_common_codelet_update_u21(descr, 0, _args);
}

#ifdef USE_CUDA
void chol_cublas_codelet_update_u21(starpu_data_interface_t *descr, void *_args)
{
	chol_common_codelet_update_u21(descr, 1, _args);
}
#endif 

/*
 *	U11
 */

static inline void chol_common_codelet_update_u11(starpu_data_interface_t *descr, int s, __attribute__((unused)) void *_args) 
{
//	printf("11\n");
	float *sub11;

	sub11 = (float *)descr[0].blas.ptr; 

	unsigned nx = descr[0].blas.ny;
	unsigned ld = descr[0].blas.ld;

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
				/* ok that's dirty and ridiculous ... */
				cublasGetVector(1, sizeof(float), &sub11[z+z*ld], sizeof(float), &lambda11, sizeof(float));

				lambda11 = sqrt(lambda11);

				cublasSetVector(1, sizeof(float), &lambda11, sizeof(float), &sub11[z+z*ld], sizeof(float));

				STARPU_ASSERT(lambda11 != 0.0f);
				
				cublasSscal(nx - z - 1, 1.0f/lambda11, &sub11[(z+1)+z*ld], 1);

				cublasSsyr('U', nx - z - 1, -1.0f,
							&sub11[(z+1)+z*ld], 1,
							&sub11[(z+1)+(z+1)*ld], ld);
			}
			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}

}


void chol_core_codelet_update_u11(starpu_data_interface_t *descr, void *_args)
{
	chol_common_codelet_update_u11(descr, 0, _args);
}

#ifdef USE_CUDA
void chol_cublas_codelet_update_u11(starpu_data_interface_t *descr, void *_args)
{
	chol_common_codelet_update_u11(descr, 1, _args);
}
#endif// USE_CUDA
