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

#include "xlu.h"
#include <math.h>

/*
 *   U22 
 */

static inline void STARPU_LU(common_u22)(starpu_data_interface_t *buffers,
				int s, __attribute__((unused)) void *_args)
{
	TYPE *right 	= (TYPE *)buffers[0].blas.ptr;
	TYPE *left 	= (TYPE *)buffers[1].blas.ptr;
	TYPE *center 	= (TYPE *)buffers[2].blas.ptr;

	unsigned dx = buffers[2].blas.nx;
	unsigned dy = buffers[2].blas.ny;
	unsigned dz = buffers[0].blas.ny;

	unsigned ld12 = buffers[0].blas.ld;
	unsigned ld21 = buffers[1].blas.ld;
	unsigned ld22 = buffers[2].blas.ld;

#ifdef USE_CUDA
	cublasStatus status;
	cudaError_t cures;
#endif

	switch (s) {
		case 0:
			CPU_GEMM("N", "N", dy, dx, dz, 
				(TYPE)-1.0, right, ld21, left, ld12,
				(TYPE)1.0, center, ld22);
			break;

#ifdef USE_CUDA
		case 1:
			CUBLAS_GEMM('n', 'n', dx, dy, dz,
				(TYPE)-1.0, right, ld21, left, ld12,
				(TYPE)1.0f, center, ld22);

			status = cublasGetError();
			if (STARPU_UNLIKELY(status != CUBLAS_STATUS_SUCCESS))
				STARPU_ASSERT(0);

			if (STARPU_UNLIKELY((cures = cudaThreadSynchronize()) != cudaSuccess))
				CUDA_REPORT_ERROR(cures);

			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}
}

void STARPU_LU(cpu_u22)(starpu_data_interface_t *descr, void *_args)
{
	STARPU_LU(common_u22)(descr, 0, _args);
}

#ifdef USE_CUDA
void STARPU_LU(cublas_u22)(starpu_data_interface_t *descr, void *_args)
{
	STARPU_LU(common_u22)(descr, 1, _args);
}
#endif// USE_CUDA

/*
 * U12
 */

static inline void STARPU_LU(common_u12)(starpu_data_interface_t *buffers,
				int s, __attribute__((unused)) void *_args)
{
	TYPE *sub11;
	TYPE *sub12;

	sub11 = (TYPE *)buffers[0].blas.ptr;	
	sub12 = (TYPE *)buffers[1].blas.ptr;

	unsigned ld11 = buffers[0].blas.ld;
	unsigned ld12 = buffers[1].blas.ld;

	unsigned nx12 = buffers[1].blas.nx;
	unsigned ny12 = buffers[1].blas.ny;

#ifdef USE_CUDA
	cublasStatus status;
	cudaError_t cures;
#endif

	/* solve L11 U12 = A12 (find U12) */
	switch (s) {
		case 0:
			CPU_TRSM("L", "L", "N", "N", nx12, ny12,
					(TYPE)1.0, sub11, ld11, sub12, ld12);
			break;
#ifdef USE_CUDA
		case 1:
			CUBLAS_TRSM('L', 'L', 'N', 'N', ny12, nx12,
					(TYPE)1.0, sub11, ld11, sub12, ld12);

			status = cublasGetError();
			if (STARPU_UNLIKELY(status != CUBLAS_STATUS_SUCCESS))
				STARPU_ASSERT(0);

			if (STARPU_UNLIKELY((cures = cudaThreadSynchronize()) != cudaSuccess))
				CUDA_REPORT_ERROR(cures);

			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}
}

void STARPU_LU(cpu_u12)(starpu_data_interface_t *descr, void *_args)
{
	STARPU_LU(common_u12)(descr, 0, _args);
}

#ifdef USE_CUDA
void STARPU_LU(cublas_u12)(starpu_data_interface_t *descr, void *_args)
{
	STARPU_LU(common_u12)(descr, 1, _args);
}
#endif // USE_CUDA

/* 
 * U21
 */

static inline void STARPU_LU(common_u21)(starpu_data_interface_t *buffers,
				int s, __attribute__((unused)) void *_args)
{
	TYPE *sub11;
	TYPE *sub21;

	sub11 = (TYPE *)buffers[0].blas.ptr;
	sub21 = (TYPE *)buffers[1].blas.ptr;

	unsigned ld11 = buffers[0].blas.ld;
	unsigned ld21 = buffers[1].blas.ld;

	unsigned nx21 = buffers[1].blas.nx;
	unsigned ny21 = buffers[1].blas.ny;
	
#ifdef USE_CUDA
	cublasStatus status;
	cudaError_t cures;
#endif

	switch (s) {
		case 0:
			CPU_TRSM("R", "U", "N", "U", nx21, ny21,
					(TYPE)1.0, sub11, ld11, sub21, ld21);
			break;
#ifdef USE_CUDA
		case 1:
			CUBLAS_TRSM('R', 'U', 'N', 'U', ny21, nx21,
					(TYPE)1.0, sub11, ld11, sub21, ld21);

			status = cublasGetError();
			if (status != CUBLAS_STATUS_SUCCESS)
				STARPU_ASSERT(0);

			cudaThreadSynchronize();

			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}
}

void STARPU_LU(cpu_u21)(starpu_data_interface_t *descr, void *_args)
{
	STARPU_LU(common_u21)(descr, 0, _args);
}

#ifdef USE_CUDA
void STARPU_LU(cublas_u21)(starpu_data_interface_t *descr, void *_args)
{
	STARPU_LU(common_u21)(descr, 1, _args);
}
#endif 

/*
 *	U11
 */

static inline void STARPU_LU(common_u11)(starpu_data_interface_t *descr,
				int s, __attribute__((unused)) void *_args)
{
	TYPE *sub11;

	sub11 = (TYPE *)descr[0].blas.ptr; 

	unsigned long nx = descr[0].blas.nx;
	unsigned long ld = descr[0].blas.ld;

	unsigned long z;

	switch (s) {
		case 0:
			for (z = 0; z < nx; z++)
			{
				TYPE pivot;
				pivot = sub11[z+z*ld];
				STARPU_ASSERT(pivot != 0.0);
		
				CPU_SCAL(nx - z - 1, (1.0/pivot), &sub11[z+(z+1)*ld], ld);
		
				CPU_GER(nx - z - 1, nx - z - 1, -1.0,
						&sub11[(z+1)+z*ld], 1,
						&sub11[z+(z+1)*ld], ld,
						&sub11[(z+1) + (z+1)*ld],ld);
			}
			break;
#ifdef USE_CUDA
		case 1:
			for (z = 0; z < nx; z++)
			{
				TYPE pivot;
				cudaMemcpy(&pivot, &sub11[z+z*ld], sizeof(TYPE), cudaMemcpyDeviceToHost);
				cudaStreamSynchronize(0);

				STARPU_ASSERT(pivot != 0.0);
				
				CUBLAS_SCAL(nx - z - 1, 1.0/pivot, &sub11[z+(z+1)*ld], ld);
				
				CUBLAS_GER(nx - z - 1, nx - z - 1, -1.0,
						&sub11[(z+1)+z*ld], 1,
						&sub11[z+(z+1)*ld], ld,
						&sub11[(z+1) + (z+1)*ld],ld);
			}
			
			cudaThreadSynchronize();

			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}
}

void STARPU_LU(cpu_u11)(starpu_data_interface_t *descr, void *_args)
{
	STARPU_LU(common_u11)(descr, 0, _args);
}

#ifdef USE_CUDA
void STARPU_LU(cublas_u11)(starpu_data_interface_t *descr, void *_args)
{
	STARPU_LU(common_u11)(descr, 1, _args);
}
#endif// USE_CUDA

/*
 *	U11 with pivoting
 */

static inline void STARPU_LU(common_u11_pivot)(starpu_data_interface_t *descr,
				int s, void *_args)
{
	TYPE *sub11;

	sub11 = (TYPE *)descr[0].blas.ptr; 

	unsigned long nx = descr[0].blas.nx;
	unsigned long ld = descr[0].blas.ld;

	unsigned long z;

	struct piv_s *piv = _args;
	unsigned *ipiv = piv->piv;
	unsigned first = piv->first;

	int i,j;

	switch (s) {
		case 0:
			for (z = 0; z < nx; z++)
			{
				TYPE pivot;
				pivot = sub11[z+z*ld];

				if (fabs((double)(pivot)) < PIVOT_THRESHHOLD)
				{

					/* find the pivot */
					int piv_ind = CPU_IAMAX(nx - z, &sub11[z*(ld+1)], ld);

					ipiv[z + first] = piv_ind + z + first;

					/* swap if needed */
					if (piv_ind != 0)
					{
						CPU_SWAP(nx, &sub11[z*ld], 1, &sub11[(z+piv_ind)*ld], 1);
					}

					pivot = sub11[z+z*ld];
				}
			
				STARPU_ASSERT(pivot != 0.0);

				CPU_SCAL(nx - z - 1, (1.0/pivot), &sub11[z+(z+1)*ld], ld);
		
				CPU_GER(nx - z - 1, nx - z - 1, -1.0,
						&sub11[(z+1)+z*ld], 1,
						&sub11[z+(z+1)*ld], ld,
						&sub11[(z+1) + (z+1)*ld],ld);
			}

			break;
#ifdef USE_CUDA
		case 1:
			for (z = 0; z < nx; z++)
			{
				TYPE pivot;
				cudaMemcpy(&pivot, &sub11[z+z*ld], sizeof(TYPE), cudaMemcpyDeviceToHost);
				cudaStreamSynchronize(0);

				if (fabs((double)(pivot)) < PIVOT_THRESHHOLD)
				{
					/* find the pivot */
					int piv_ind = CUBLAS_IAMAX(nx - z, &sub11[z*(ld+1)], ld) - 1;
	
					ipiv[z + first] = piv_ind + z + first;

					/* swap if needed */
					if (piv_ind != 0)
					{
						CUBLAS_SWAP(nx, &sub11[z*ld], 1, &sub11[(z+piv_ind)*ld], 1);
					}

					cudaMemcpy(&pivot, &sub11[z+z*ld], sizeof(TYPE), cudaMemcpyDeviceToHost);
					cudaStreamSynchronize(0);
				}

				STARPU_ASSERT(pivot != 0.0);
				
				CUBLAS_SCAL(nx - z - 1, 1.0/pivot, &sub11[z+(z+1)*ld], ld);
				
				CUBLAS_GER(nx - z - 1, nx - z - 1, -1.0,
						&sub11[(z+1)+z*ld], 1,
						&sub11[z+(z+1)*ld], ld,
						&sub11[(z+1) + (z+1)*ld],ld);
				
			}

			cudaThreadSynchronize();

			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}
}

void STARPU_LU(cpu_u11_pivot)(starpu_data_interface_t *descr, void *_args)
{
	STARPU_LU(common_u11_pivot)(descr, 0, _args);
}

#ifdef USE_CUDA
void STARPU_LU(cublas_u11_pivot)(starpu_data_interface_t *descr, void *_args)
{
	STARPU_LU(common_u11_pivot)(descr, 1, _args);
}
#endif// USE_CUDA

/*
 *	Pivoting
 */

static inline void STARPU_LU(common_pivot)(starpu_data_interface_t *descr,
				int s, void *_args)
{
	TYPE *matrix;

	matrix = (TYPE *)descr[0].blas.ptr; 
	unsigned long nx = descr[0].blas.nx;
	unsigned long ld = descr[0].blas.ld;

	unsigned row, rowaux;

	struct piv_s *piv = _args;
	unsigned *ipiv = piv->piv;
	unsigned first = piv->first;
	unsigned last = piv->last;

	switch (s) {
		case 0:
			for (row = 0; row < nx; row++)
			{
				unsigned rowpiv = ipiv[row+first] - first;
				if (rowpiv != row)
				{
					CPU_SWAP(nx, &matrix[row*ld], 1, &matrix[rowpiv*ld], 1);
				}
			}
			break;
#ifdef USE_CUDA
		case 1:
			for (row = 0; row < nx; row++)
			{
				unsigned rowpiv = ipiv[row+first] - first;
				if (rowpiv != row)
				{
					CUBLAS_SWAP(nx, &matrix[row*ld], 1, &matrix[rowpiv*ld], 1);
				}
			}

			cudaThreadSynchronize();

			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}
}

void STARPU_LU(cpu_pivot)(starpu_data_interface_t *descr, void *_args)
{
	STARPU_LU(common_pivot)(descr, 0, _args);
}

#ifdef USE_CUDA
void STARPU_LU(cublas_pivot)(starpu_data_interface_t *descr, void *_args)
{
	STARPU_LU(common_pivot)(descr, 1, _args);
}
#endif// USE_CUDA


