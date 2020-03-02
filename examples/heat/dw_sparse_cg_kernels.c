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

#include "dw_sparse_cg.h"

#ifdef STARPU_USE_CUDA
#include <starpu_cublas_v2.h>
#endif

/*
 *	Algorithm :
 *
 *		i = 0
 *		r = b - A x
 *			( d = A x ; r = r - d )
 *		d = r
 *		delta_new = trans(r) r
 *		delta_0 = delta_new
 *
 * 		while (i < i_max && delta_new > eps^2 delta_0)
 * 		{
 *			q = A d
 *			alpha = delta_new / ( trans(d) q )
 *			x = x + alpha d
 *			if ( i is divisible by 50 )
 *				r = b - A x
 *			else
 *				r = r - alpha q
 *			delta_old = delta_new
 *			delta_new = trans(r) r
 *			beta = delta_new / delta_old
 *			d = r + beta d
 *			i = i + 1
 * 		}
 */


/*
 *	compute r = b - A x
 *
 *		descr[0] = A, descr[1] = x, descr [2] = r, descr[3] = b
 */

void cpu_codelet_func_1(void *descr[], void *arg)
{
	(void)arg;
	float *nzval = (float *)STARPU_CSR_GET_NZVAL(descr[0]);
	uint32_t *colind = STARPU_CSR_GET_COLIND(descr[0]);
	uint32_t *rowptr = STARPU_CSR_GET_ROWPTR(descr[0]);

	uint32_t firstentry = STARPU_CSR_GET_ELEMSIZE(descr[0]);

	float *vecx = (float *)STARPU_VECTOR_GET_PTR(descr[1]);
	float *vecr = (float *)STARPU_VECTOR_GET_PTR(descr[2]);
	float *vecb = (float *)STARPU_VECTOR_GET_PTR(descr[3]);


	uint32_t nrow;

	nrow = STARPU_CSR_GET_NROW(descr[0]);

	unsigned row;
	for (row = 0; row < nrow; row++)
	{
		float tmp = 0.0f;
		unsigned index;

		unsigned firstindex = rowptr[row] - firstentry;
		unsigned lastindex = rowptr[row+1] - firstentry;

		for (index = firstindex; index < lastindex; index++)
		{
			unsigned col;

			col = colind[index];
			tmp += nzval[index]*vecx[col];
		}

		vecr[row] = vecb[row] - tmp;
	}
}

/*
 *	compute d = r
 *		descr[0] = d, descr[1] = r
 */
void cpu_codelet_func_2(void *descr[], void *arg)
{
	(void)arg;
	/* simply copy r into d */
	uint32_t nx = STARPU_VECTOR_GET_NX(descr[0]);
	size_t elemsize = STARPU_VECTOR_GET_ELEMSIZE(descr[0]);

	STARPU_ASSERT(STARPU_VECTOR_GET_NX(descr[0]) == STARPU_VECTOR_GET_NX(descr[1]));
	STARPU_ASSERT(STARPU_VECTOR_GET_ELEMSIZE(descr[0]) == STARPU_VECTOR_GET_ELEMSIZE(descr[1]));

	float *src = (float *)STARPU_VECTOR_GET_PTR(descr[1]);
	float *dst = (float *)STARPU_VECTOR_GET_PTR(descr[0]);

	memcpy(dst, src, nx*elemsize);
}

/*
 *	compute delta_new = trans(r) r
 *		delta_0   = delta_new
 *
 *		args = &delta_new, &delta_0
 */

void cpu_codelet_func_3(void *descr[], void *arg)
{
	struct cg_problem *pb = arg;
	float dot;
	float *vec;
	int size;

	/* get the vector */
	vec = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	size = (int)STARPU_VECTOR_GET_NX(descr[0]);

	dot = STARPU_SDOT(size, vec, 1, vec, 1);

	fprintf(stderr, "func 3 : DOT = %f\n", dot);

	pb->delta_new = dot;
	pb->delta_0 = dot;
}

#ifdef STARPU_USE_CUDA
void cublas_codelet_func_3(void *descr[], void *arg)
{
	struct cg_problem *pb = arg;
	float dot;
	float *vec;
	uint32_t size;

	/* get the vector */
	vec = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	size = STARPU_VECTOR_GET_NX(descr[0]);

	cublasStatus_t status = cublasSdot (starpu_cublas_get_local_handle(), size, vec, 1, vec, 1, &dot);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
	cudaStreamSynchronize(starpu_cuda_get_local_stream());

	pb->delta_new = dot;
	pb->delta_0 = dot;
}
#endif


/*
 *	compute q with : q = A d
 *
 *		descr[0] = A, descr[1] = d, descr [2] = q
 */

void cpu_codelet_func_4(void *descr[], void *arg)
{
	(void)arg;
	float *nzval = (float *)STARPU_CSR_GET_NZVAL(descr[0]);
	uint32_t *colind = STARPU_CSR_GET_COLIND(descr[0]);
	uint32_t *rowptr = STARPU_CSR_GET_ROWPTR(descr[0]);

	uint32_t firstentry = STARPU_CSR_GET_FIRSTENTRY(descr[0]);

	float *vecd = (float *)STARPU_VECTOR_GET_PTR(descr[1]);
	float *vecq = (float *)STARPU_VECTOR_GET_PTR(descr[2]);

	uint32_t nrow;

	nrow = STARPU_CSR_GET_NROW(descr[0]);

	unsigned row;
	for (row = 0; row < nrow; row++)
	{
		float tmp = 0.0f;
		unsigned index;

		unsigned firstindex = rowptr[row] - firstentry;
		unsigned lastindex = rowptr[row+1] - firstentry;

		for (index = firstindex; index < lastindex; index++)
		{
			unsigned col;

			col = colind[index];
			tmp += nzval[index]*vecd[col];
		}

		vecq[row] = tmp;
	}

}

/*
 *	compute alpha = delta_new / ( trans(d) q )
 *
 * 		descr[0] = d, descr[1] = q
 *		args = &alpha, &delta_new
 */

void cpu_codelet_func_5(void *descr[], void *arg)
{
	float dot;
	struct cg_problem *pb = arg;
	float *vecd, *vecq;
	uint32_t size;

	/* get the vector */
	vecd = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	vecq = (float *)STARPU_VECTOR_GET_PTR(descr[1]);

	STARPU_ASSERT(STARPU_VECTOR_GET_NX(descr[0]) == STARPU_VECTOR_GET_NX(descr[1]));
	size = STARPU_VECTOR_GET_NX(descr[0]);

	dot = STARPU_SDOT(size, vecd, 1, vecq, 1);

	pb->alpha = pb->delta_new / dot;
}

#ifdef STARPU_USE_CUDA
void cublas_codelet_func_5(void *descr[], void *arg)
{
	float dot;
	struct cg_problem *pb = arg;
	float *vecd, *vecq;
	uint32_t size;

	/* get the vector */
	vecd = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	vecq = (float *)STARPU_VECTOR_GET_PTR(descr[1]);

	STARPU_ASSERT(STARPU_VECTOR_GET_NX(descr[0]) == STARPU_VECTOR_GET_NX(descr[1]));
	size = STARPU_VECTOR_GET_NX(descr[0]);

	cublasStatus_t status = cublasSdot (starpu_cublas_get_local_handle(), size, vecd, 1, vecq, 1, &dot);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
	cudaStreamSynchronize(starpu_cuda_get_local_stream());

	pb->alpha = pb->delta_new / dot;
}
#endif



/*
 *	compute x = x + alpha d
 *
 * 		descr[0] : x, descr[1] : d
 *		args = &alpha
 */

void cpu_codelet_func_6(void *descr[], void *arg)
{
	struct cg_problem *pb = arg;
	float *vecx, *vecd;
	uint32_t size;

	/* get the vector */
	vecx = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	vecd = (float *)STARPU_VECTOR_GET_PTR(descr[1]);

	size = STARPU_VECTOR_GET_NX(descr[0]);

	STARPU_SAXPY(size, pb->alpha, vecd, 1, vecx, 1);
}

#ifdef STARPU_USE_CUDA
void cublas_codelet_func_6(void *descr[], void *arg)
{
	struct cg_problem *pb = arg;
	float *vecx, *vecd;
	uint32_t size;

	/* get the vector */
	vecx = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	vecd = (float *)STARPU_VECTOR_GET_PTR(descr[1]);

	size = STARPU_VECTOR_GET_NX(descr[0]);

	cublasStatus_t status = cublasSaxpy (starpu_cublas_get_local_handle(), size, &pb->alpha, vecd, 1, vecx, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}
#endif

/*
 *	compute r = r - alpha q
 *
 * 		descr[0] : r, descr[1] : q
 *		args = &alpha
 */

void cpu_codelet_func_7(void *descr[], void *arg)
{
	struct cg_problem *pb = arg;
	float *vecr, *vecq;
	uint32_t size;

	/* get the vector */
	vecr = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	vecq = (float *)STARPU_VECTOR_GET_PTR(descr[1]);

	size = STARPU_VECTOR_GET_NX(descr[0]);

	STARPU_SAXPY(size, -pb->alpha, vecq, 1, vecr, 1);
}

#ifdef STARPU_USE_CUDA
void cublas_codelet_func_7(void *descr[], void *arg)
{
	struct cg_problem *pb = arg;
	float *vecr, *vecq;
	uint32_t size;

	/* get the vector */
	vecr = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	vecq = (float *)STARPU_VECTOR_GET_PTR(descr[1]);

	size = STARPU_VECTOR_GET_NX(descr[0]);

	float scal = -pb->alpha;

	cublasStatus_t status = cublasSaxpy (starpu_cublas_get_local_handle(), size, &scal, vecq, 1, vecr, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}
#endif

/*
 *	compute delta_old = delta_new
 *		delta_new = trans(r) r
 *		beta = delta_new / delta_old
 *
 * 		descr[0] = r
 *		args = &delta_old, &delta_new, &beta
 */

void cpu_codelet_func_8(void *descr[], void *arg)
{
	float dot;
	struct cg_problem *pb = arg;
	float *vecr;
	uint32_t size;

	/* get the vector */
	vecr = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	size = STARPU_VECTOR_GET_NX(descr[0]);

	dot = STARPU_SDOT(size, vecr, 1, vecr, 1);

	pb->delta_old = pb->delta_new;
	pb->delta_new = dot;
	pb->beta = pb->delta_new/pb->delta_old;
}

#ifdef STARPU_USE_CUDA
void cublas_codelet_func_8(void *descr[], void *arg)
{
	float dot;
	struct cg_problem *pb = arg;
	float *vecr;
	uint32_t size;

	/* get the vector */
	vecr = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	size = STARPU_VECTOR_GET_NX(descr[0]);

	cublasStatus_t status = cublasSdot(starpu_cublas_get_local_handle(), size, vecr, 1, vecr, 1, &dot);
	if (status != CUBLAS_STATUS_SUCCESS) STARPU_CUBLAS_REPORT_ERROR(status);
	cudaStreamSynchronize(starpu_cuda_get_local_stream());

	pb->delta_old = pb->delta_new;
	pb->delta_new = dot;
	pb->beta = pb->delta_new/pb->delta_old;
}
#endif

/*
 *	compute d = r + beta d
 *
 * 		descr[0] : d, descr[1] : r
 *		args = &beta
 *
 */

void cpu_codelet_func_9(void *descr[], void *arg)
{
	struct cg_problem *pb = arg;
	float *vecd, *vecr;
	uint32_t size;

	/* get the vector */
	vecd = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	vecr = (float *)STARPU_VECTOR_GET_PTR(descr[1]);

	size = STARPU_VECTOR_GET_NX(descr[0]);

	/* d = beta d */
	STARPU_SSCAL(size, pb->beta, vecd, 1);

	/* d = r + d */
	STARPU_SAXPY (size, 1.0f, vecr, 1, vecd, 1);
}

#ifdef STARPU_USE_CUDA
void cublas_codelet_func_9(void *descr[], void *arg)
{
	struct cg_problem *pb = arg;
	float *vecd, *vecr;
	uint32_t size;

	/* get the vector */
	vecd = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	vecr = (float *)STARPU_VECTOR_GET_PTR(descr[1]);

	size = STARPU_VECTOR_GET_NX(descr[0]);

	/* d = beta d */
	cublasStatus_t status;
	status = cublasSscal(starpu_cublas_get_local_handle(), size, &pb->beta, vecd, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);

	/* d = r + d */
	float scal = 1.0f;
	status = cublasSaxpy (starpu_cublas_get_local_handle(), size, &scal, vecr, 1, vecd, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}
#endif
