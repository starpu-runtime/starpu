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

#include "strassen.h"


static void mult_common_codelet(void *descr[], int s, __attribute__((unused))  void *arg)
{
	float *center 	= (float *)STARPU_GET_MATRIX_PTR(descr[0]);
	float *left 	= (float *)STARPU_GET_MATRIX_PTR(descr[1]);
	float *right 	= (float *)STARPU_GET_MATRIX_PTR(descr[2]);

	unsigned dx = STARPU_GET_MATRIX_NX(descr[0]);
	unsigned dy = STARPU_GET_MATRIX_NY(descr[0]);
	unsigned dz = STARPU_GET_MATRIX_NX(descr[1]);

	unsigned ld21 = STARPU_GET_MATRIX_LD(descr[1]);
	unsigned ld12 = STARPU_GET_MATRIX_LD(descr[2]);
	unsigned ld22 = STARPU_GET_MATRIX_LD(descr[0]);

	switch (s) {
		case 0:
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
				dy, dx, dz, -1.0f, left, ld21, right, ld12,
					     1.0f, center, ld22);
			break;
#ifdef STARPU_USE_CUDA
		case 1:
			cublasSgemm('t', 'n', dx, dy, dz, 
					-1.0f, right, ld12, left, ld21, 
					 1.0f, center, ld22);
			cudaThreadSynchronize();
			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

void mult_cpu_codelet(void *descr[], void *_args)
{
	mult_common_codelet(descr, 0, _args);
}

#ifdef STARPU_USE_CUDA
void mult_cublas_codelet(void *descr[], void *_args)
{
	mult_common_codelet(descr, 1, _args);
}
#endif

static void add_sub_common_codelet(void *descr[], int s, __attribute__((unused))  void *arg, float alpha)
{
	/* C = A op B */

	float *C 	= (float *)STARPU_GET_MATRIX_PTR(descr[0]);
	float *A 	= (float *)STARPU_GET_MATRIX_PTR(descr[1]);
	float *B 	= (float *)STARPU_GET_MATRIX_PTR(descr[2]);

	unsigned dx = STARPU_GET_MATRIX_NX(descr[0]);
	unsigned dy = STARPU_GET_MATRIX_NY(descr[0]);

	unsigned ldA = STARPU_GET_MATRIX_LD(descr[1]);
	unsigned ldB = STARPU_GET_MATRIX_LD(descr[2]);
	unsigned ldC = STARPU_GET_MATRIX_LD(descr[0]);

	// TODO check dim ...

	unsigned line;

	switch (s) {
		case 0:
			for (line = 0; line < dy; line++)
			{
				/* copy line A into C */
				cblas_saxpy(dx, 1.0f, &A[line*ldA], 1, &C[line*ldC], 1);
				/* add line B to C = A */
				cblas_saxpy(dx, alpha, &B[line*ldB], 1, &C[line*ldC], 1);
			}
			break;
#ifdef STARPU_USE_CUDA
		case 1:
			for (line = 0; line < dy; line++)
			{
				/* copy line A into C */
				cublasSaxpy(dx, 1.0f, &A[line*ldA], 1, &C[line*ldC], 1);
				/* add line B to C = A */
				cublasSaxpy(dx, alpha, &B[line*ldB], 1, &C[line*ldC], 1);
			}
			
			cudaThreadSynchronize();

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}

void sub_cpu_codelet(void *descr[], __attribute__((unused))  void *arg)
{
	add_sub_common_codelet(descr, 0, arg, -1.0f);
}

void add_cpu_codelet(void *descr[], __attribute__((unused))  void *arg)
{
	add_sub_common_codelet(descr, 0, arg, 1.0f);
}

#ifdef STARPU_USE_CUDA
void sub_cublas_codelet(void *descr[], __attribute__((unused))  void *arg)
{
	add_sub_common_codelet(descr, 1, arg, -1.0f);
}

void add_cublas_codelet(void *descr[], __attribute__((unused))  void *arg)
{
	add_sub_common_codelet(descr, 1, arg, 1.0f);
}
#endif


static void self_add_sub_common_codelet(void *descr[], int s, __attribute__((unused))  void *arg, float alpha)
{
	/* C +=/-= A */

	float *C 	= (float *)STARPU_GET_MATRIX_PTR(descr[0]);
	float *A 	= (float *)STARPU_GET_MATRIX_PTR(descr[1]);

	unsigned dx = STARPU_GET_MATRIX_NX(descr[0]);
	unsigned dy = STARPU_GET_MATRIX_NY(descr[0]);

	unsigned ldA = STARPU_GET_MATRIX_LD(descr[1]);
	unsigned ldC = STARPU_GET_MATRIX_LD(descr[0]);

	// TODO check dim ...
	
	unsigned line;

	switch (s) {
		case 0:
			for (line = 0; line < dy; line++)
			{
				/* add line A to C */
				cblas_saxpy(dx, alpha, &A[line*ldA], 1, &C[line*ldC], 1);
			}
			break;
#ifdef STARPU_USE_CUDA
		case 1:
			for (line = 0; line < dy; line++)
			{
				/* add line A to C */
				cublasSaxpy(dx, alpha, &A[line*ldA], 1, &C[line*ldC], 1);
			}
			
			cudaThreadSynchronize();

			break;
#endif
		default:
			STARPU_ABORT();
			break;
	}
}




void self_add_cpu_codelet(void *descr[], __attribute__((unused))  void *arg)
{
	self_add_sub_common_codelet(descr, 0, arg, 1.0f);
}

void self_sub_cpu_codelet(void *descr[], __attribute__((unused))  void *arg)
{
	self_add_sub_common_codelet(descr, 0, arg, -1.0f);
}

#ifdef STARPU_USE_CUDA
void self_add_cublas_codelet(void *descr[], __attribute__((unused))  void *arg)
{
	self_add_sub_common_codelet(descr, 1, arg, 1.0f);
}

void self_sub_cublas_codelet(void *descr[], __attribute__((unused))  void *arg)
{
	self_add_sub_common_codelet(descr, 1, arg, -1.0f);
}
#endif
