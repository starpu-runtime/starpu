/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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

#include <starpu.h>
#include <starpu_cuda.h>
#include <common/blas.h>

#define COMMON_CODE			\
	uint32_t nxC, nyC, nyA;		\
	uint32_t ldA, ldB, ldC;		\
					\
	float *subA;			\
	float *subB;			\
	float *subC;			\
					\
	subA = (float *)STARPU_MATRIX_GET_PTR(descr[0]);	\
	subB = (float *)STARPU_MATRIX_GET_PTR(descr[1]);	\
	subC = (float *)STARPU_MATRIX_GET_PTR(descr[2]);	\
					\
	nxC = STARPU_MATRIX_GET_NX(descr[2]);		\
	nyC = STARPU_MATRIX_GET_NY(descr[2]);		\
	nyA = STARPU_MATRIX_GET_NY(descr[0]);		\
					\
	ldA = STARPU_MATRIX_GET_LD(descr[0]);		\
	ldB = STARPU_MATRIX_GET_LD(descr[1]);		\
	ldC = STARPU_MATRIX_GET_LD(descr[2]);



#ifdef STARPU_USE_CUDA

#ifdef STARPU_HAVE_MAGMA
#define GPU_SGEMM magmablas_sgemm
#else
#define GPU_SGEMM cublasSgemm
#endif

void cublas_mult(void *descr[], __attribute__((unused)) void *arg)
{
	COMMON_CODE

	starpu_trace_user_event(0x42);

	GPU_SGEMM('n', 'n', nxC, nyC, nyA, 1.0f, subA, ldA, subB, ldB, 
					     0.0f, subC, ldC);
	cublasStatus st;
	st = cublasGetError();
	if (st != CUBLAS_STATUS_SUCCESS)
		STARPU_ABORT();

	cudaThreadSynchronize();

	starpu_trace_user_event(0x43);
}
#endif

void cpu_mult(void *descr[], __attribute__((unused))  void *arg)
{
	COMMON_CODE

	starpu_trace_user_event(0x42);
	SGEMM("N", "N", nxC, nyC, nyA, 1.0f, subA, ldA, subB, ldB, 0.0f, subC, ldC);
	starpu_trace_user_event(0x43);
}
