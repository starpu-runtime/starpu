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

#include <starpu.h>
#include <common/blas.h>

#define COMMON_CODE			\
	uint32_t nxC, nyC, nyA;		\
	uint32_t ldA, ldB, ldC;		\
					\
	float *subA;			\
	float *subB;			\
	float *subC;			\
					\
	subA = (float *)GET_BLAS_PTR(descr[0]);	\
	subB = (float *)GET_BLAS_PTR(descr[1]);	\
	subC = (float *)GET_BLAS_PTR(descr[2]);	\
					\
	nxC = GET_BLAS_NX(descr[2]);		\
	nyC = GET_BLAS_NY(descr[2]);		\
	nyA = GET_BLAS_NY(descr[0]);		\
					\
	ldA = GET_BLAS_LD(descr[0]);		\
	ldB = GET_BLAS_LD(descr[1]);		\
	ldC = GET_BLAS_LD(descr[2]);



#ifdef USE_CUDA
void cublas_mult(void *descr[], __attribute__((unused)) void *arg)
{
	COMMON_CODE

	starpu_trace_user_event(0x42);

	cublasSgemm('n', 'n', nxC, nyC, nyA, 1.0f, subA, ldA, subB, ldB, 
					     0.0f, subC, ldC);
	cublasStatus st;
	st = cublasGetError();
	if (st != CUBLAS_STATUS_SUCCESS)
		STARPU_ABORT();

	cudaThreadSynchronize();

	starpu_trace_user_event(0x43);
}
#endif

void core_mult(void *descr[], __attribute__((unused))  void *arg)
{
	COMMON_CODE

	starpu_trace_user_event(0x42);
	SGEMM("N", "N", nxC, nyC, nyA, 1.0f, subA, ldA, subB, ldB, 0.0f, subC, ldC);
	starpu_trace_user_event(0x43);
}
