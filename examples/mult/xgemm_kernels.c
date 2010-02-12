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
	TYPE *subA;			\
	TYPE *subB;			\
	TYPE *subC;			\
					\
	subA = (TYPE *)STARPU_GET_BLAS_PTR(descr[0]);	\
	subB = (TYPE *)STARPU_GET_BLAS_PTR(descr[1]);	\
	subC = (TYPE *)STARPU_GET_BLAS_PTR(descr[2]);	\
					\
	nxC = STARPU_GET_BLAS_NX(descr[2]);		\
	nyC = STARPU_GET_BLAS_NY(descr[2]);		\
	nyA = STARPU_GET_BLAS_NY(descr[0]);		\
					\
	ldA = STARPU_GET_BLAS_LD(descr[0]);		\
	ldB = STARPU_GET_BLAS_LD(descr[1]);		\
	ldC = STARPU_GET_BLAS_LD(descr[2]);



#ifdef USE_CUDA
void STARPU_GEMM(cublas_mult)(void *descr[], __attribute__((unused)) void *arg)
{
	COMMON_CODE

	starpu_trace_user_event(0x42);

	CUBLAS_GEMM('n', 'n', nxC, nyC, nyA, (TYPE)1.0, subA, ldA, subB, ldB,
					     (TYPE)0.0, subC, ldC);
	cublasStatus st;
	st = cublasGetError();
	if (st != CUBLAS_STATUS_SUCCESS)
		STARPU_ABORT();

	cudaThreadSynchronize();

	starpu_trace_user_event(0x42);
}
#endif

void STARPU_GEMM(core_mult)(void *descr[], __attribute__((unused))  void *arg)
{
	COMMON_CODE

	starpu_trace_user_event(0x42);
	CPU_GEMM("N", "N", nxC, nyC, nyA, (TYPE)1.0, subA, ldA, subB, ldB,
					  (TYPE)0.0, subC, ldC);

	starpu_trace_user_event(0x43);
}
