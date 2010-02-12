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

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>
#include <semaphore.h>

#include <starpu_config.h>
#ifdef USE_CUDA
#include <cublas.h>
#endif

#include "../common/blas.h"

#include <starpu.h>

static double cublas_flop = 0.0;
static double cpus_flop = 0.0;

void display_perf(double timing, unsigned size)
{
	double total_flop_n3 = (2.0*size*size*size);
	double total_flop = cublas_flop + cpus_flop;

	fprintf(stderr, "Computation took (ms):\n");
	printf("%2.2f\n", timing/1000);
	fprintf(stderr, "       GFlop : O(n3) -> %2.2f\n",
			(double)total_flop_n3/1000000000.0f);
	fprintf(stderr, "       GFlop : real %2.2f\n",
			(double)total_flop/1000000000.0f);
	fprintf(stderr, "	CPU : %2.2f (%2.2f%%)\n", (double)cpus_flop/1000000000.0, (100.0*cpus_flop)/(cpus_flop + cublas_flop));
	fprintf(stderr, "	GPU : %2.2f (%2.2f%%)\n", (double)cublas_flop/1000000000.0, (100.0*cublas_flop)/(cpus_flop + cublas_flop));
	fprintf(stderr, "       GFlop/s : %2.2f\n", (double)total_flop / (double)timing/1000);
}

static void mult_common_codelet(void *descr[], int s, __attribute__((unused))  void *arg)
{
	float *center 	= (float *)STARPU_GET_BLAS_PTR(descr[0]);
	float *left 	= (float *)STARPU_GET_BLAS_PTR(descr[1]);
	float *right 	= (float *)STARPU_GET_BLAS_PTR(descr[2]);

	unsigned n = STARPU_GET_BLAS_NX(descr[0]);

	unsigned ld21 = STARPU_GET_BLAS_LD(descr[1]);
	unsigned ld12 = STARPU_GET_BLAS_LD(descr[2]);
	unsigned ld22 = STARPU_GET_BLAS_LD(descr[0]);

	double flop = 2.0*n*n*n;

#ifdef USE_CUDA
	cublasStatus cublasres;
#endif

	switch (s) {
		case 0:
			cpus_flop += flop;
			SGEMM("N", "N", n, n, n, 1.0f, right, ld21, left, ld12, 0.0f, center, ld22);
			break;
#ifdef USE_CUDA
		case 1:
			cublas_flop += flop;

			cublasSgemm('n', 'n', n, n, n, 1.0f, right, ld12, left, ld21, 0.0f, center, ld22);
			cublasres = cublasGetError();
			if (STARPU_UNLIKELY(cublasres))
				CUBLAS_REPORT_ERROR(cublasres);
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

#ifdef USE_CUDA
void mult_cublas_codelet(void *descr[], void *_args)
{
	mult_common_codelet(descr, 1, _args);
}
#endif

static void add_sub_common_codelet(void *descr[], int s, __attribute__((unused))  void *arg, float alpha)
{
	/* C = A op B */

	float *C 	= (float *)STARPU_GET_BLAS_PTR(descr[0]);
	float *A 	= (float *)STARPU_GET_BLAS_PTR(descr[1]);
	float *B 	= (float *)STARPU_GET_BLAS_PTR(descr[2]);

	unsigned n = STARPU_GET_BLAS_NX(descr[0]);

	unsigned ldA = STARPU_GET_BLAS_LD(descr[1]);
	unsigned ldB = STARPU_GET_BLAS_LD(descr[2]);
	unsigned ldC = STARPU_GET_BLAS_LD(descr[0]);

	double flop = 2.0*n*n;

	// TODO check dim ...

	unsigned line;
#ifdef USE_CUDA
	cublasStatus cublasres;
#endif

	switch (s) {
		case 0:
			cpus_flop += flop;
			for (line = 0; line < n; line++)
			{
				/* copy line A into C */
				SAXPY(n, 1.0f, &A[line*ldA], 1, &C[line*ldC], 1);
				/* add line B to C = A */
				SAXPY(n, alpha, &B[line*ldB], 1, &C[line*ldC], 1);
			}
			break;
#ifdef USE_CUDA
		case 1:
			cublas_flop += flop;
			for (line = 0; line < n; line++)
			{
				/* copy line A into C */
				cublasSaxpy(n, 1.0f, &A[line*ldA], 1, &C[line*ldC], 1);
				cublasres = cublasGetError();
				if (STARPU_UNLIKELY(cublasres))
					CUBLAS_REPORT_ERROR(cublasres);
				/* add line B to C = A */
				cublasSaxpy(n, alpha, &B[line*ldB], 1, &C[line*ldC], 1);
				cublasres = cublasGetError();
				if (STARPU_UNLIKELY(cublasres))
					CUBLAS_REPORT_ERROR(cublasres);
			}

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

#ifdef USE_CUDA
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

	float *C 	= (float *)STARPU_GET_BLAS_PTR(descr[0]);
	float *A 	= (float *)STARPU_GET_BLAS_PTR(descr[1]);

	unsigned n = STARPU_GET_BLAS_NX(descr[0]);

	unsigned ldA = STARPU_GET_BLAS_LD(descr[1]);
	unsigned ldC = STARPU_GET_BLAS_LD(descr[0]);

	double flop = 1.0*n*n;

	// TODO check dim ...
	
	unsigned line;

#ifdef USE_CUDA
	cublasStatus cublasres;
#endif

	switch (s) {
		case 0:
			cpus_flop += flop;
			for (line = 0; line < n; line++)
			{
				/* add line A to C */
				SAXPY(n, alpha, &A[line*ldA], 1, &C[line*ldC], 1);
			}
			break;
#ifdef USE_CUDA
		case 1:
			cublas_flop += flop;
			for (line = 0; line < n; line++)
			{
				/* add line A to C */
				cublasSaxpy(n, alpha, &A[line*ldA], 1, &C[line*ldC], 1);
				cublasres = cublasGetError();
				if (STARPU_UNLIKELY(cublasres))
					CUBLAS_REPORT_ERROR(cublasres);
			}
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

#ifdef USE_CUDA
void self_add_cublas_codelet(void *descr[], __attribute__((unused))  void *arg)
{
	self_add_sub_common_codelet(descr, 1, arg, 1.0f);
}

void self_sub_cublas_codelet(void *descr[], __attribute__((unused))  void *arg)
{
	self_add_sub_common_codelet(descr, 1, arg, -1.0f);
}
#endif

/* this codelet does nothing  */
void null_codelet(__attribute__((unused)) void *descr[],
		  __attribute__((unused))  void *arg)
{
}
