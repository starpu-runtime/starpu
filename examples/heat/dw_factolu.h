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

#ifndef __DW_FACTO_LU_H__
#define __DW_FACTO_LU_H__

#include <semaphore.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
/* for STARPU_USE_CUDA */
#include <starpu_config.h>
#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>
#endif

#include "../common/blas.h"

#include <starpu.h>

#include "lu_kernels_model.h"

#define BLAS3_FLOP(n1,n2,n3)    \
        (2*((uint64_t)n1)*((uint64_t)n2)*((uint64_t)n3))

typedef struct {
	starpu_data_handle dataA;
	unsigned i;
	unsigned j;
	unsigned k;
	unsigned nblocks;
	unsigned *remaining;
} cl_args;

#ifdef CHECK_RESULTS
static void __attribute__ ((unused)) compare_A_LU(float *A, float *LU,
				unsigned size, unsigned ld)
{
	unsigned i,j;
	float *L;
	float *U;

	L = malloc(size*size*sizeof(float));
	U = malloc(size*size*sizeof(float));

	memset(L, 0, size*size*sizeof(float));
	memset(U, 0, size*size*sizeof(float));

	/* only keep the lower part */
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < j; i++)
		{
			L[j+i*size] = LU[j+i*ld];
		}

		/* diag i = j */
		L[j+j*size] = LU[j+j*ld];
		U[j+j*size] = 1.0f;

		for (i = j+1; i < size; i++)
		{
			U[j+i*size] = LU[j+i*ld];
		}
	}

#if 0
	/* display L */
	printf("(LU): \n");
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
//			if (i <= j) {
				printf("%2.2f\t", LU[j +i*size]);
//			}
//			else {
//				printf(".\t");
//			}
		}
		printf("\n");
	}



	/* display L */
	printf("L: \n");
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
//			if (i <= j) {
				printf("%2.2f\t", L[j +i*size]);
//			}
//			else {
//				printf(".\t");
//			}
		}
		printf("\n");
	}

	/* display U */
	printf("U: \n");
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
//			if (i <= j) {
				printf("%2.2f\t", U[j +i*size]);
//			}
//			else {
//				printf(".\t");
//			}
		}
		printf("\n");
	}

#endif


        /* now A_err = L, compute L*U */
	STRMM("R", "U", "N", "U", size, size, 1.0f, U, size, L, size);

	float max_err = 0.0f;
	for (i = 0; i < size ; i++)
	{
		for (j = 0; j < size; j++) 
		{
			max_err = STARPU_MAX(max_err, fabs(  L[j+i*size] - A[j+i*ld]  ));
		}
	}

#if 0
	/* display A */
	printf("A: \n");
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
	//		if (i <= j) {
	      			printf("%2.2f\t", A[j +i*size]);
	//		}
	//		else {
	//			printf(".\t");
	//		}
		}
		printf("\n");
	}


	/* display LU */
	printf("LU: \n");
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
	//		if (i <= j) {
	      			printf("%2.2f\t", L[j +i*size]);
	//		}
	//		else {
	//			printf(".\t");
	//		}
		}
		printf("\n");
	}
#endif

	printf("max error between A and L*U = %f \n", max_err);
}
#endif // CHECK_RESULTS

void dw_cpu_codelet_update_u11(void **, void *);
void dw_cpu_codelet_update_u12(void **, void *);
void dw_cpu_codelet_update_u21(void **, void *);
void dw_cpu_codelet_update_u22(void **, void *);

#ifdef STARPU_USE_CUDA
void dw_cublas_codelet_update_u11(void *descr[], void *_args);
void dw_cublas_codelet_update_u12(void *descr[], void *_args);
void dw_cublas_codelet_update_u21(void *descr[], void *_args);
void dw_cublas_codelet_update_u22(void *descr[], void *_args);
#endif

void dw_callback_codelet_update_u11(void *);
void dw_callback_codelet_update_u12_21(void *);
void dw_callback_codelet_update_u22(void *);

void dw_callback_v2_codelet_update_u11(void *);
void dw_callback_v2_codelet_update_u12(void *);
void dw_callback_v2_codelet_update_u21(void *);
void dw_callback_v2_codelet_update_u22(void *);

extern struct starpu_perfmodel_t model_11;
extern struct starpu_perfmodel_t model_12;
extern struct starpu_perfmodel_t model_21;
extern struct starpu_perfmodel_t model_22;

#endif // __DW_FACTO_LU_H__
