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

#ifndef __XLU_H__
#define __XLU_H__

/* for USE_CUDA */
#include <starpu_config.h>
#include <starpu.h>

#include <common/blas.h>

#define BLAS3_FLOP(n1,n2,n3)    \
        (2*((uint64_t)n1)*((uint64_t)n2)*((uint64_t)n3))

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

	printf("max error between A and L*U = %f \n", max_err);
}
#endif // CHECK_RESULTS

void dw_core_codelet_update_u11(starpu_data_interface_t *, void *);
void dw_core_codelet_update_u12(starpu_data_interface_t *, void *);
void dw_core_codelet_update_u21(starpu_data_interface_t *, void *);
void dw_core_codelet_update_u22(starpu_data_interface_t *, void *);

#ifdef USE_CUDA
void dw_cublas_codelet_update_u11(starpu_data_interface_t *descr, void *_args);
void dw_cublas_codelet_update_u12(starpu_data_interface_t *descr, void *_args);
void dw_cublas_codelet_update_u21(starpu_data_interface_t *descr, void *_args);
void dw_cublas_codelet_update_u22(starpu_data_interface_t *descr, void *_args);
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

struct piv_s {
	unsigned *piv; /* complete pivot array */
	unsigned first; /* first element */
	unsigned last; /* last element */
};

#endif // __XLU_H__
