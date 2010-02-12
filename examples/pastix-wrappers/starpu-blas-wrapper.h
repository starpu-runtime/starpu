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

#ifndef __STARPU_BLAS_WRAPPER_H__
#define __STARPU_BLAS_WRAPPER_H__

#include "generated_model.h"

#define OVERHEAD	150000.0

static double transfer_time_dtoh(unsigned size)
{
	double latency = 0.0;
	double bandwith = 0.0;

	return latency + size*bandwith;
}

static double transfer_time_htod(unsigned size)
{
	double latency = 0.0;
	double bandwith = 0.0;

	return latency + size*bandwith;
}

/*GEMM CPU */

#define PERF_GEMM_CPU(i,j,k) (GEMM_CPU_A*(double)(i)*(double)(j)*(double)(k)+GEMM_CPU_B*(double)(i)*(double)(j)+GEMM_CPU_C*(double)(j)*(double)(k)+GEMM_CPU_D*(double)(i)+GEMM_CPU_E*(double)(j)+GEMM_CPU_F)

static double starpu_compute_contrib_compact_cpu_cost(starpu_buffer_descr *descr)
{
	unsigned nx0, ny0, ny2;
	nx0 = descr[0].handle->interface->blas.nx;
	ny0 = descr[0].handle->interface->blas.ny;
	ny2 = descr[2].handle->interface->blas.ny;

	return PERF_GEMM_CPU(nx0-ny0, ny2, ny0); 
}



/*GEMM GPU */

#define PERF_GEMM_GPU(i,j,k) (GEMM_GPU_A*(double)(i)*(double)(j)*(double)(k)+GEMM_GPU_B*(double)(i)*(double)(j)+GEMM_GPU_C*(double)(j)*(double)(k)+GEMM_GPU_D*(double)(i)+GEMM_GPU_E*(double)(j)+GEMM_GPU_F)

static double starpu_compute_contrib_compact_cuda_cost(starpu_buffer_descr *descr)
{
	unsigned nx0, ny0, ny2;
	nx0 = descr[0].handle->interface->blas.nx;
	ny0 = descr[0].handle->interface->blas.ny;
	ny2 = descr[2].handle->interface->blas.ny;

	return PERF_GEMM_GPU(nx0-ny0, ny2, ny0) + OVERHEAD; 
}


/*TRSM CPU */

#define PERF_TRSM_GPU(i,j)   (TRSM_GPU_A*(double)(i)*(double)(i)*(double)(j)+TRSM_GPU_B*(double)(i)+TRSM_GPU_C)

static double starpu_cblk_strsm_cuda_cost(starpu_buffer_descr *descr)
{
	unsigned nx, ny;
	nx = descr[0].handle->interface->blas.nx;
	ny = descr[0].handle->interface->blas.ny;

	return PERF_TRSM_GPU(nx-ny, ny) + OVERHEAD; 
}

/*TRSM CPU */

#define PERF_TRSM_CPU(i,j)   (TRSM_CPU_A*(double)(i)*(double)(i)*(double)(j)+TRSM_CPU_B*(double)(i)+TRSM_CPU_C)

static double starpu_cblk_strsm_cpu_cost(starpu_buffer_descr *descr)
{
	unsigned nx, ny;
	nx = descr[0].handle->interface->blas.nx;
	ny = descr[0].handle->interface->blas.ny;

	return PERF_TRSM_CPU(nx-ny, ny); 
}

void STARPU_INIT(void);
void STARPU_TERMINATE(void);
void STARPU_SGEMM (const char *transa, const char *transb, const int m,
                   const int n, const int k, const float alpha,
                   const float *A, const int lda, const float *B,
                   const int ldb, const float beta, float *C, const int ldc);
void STARPU_STRSM (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const int m, const int n, 
                   const float alpha, const float *A, const int lda,
                   float *B, const int ldb);

#endif // __STARPU_BLAS_WRAPPER_H__
