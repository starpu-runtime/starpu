/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <ctype.h>
#include <stdio.h>

#include "blas.h"

inline void STARPU_SGEMM(char *transa, char *transb, BLASINT M, BLASINT N, BLASINT K, 
			float alpha, const float *A, BLASINT lda, const float *B, BLASINT ldb, 
			float beta, float *C, BLASINT ldc)
{
	sgemm_64_(transa, transb, &M, &N, &K, &alpha,
			 A, &lda, B, &ldb,
			 &beta, C, &ldc);	
}

inline void STARPU_DGEMM(char *transa, char *transb, BLASINT M, BLASINT N, BLASINT K, 
			double alpha, double *A, BLASINT lda, double *B, BLASINT ldb, 
			double beta, double *C, BLASINT ldc)
{
	dgemm_64_(transa, transb, &M, &N, &K, &alpha,
			 A, &lda, B, &ldb,
			 &beta, C, &ldc);	
}


inline void STARPU_SGEMV(char *transa, BLASINT M, BLASINT N, float alpha, float *A, BLASINT lda,
		float *X, BLASINT incX, float beta, float *Y, BLASINT incY)
{
	sgemv_64_(transa, &M, &N, &alpha, A, &lda, X, &incX, &beta, Y, &incY);
}

inline void STARPU_DGEMV(char *transa, BLASINT M, BLASINT N, double alpha, double *A, BLASINT lda,
		double *X, BLASINT incX, double beta, double *Y, BLASINT incY)
{
	dgemv_64_(transa, &M, &N, &alpha, A, &lda, X, &incX, &beta, Y, &incY);
}

inline float STARPU_SASUM(BLASINT N, float *X, BLASINT incX)
{
	return sasum_64_(&N, X, &incX);
}

inline double STARPU_DASUM(BLASINT N, double *X, BLASINT incX)
{
	return dasum_64_(&N, X, &incX);
}

void STARPU_SSCAL(BLASINT N, float alpha, float *X, BLASINT incX)
{
	sscal_64_(&N, &alpha, X, &incX);
}

void STARPU_DSCAL(BLASINT N, double alpha, double *X, BLASINT incX)
{
	dscal_64_(&N, &alpha, X, &incX);
}

void STARPU_STRSM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const BLASINT m, const BLASINT n,
                   const float alpha, const float *A, const BLASINT lda,
                   float *B, const BLASINT ldb)
{
	strsm_64_(side, uplo, transa, diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

void STARPU_DTRSM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const BLASINT m, const BLASINT n,
                   const double alpha, const double *A, const BLASINT lda,
                   double *B, const BLASINT ldb)
{
	dtrsm_64_(side, uplo, transa, diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

void STARPU_SSYR (const char *uplo, const BLASINT n, const float alpha,
                  const float *x, const BLASINT incx, float *A, const BLASINT lda)
{
	ssyr_64_(uplo, &n, &alpha, x, &incx, A, &lda); 
}

void STARPU_SSYRK (const char *uplo, const char *trans, const BLASINT n,
                   const BLASINT k, const float alpha, const float *A,
                   const BLASINT lda, const float beta, float *C,
                   const BLASINT ldc)
{
	ssyrk_64_(uplo, trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc); 
}

void STARPU_SGER(const BLASINT m, const BLASINT n, const float alpha,
                  const float *x, const BLASINT incx, const float *y,
                  const BLASINT incy, float *A, const BLASINT lda)
{
	sger_64_(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
}

void STARPU_DGER(const BLASINT m, const BLASINT n, const double alpha,
                  const double *x, const BLASINT incx, const double *y,
                  const BLASINT incy, double *A, const BLASINT lda)
{
	dger_64_(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
}

void STARPU_STRSV (const char *uplo, const char *trans, const char *diag, 
                   const BLASINT n, const float *A, const BLASINT lda, float *x, 
                   const BLASINT incx)
{
	strsv_64_(uplo, trans, diag, &n, A, &lda, x, &incx);
}

void STARPU_STRMM(const char *side, const char *uplo, const char *transA,
                 const char *diag, const BLASINT m, const BLASINT n,
                 const float alpha, const float *A, const BLASINT lda,
                 float *B, const BLASINT ldb)
{
	strmm_64_(side, uplo, transA, diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

void STARPU_DTRMM(const char *side, const char *uplo, const char *transA,
                 const char *diag, const BLASINT m, const BLASINT n,
                 const double alpha, const double *A, const BLASINT lda,
                 double *B, const BLASINT ldb)
{
	dtrmm_64_(side, uplo, transA, diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

void STARPU_STRMV(const char *uplo, const char *transA, const char *diag,
                 const BLASINT n, const float *A, const BLASINT lda, float *X,
                 const BLASINT incX)
{
	strmv_64_(uplo, transA, diag, &n, A, &lda, X, &incX);
}

void STARPU_SAXPY(const BLASINT n, const float alpha, float *X, const BLASINT incX, float *Y, const BLASINT incY)
{
	saxpy_64_(&n, &alpha, X, &incX, Y, &incY);
}

void STARPU_DAXPY(const BLASINT n, const double alpha, double *X, const BLASINT incX, double *Y, const BLASINT incY)
{
	daxpy_64_(&n, &alpha, X, &incX, Y, &incY);
}

BLASINT STARPU_ISAMAX (const BLASINT n, float *X, const BLASINT incX)
{
    BLASINT retVal;
    retVal = isamax_64_ (&n, X, &incX);
    return retVal;
}

BLASINT STARPU_IDAMAX (const BLASINT n, double *X, const BLASINT incX)
{
    BLASINT retVal;
    retVal = idamax_64_ (&n, X, &incX);
    return retVal;
}

float STARPU_SDOT(const BLASINT n, const float *x, const BLASINT incx, const float *y, const BLASINT incy)
{
	float retVal = 0;

	/* GOTOBLAS will return a FLOATRET which is a double, not a float */
	retVal = (float)sdot_64_(&n, x, &incx, y, &incy);

	return retVal;
}

double STARPU_DDOT(const BLASINT n, const double *x, const BLASINT incx, const double *y, const BLASINT incy)
{
	return ddot_64_(&n, x, &incx, y, &incy);
}

void STARPU_SSWAP(const BLASINT n, float *X, const BLASINT incX, float *Y, const BLASINT incY)
{
	sswap_64_(&n, X, &incX, Y, &incY);
}

void STARPU_DSWAP(const BLASINT n, double *X, const BLASINT incX, double *Y, const BLASINT incY)
{
	dswap_64_(&n, X, &incX, Y, &incY);
}
