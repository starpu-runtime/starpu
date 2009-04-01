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

#ifndef __BLAS_H__
#define __BLAS_H__

#include <starpu.h>

#ifdef ATLAS
#include <cblas.h>
#endif

void SGEMM(char *transa, char *transb, int M, int N, int K, float alpha, float *A, int lda, 
		float *B, int ldb, float beta, float *C, int ldc);
float SASUM(int N, float *X, int incX);
void SSCAL(int N, float alpha, float *X, int incX);
void STRSM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int m, const int n,
                   const float alpha, const float *A, const int lda,
                   float *B, const int ldb);
void SSYR (const char *uplo, const int n, const float alpha,
                  const float *x, const int incx, float *A, const int lda);
void SSYRK (const char *uplo, const char *trans, const int n,
                   const int k, const float alpha, const float *A,
                   const int lda, const float beta, float *C,
                   const int ldc);
void SGER (const int m, const int n, const float alpha,
                  const float *x, const int incx, const float *y,
                  const int incy, float *A, const int lda);
void STRSV (const char *uplo, const char *trans, const char *diag, 
                   const int n, const float *A, const int lda, float *x, 
                   const int incx);
void STRMM(const char *side, const char *uplo, const char *transA,
                 const char *diag, const int m, const int n,
                 const float alpha, const float *A, const int lda,
                 float *B, const int ldb);
void STRMV(const char *uplo, const char *transA, const char *diag,
                 const int n, const float *A, const int lda, float *X,
                 const int incX);
void SAXPY(const int n, const float alpha, float *X, const int incX, float *Y, const int incy);
int ISAMAX (const int n, float *X, const int incX);
float SDOT(const int n, const float *x, const int incx, const float *y, const int incy);

#if defined(GOTO) || defined(SYSTEM_BLAS)

extern void sgemm_ (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const float *alpha, 
                   const float *A, const int *lda, const float *B, 
                   const int *ldb, const float *beta, float *C, 
                   const int *ldc);
extern void ssyr_ (const char *uplo, const int *n, const float *alpha,
                  const float *x, const int *incx, float *A, const int *lda);
extern void ssyrk_ (const char *uplo, const char *trans, const int *n,
                   const int *k, const float *alpha, const float *A,
                   const int *lda, const float *beta, float *C,
                   const int *ldc);
extern void strsm_ (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const int *m, const int *n,
                   const float *alpha, const float *A, const int *lda,
                   float *B, const int *ldb);
extern double sasum_ (const int *n, const float *x, const int *incx);
extern void sscal_ (const int *n, const float *alpha, float *x,
                   const int *incx);
extern void sger_(const int *m, const int *n, const float *alpha,
                  const float *x, const int *incx, const float *y,
                  const int *incy, float *A, const int *lda);
extern void strsv_ (const char *uplo, const char *trans, const char *diag, 
                   const int *n, const float *A, const int *lda, float *x, 
                   const int *incx);
extern void strmm_(const char *side, const char *uplo, const char *transA,
                 const char *diag, const int *m, const int *n,
                 const float *alpha, const float *A, const int *lda,
                 float *B, const int *ldb);
extern void strmv_(const char *uplo, const char *transA, const char *diag,
                 const int *n, const float *A, const int *lda, float *X,
                 const int *incX);
extern void saxpy_(const int *n, const float *alpha, float *X, const int *incX,
		float *Y, const int *incy);
extern int isamax_(const int *n, float *X, const int *incX);
/* for some reason, FLOATRET is not a float but a double in GOTOBLAS */
extern double sdot_(const int *n, const float *x, const int *incx, const float *y, const int *incy);
#endif

#endif // __BLAS_H__
