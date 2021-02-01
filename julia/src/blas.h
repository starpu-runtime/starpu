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

#ifndef __BLAS_H__
#define __BLAS_H__

#include <stdint.h>

#define BLASINT int64_t

void STARPU_SGEMM(char *transa, char *transb, BLASINT M, BLASINT N, BLASINT K, float alpha, const float *A, BLASINT lda, 
		const float *B, BLASINT ldb, float beta, float *C, BLASINT ldc);
void STARPU_DGEMM(char *transa, char *transb, BLASINT M, BLASINT N, BLASINT K, double alpha, double *A, BLASINT lda, 
		double *B, BLASINT ldb, double beta, double *C, BLASINT ldc);
void STARPU_SGEMV(char *transa, BLASINT M, BLASINT N, float alpha, float *A, BLASINT lda,
		float *X, BLASINT incX, float beta, float *Y, BLASINT incY);
void STARPU_DGEMV(char *transa, BLASINT M, BLASINT N, double alpha, double *A, BLASINT lda,
		double *X, BLASINT incX, double beta, double *Y, BLASINT incY);
float STARPU_SASUM(BLASINT N, float *X, BLASINT incX);
double STARPU_DASUM(BLASINT N, double *X, BLASINT incX);
void STARPU_SSCAL(BLASINT N, float alpha, float *X, BLASINT incX);
void STARPU_DSCAL(BLASINT N, double alpha, double *X, BLASINT incX);
void STARPU_STRSM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const BLASINT m, const BLASINT n,
                   const float alpha, const float *A, const BLASINT lda,
                   float *B, const BLASINT ldb);
void STARPU_DTRSM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const BLASINT m, const BLASINT n,
                   const double alpha, const double *A, const BLASINT lda,
                   double *B, const BLASINT ldb);
void STARPU_SSYR (const char *uplo, const BLASINT n, const float alpha,
                  const float *x, const BLASINT incx, float *A, const BLASINT lda);
void STARPU_SSYRK (const char *uplo, const char *trans, const BLASINT n,
                   const BLASINT k, const float alpha, const float *A,
                   const BLASINT lda, const float beta, float *C,
                   const BLASINT ldc);
void STARPU_SGER (const BLASINT m, const BLASINT n, const float alpha,
                  const float *x, const BLASINT incx, const float *y,
                  const BLASINT incy, float *A, const BLASINT lda);
void STARPU_DGER(const BLASINT m, const BLASINT n, const double alpha,
                  const double *x, const BLASINT incx, const double *y,
                  const BLASINT incy, double *A, const BLASINT lda);
void STARPU_STRSV (const char *uplo, const char *trans, const char *diag, 
                   const BLASINT n, const float *A, const BLASINT lda, float *x, 
                   const BLASINT incx);
void STARPU_STRMM(const char *side, const char *uplo, const char *transA,
                 const char *diag, const BLASINT m, const BLASINT n,
                 const float alpha, const float *A, const BLASINT lda,
                 float *B, const BLASINT ldb);
void STARPU_DTRMM(const char *side, const char *uplo, const char *transA,
                 const char *diag, const BLASINT m, const BLASINT n,
                 const double alpha, const double *A, const BLASINT lda,
                 double *B, const BLASINT ldb);
void STARPU_STRMV(const char *uplo, const char *transA, const char *diag,
                 const BLASINT n, const float *A, const BLASINT lda, float *X,
                 const BLASINT incX);
void STARPU_SAXPY(const BLASINT n, const float alpha, float *X, const BLASINT incX, float *Y, const BLASINT incy);
void STARPU_DAXPY(const BLASINT n, const double alpha, double *X, const BLASINT incX, double *Y, const BLASINT incY);
BLASINT STARPU_ISAMAX (const BLASINT n, float *X, const BLASINT incX);
BLASINT STARPU_IDAMAX (const BLASINT n, double *X, const BLASINT incX);
float STARPU_SDOT(const BLASINT n, const float *x, const BLASINT incx, const float *y, const BLASINT incy);
double STARPU_DDOT(const BLASINT n, const double *x, const BLASINT incx, const double *y, const BLASINT incy);
void STARPU_SSWAP(const BLASINT n, float *x, const BLASINT incx, float *y, const BLASINT incy);
void STARPU_DSWAP(const BLASINT n, double *x, const BLASINT incx, double *y, const BLASINT incy);


extern void sgemm_64_ (const char *transa, const char *transb, const BLASINT *m,
                   const BLASINT *n, const BLASINT *k, const float *alpha, 
                   const float *A, const BLASINT *lda, const float *B, 
                   const BLASINT *ldb, const float *beta, float *C, 
                   const BLASINT *ldc);
extern void dgemm_64_ (const char *transa, const char *transb, const BLASINT *m,
                   const BLASINT *n, const BLASINT *k, const double *alpha, 
                   const double *A, const BLASINT *lda, const double *B, 
                   const BLASINT *ldb, const double *beta, double *C, 
                   const BLASINT *ldc);
extern void sgemv_64_(const char *trans, const BLASINT *m, const BLASINT *n, const float *alpha,
                   const float *a, const BLASINT *lda, const float *x, const BLASINT *incx, 
                   const float *beta, float *y, const BLASINT *incy);
extern void dgemv_64_(const char *trans, const BLASINT *m, const BLASINT *n, const double *alpha,
                   const double *a, const BLASINT *lda, const double *x, const BLASINT *incx,
                   const double *beta, double *y, const BLASINT *incy);
extern void ssyr_64_ (const char *uplo, const BLASINT *n, const float *alpha,
                  const float *x, const BLASINT *incx, float *A, const BLASINT *lda);
extern void ssyrk_64_ (const char *uplo, const char *trans, const BLASINT *n,
                   const BLASINT *k, const float *alpha, const float *A,
                   const BLASINT *lda, const float *beta, float *C,
                   const BLASINT *ldc);
extern void strsm_64_ (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const BLASINT *m, const BLASINT *n,
                   const float *alpha, const float *A, const BLASINT *lda,
                   float *B, const BLASINT *ldb);
extern void dtrsm_64_ (const char *side, const char *uplo, const char *transa, 
                   const char *diag, const BLASINT *m, const BLASINT *n,
                   const double *alpha, const double *A, const BLASINT *lda,
                   double *B, const BLASINT *ldb);
extern double sasum_64_ (const BLASINT *n, const float *x, const BLASINT *incx);
extern double dasum_64_ (const BLASINT *n, const double *x, const BLASINT *incx);
extern void sscal_64_ (const BLASINT *n, const float *alpha, float *x,
                   const BLASINT *incx);
extern void dscal_64_ (const BLASINT *n, const double *alpha, double *x,
                   const BLASINT *incx);
extern void sger_64_(const BLASINT *m, const BLASINT *n, const float *alpha,
                  const float *x, const BLASINT *incx, const float *y,
                  const BLASINT *incy, float *A, const BLASINT *lda);
extern void dger_64_(const BLASINT *m, const BLASINT *n, const double *alpha,
                  const double *x, const BLASINT *incx, const double *y,
                  const BLASINT *incy, double *A, const BLASINT *lda);
extern void strsv_64_ (const char *uplo, const char *trans, const char *diag, 
                   const BLASINT *n, const float *A, const BLASINT *lda, float *x, 
                   const BLASINT *incx);
extern void strmm_64_(const char *side, const char *uplo, const char *transA,
                 const char *diag, const BLASINT *m, const BLASINT *n,
                 const float *alpha, const float *A, const BLASINT *lda,
                 float *B, const BLASINT *ldb);
extern void dtrmm_64_(const char *side, const char *uplo, const char *transA,
                 const char *diag, const BLASINT *m, const BLASINT *n,
                 const double *alpha, const double *A, const BLASINT *lda,
                 double *B, const BLASINT *ldb);
extern void strmv_64_(const char *uplo, const char *transA, const char *diag,
                 const BLASINT *n, const float *A, const BLASINT *lda, float *X,
                 const BLASINT *incX);
extern void saxpy_64_(const BLASINT *n, const float *alpha, const float *X, const BLASINT *incX,
		float *Y, const BLASINT *incy);
extern void daxpy_64_(const BLASINT *n, const double *alpha, const double *X, const BLASINT *incX,
		double *Y, const BLASINT *incy);
extern BLASINT isamax_64_(const BLASINT *n, const float *X, const BLASINT *incX);
extern BLASINT idamax_64_(const BLASINT *n, const double *X, const BLASINT *incX);
/* for some reason, FLOATRET is not a float but a double in GOTOBLAS */
extern double sdot_64_(const BLASINT *n, const float *x, const BLASINT *incx, const float *y, const BLASINT *incy);
extern double ddot_64_(const BLASINT *n, const double *x, const BLASINT *incx, const double *y, const BLASINT *incy);
extern void sswap_64_(const BLASINT *n, float *x, const BLASINT *incx, float *y, const BLASINT *incy);
extern void dswap_64_(const BLASINT *n, double *x, const BLASINT *incx, double *y, const BLASINT *incy);

#endif /* __BLAS_H__ */
