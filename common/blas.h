#ifndef __BLAS_H__
#define __BLAS_H__

#include <cblas.h>

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

#endif // __BLAS_H__
