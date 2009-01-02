#ifndef __BLAS_H__
#define __BLAS_H__

#include <cblas.h>

void SGEMM(char *transa, char *transb, int M, int N, int K, float alpha, float *A, int lda, 
		float *B, int ldb, float beta, float *C, int ldc);
float SASUM(int N, float *X, int incX);

#endif // __BLAS_H__
