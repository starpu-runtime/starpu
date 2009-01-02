#include <common/blas.h>

/*
    This files contains BLAS wrappers for the different BLAS implementations
  (eg. REFBLAS, ATLAS, GOTOBLAS ...). We assume a Fortran orientation as most
  libraries do not supply C-based ordering.
 */

#ifdef ATLAS

inline void SGEMM(int M, int N, int K, 
			float alpha, float *A, int lda, float *B, int ldb, 
			float beta, float *C, int ldc)
{
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
			M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);				
}

inline float SASUM(int N, float *X, int incX)
{
	return cblas_sasum(N, X, incX);
}

#else
#ifdef GOTO

inline void SGEMM(int M, int N, int K, 
			float alpha, float *A, int lda, float *B, int ldb, 
			float beta, float *C, int ldc)
{
	sgemm_("N", "N", &M, &N, &K, &alpha,
			 A, &lda, B, &ldb,
			 &beta, C, &ldc);	
}

inline float SASUM(int N, float *X, int incX)
{
	return sasum_(&N, X, &incX);
}

#else
#error "no BLAS lib available..."
#endif
#endif



