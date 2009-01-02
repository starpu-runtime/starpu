#include <common/blas.h>
#include <ctype.h>
#include <stdio.h>

/*
    This files contains BLAS wrappers for the different BLAS implementations
  (eg. REFBLAS, ATLAS, GOTOBLAS ...). We assume a Fortran orientation as most
  libraries do not supply C-based ordering.
 */

#ifdef ATLAS

inline void SGEMM(char *transa, char *transb, int M, int N, int K, 
			float alpha, float *A, int lda, float *B, int ldb, 
			float beta, float *C, int ldc)
{
	enum CBLAS_TRANSPOSE ta = (toupper(transa[0]) == 'N')?CblasNoTrans:CblasTrans;
	enum CBLAS_TRANSPOSE tb = (toupper(transb[0]) == 'N')?CblasNoTrans:CblasTrans;

	cblas_sgemm(CblasColMajor, ta, tb,
			M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);				
}

inline float SASUM(int N, float *X, int incX)
{
	return cblas_sasum(N, X, incX);
}

#else
#ifdef GOTO

inline void SGEMM(char *transa, char *transb, int M, int N, int K, 
			float alpha, float *A, int lda, float *B, int ldb, 
			float beta, float *C, int ldc)
{
	sgemm_(transa, transb, &M, &N, &K, &alpha,
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



