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

void SSCAL(int N, float alpha, float *X, int incX)
{
	cblas_sscal(N, alpha, X, incX);
}

void STRSM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int m, const int n,
                   const float alpha, const float *A, const int lda,
                   float *B, const int ldb)
{
	enum CBLAS_SIDE side_ = (toupper(side[0]) == 'L')?CblasLeft:CblasRight;
	enum CBLAS_UPLO uplo_ = (toupper(uplo[0]) == 'U')?CblasUpper:CblasLower;
	enum CBLAS_TRANSPOSE transa_ = (toupper(transa[0]) == 'N')?CblasNoTrans:CblasTrans;
	enum CBLAS_DIAG diag_ = (toupper(diag[0]) == 'N')?CblasNonUnit:CblasUnit;

	cblas_strsm(CblasColMajor, side_, uplo_, transa_, diag_, m, n, alpha, A, lda, B, ldb);
}

void SSYR (const char *uplo, const int n, const float alpha,
                  const float *x, const int incx, float *A, const int lda)
{
	enum CBLAS_UPLO uplo_ = (toupper(uplo[0]) == 'U')?CblasUpper:CblasLower;

	cblas_ssyr(CblasColMajor, uplo_, n, alpha, x, incx, A, lda); 
}

void SSYRK (const char *uplo, const char *trans, const int n,
                   const int k, const float alpha, const float *A,
                   const int lda, const float beta, float *C,
                   const int ldc)
{
	enum CBLAS_UPLO uplo_ = (toupper(uplo[0]) == 'U')?CblasUpper:CblasLower;
	enum CBLAS_TRANSPOSE trans_ = (toupper(trans[0]) == 'N')?CblasNoTrans:CblasTrans;
	
	cblas_ssyrk(CblasColMajor, uplo_, trans_, n, k, alpha, A, lda, beta, C, ldc); 
}

#elif GOTO

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

void SSCAL(int N, float alpha, float *X, int incX)
{
	sscal_(&N, &alpha, X, &incX);
}

void STRSM (const char *side, const char *uplo, const char *transa,
                   const char *diag, const int m, const int n,
                   const float alpha, const float *A, const int lda,
                   float *B, const int ldb)
{
	strsm_(side, uplo, transa, diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

void SSYR (const char *uplo, const int n, const float alpha,
                  const float *x, const int incx, float *A, const int lda)
{
	ssyr_(uplo, &n, &alpha, x, &incx, A, &lda); 
}

void SSYRK (const char *uplo, const char *trans, const int n,
                   const int k, const float alpha, const float *A,
                   const int lda, const float beta, float *C,
                   const int ldc)
{
	ssyrk_(uplo, trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc); 
}


#else
#error "no BLAS lib available..."
#endif
