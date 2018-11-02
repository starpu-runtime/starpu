#include "f2c.h"
#include "fblaswr.h"

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions (complex are recast as routines)
 * ===========================================================================
 */

doublereal 
f2c_sdot(integer* N, 
         real* X, integer* incX, 
         real* Y, integer* incY)
{
    return _starpu_sdot_(N, X, incX, Y, incY);
}

doublereal 
f2c_ddot(integer* N, 
         doublereal* X, integer* incX, 
         doublereal* Y, integer* incY)
{
    return _starpu_ddot_(N, X, incX, Y, incY);
}


/*
 * Functions having prefixes Z and C only
 */

void
f2c_cdotu(complex* retval,
          integer* N, 
          complex* X, integer* incX, 
          complex* Y, integer* incY)
{
    _starpu_cdotu_(retval, N, X, incX, Y, incY);
}

void
f2c_cdotc(complex* retval,
          integer* N, 
          complex* X, integer* incX, 
          complex* Y, integer* incY)
{
    _starpu_cdotc_(retval, N, X, incX, Y, incY);
}

void
f2c_zdotu(doublecomplex* retval,
          integer* N, 
          doublecomplex* X, integer* incX, 
          doublecomplex* Y, integer* incY)
{
    _starpu_zdotu_(retval, N, X, incX, Y, incY);
}

void
f2c_zdotc(doublecomplex* retval,
          integer* N, 
          doublecomplex* X, integer* incX, 
          doublecomplex* Y, integer* incY)
{
    _starpu_zdotc_(retval, N, X, incX, Y, incY);
}


/*
 * Functions having prefixes S D SC DZ
 */

doublereal 
f2c_snrm2(integer* N, 
          real* X, integer* incX)
{
    return _starpu_snrm2_(N, X, incX);
}

doublereal
f2c_sasum(integer* N, 
          real* X, integer* incX)
{
    return _starpu_sasum_(N, X, incX);
}

doublereal 
f2c_dnrm2(integer* N, 
          doublereal* X, integer* incX)
{
    return _starpu_dnrm2_(N, X, incX);
}

doublereal
f2c_dasum(integer* N, 
          doublereal* X, integer* incX)
{
    return _starpu_dasum_(N, X, incX);
}

doublereal 
f2c_scnrm2(integer* N, 
           complex* X, integer* incX)
{
    return _starpu_scnrm2_(N, X, incX);
}

doublereal
f2c_scasum(integer* N, 
           complex* X, integer* incX)
{
    return _starpu_scasum_(N, X, incX);
}

doublereal 
f2c_dznrm2(integer* N, 
           doublecomplex* X, integer* incX)
{
    return _starpu_dznrm2_(N, X, incX);
}

doublereal
f2c_dzasum(integer* N, 
           doublecomplex* X, integer* incX)
{
    return _starpu_dzasum_(N, X, incX);
}


/*
 * Functions having standard 4 prefixes (S D C Z)
 */
integer
f2c_isamax(integer* N,
           real* X, integer* incX)
{
    return _starpu_isamax_(N, X, incX);
}

integer
f2c_idamax(integer* N,
           doublereal* X, integer* incX)
{
    return _starpu_idamax_(N, X, incX);
}

integer
f2c_icamax(integer* N,
           complex* X, integer* incX)
{
    return _starpu_icamax_(N, X, incX);
}

integer
f2c_izamax(integer* N,
           doublecomplex* X, integer* incX)
{
    return _starpu_izamax_(N, X, incX);
}

/*
 * ===========================================================================
 * Prototypes for level 0 BLAS routines
 * ===========================================================================
 */
int
f2c_srotg(real* a,
	      real* b,
		  real* c,
		  real* s)
{
    _starpu_srotg_(a, b, c, s);
    return 0;
}

int
f2c_crotg(complex* CA,
          complex* CB,
          complex* C,
          real* S)
{
    _starpu_crotg_(CA, CB, C, S);
    return 0;
}

int
f2c_drotg(doublereal* a,
		  doublereal* b,
		  doublereal* c,
		  doublereal* s)
{
    _starpu_drotg_(a, b, c, s);
    return 0;
}

int
f2c_zrotg(doublecomplex* CA,
          doublecomplex* CB,
          doublecomplex* C,
          doublereal* S)
{
    _starpu_zrotg_(CA, CB, C, S);
    return 0;
}
/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (s, d, c, z)
 */

int
f2c_sswap(integer* N,
          real* X, integer* incX,
          real* Y, integer* incY)
{
    _starpu_sswap_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_scopy(integer* N,
          real* X, integer* incX,
          real* Y, integer* incY)
{
    _starpu_scopy_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_saxpy(integer* N,
          real* alpha,
          real* X, integer* incX,
          real* Y, integer* incY)
{
    _starpu_saxpy_(N, alpha, X, incX, Y, incY);
    return 0;
}

int
f2c_dswap(integer* N,
          doublereal* X, integer* incX,
          doublereal* Y, integer* incY)
{
    _starpu_dswap_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_dcopy(integer* N,
          doublereal* X, integer* incX,
          doublereal* Y, integer* incY)
{
    _starpu_dcopy_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_daxpy(integer* N,
          doublereal* alpha,
          doublereal* X, integer* incX,
          doublereal* Y, integer* incY)
{
    _starpu_daxpy_(N, alpha, X, incX, Y, incY);
    return 0;
}

int
f2c_cswap(integer* N,
          complex* X, integer* incX,
          complex* Y, integer* incY)
{
    _starpu_cswap_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_ccopy(integer* N,
          complex* X, integer* incX,
          complex* Y, integer* incY)
{
    _starpu_ccopy_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_caxpy(integer* N,
          complex* alpha,
          complex* X, integer* incX,
          complex* Y, integer* incY)
{
    _starpu_caxpy_(N, alpha, X, incX, Y, incY);
    return 0;
}

int
f2c_zswap(integer* N,
          doublecomplex* X, integer* incX,
          doublecomplex* Y, integer* incY)
{
    _starpu_zswap_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_zcopy(integer* N,
          doublecomplex* X, integer* incX,
          doublecomplex* Y, integer* incY)
{
    _starpu_zcopy_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_zaxpy(integer* N,
          doublecomplex* alpha,
          doublecomplex* X, integer* incX,
          doublecomplex* Y, integer* incY)
{
    _starpu_zaxpy_(N, alpha, X, incX, Y, incY);
    return 0;
}


/*
 * Routines with S and D prefix only
 */

int
f2c_srot(integer* N,
         real* X, integer* incX,
         real* Y, integer* incY,
         real* c, real* s)
{
    _starpu_srot_(N, X, incX, Y, incY, c, s);
    return 0;
}

int
f2c_drot(integer* N,
         doublereal* X, integer* incX,
         doublereal* Y, integer* incY,
         doublereal* c, doublereal* s)
{
    _starpu_drot_(N, X, incX, Y, incY, c, s);
    return 0;
}


/*
 * Routines with S D C Z CS and ZD prefixes
 */

int
f2c_sscal(integer* N,
          real* alpha,
          real* X, integer* incX)
{
    _starpu_sscal_(N, alpha, X, incX);
    return 0;
}

int
f2c_dscal(integer* N,
          doublereal* alpha,
          doublereal* X, integer* incX)
{
    _starpu_dscal_(N, alpha, X, incX);
    return 0;
}

int
f2c_cscal(integer* N,
          complex* alpha,
          complex* X, integer* incX)
{
    _starpu_cscal_(N, alpha, X, incX);
    return 0;
}


int
f2c_zscal(integer* N,
          doublecomplex* alpha,
          doublecomplex* X, integer* incX)
{
    _starpu_zscal_(N, alpha, X, incX);
    return 0;
}


int
f2c_csscal(integer* N,
           real* alpha,
           complex* X, integer* incX)
{
    _starpu_csscal_(N, alpha, X, incX);
    return 0;
}


int
f2c_zdscal(integer* N,
           doublereal* alpha,
           doublecomplex* X, integer* incX)
{
    _starpu_zdscal_(N, alpha, X, incX);
    return 0;
}



/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
int
f2c_sgemv(char* trans, integer* M, integer* N,
          real* alpha,
          real* A, integer* lda,
          real* X, integer* incX,
          real* beta,
          real* Y, integer* incY)
{
    _starpu_sgemv_(trans, M, N,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_sgbmv(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
          real *alpha, 
          real *A, integer *lda, 
          real *X, integer *incX, 
          real *beta, 
          real *Y, integer *incY)
{
    _starpu_sgbmv_(trans, M, N, KL, KU,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_strmv(char* uplo, char *trans, char* diag, integer *N,  
          real *A, integer *lda, 
          real *X, integer *incX)
{
    _starpu_strmv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_stbmv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          real* A, integer* lda,
          real* X, integer* incX)
{
    _starpu_stbmv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_stpmv(char* uplo, char* trans, char* diag, integer* N, 
          real* Ap, 
          real* X, integer* incX)
{
    _starpu_stpmv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
}

int
f2c_strsv(char* uplo, char* trans, char* diag, integer* N,
          real* A, integer* lda,
          real* X, integer* incX)
{
    _starpu_strsv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_stbsv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          real* A, integer* lda, 
          real* X, integer* incX)
{
    _starpu_stbsv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_stpsv(char* uplo, char* trans, char* diag, integer* N, 
          real* Ap, 
          real* X, integer* incX)
{
    _starpu_stpsv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
} 



int
f2c_dgemv(char* trans, integer* M, integer* N,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* X, integer* incX,
          doublereal* beta,
          doublereal* Y, integer* incY)
{
    _starpu_dgemv_(trans, M, N,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_dgbmv(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
          doublereal *alpha, 
          doublereal *A, integer *lda, 
          doublereal *X, integer *incX, 
          doublereal *beta, 
          doublereal *Y, integer *incY)
{
    _starpu_dgbmv_(trans, M, N, KL, KU,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_dtrmv(char* uplo, char *trans, char* diag, integer *N,  
          doublereal *A, integer *lda, 
          doublereal *X, integer *incX)
{
    _starpu_dtrmv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_dtbmv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          doublereal* A, integer* lda,
          doublereal* X, integer* incX)
{
    _starpu_dtbmv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_dtpmv(char* uplo, char* trans, char* diag, integer* N, 
          doublereal* Ap, 
          doublereal* X, integer* incX)
{
    _starpu_dtpmv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
}

int
f2c_dtrsv(char* uplo, char* trans, char* diag, integer* N,
          doublereal* A, integer* lda,
          doublereal* X, integer* incX)
{
    _starpu_dtrsv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_dtbsv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          doublereal* A, integer* lda, 
          doublereal* X, integer* incX)
{
    _starpu_dtbsv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_dtpsv(char* uplo, char* trans, char* diag, integer* N, 
          doublereal* Ap, 
          doublereal* X, integer* incX)
{
    _starpu_dtpsv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
} 



int
f2c_cgemv(char* trans, integer* M, integer* N,
          complex* alpha,
          complex* A, integer* lda,
          complex* X, integer* incX,
          complex* beta,
          complex* Y, integer* incY)
{
    _starpu_cgemv_(trans, M, N,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_cgbmv(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
          complex *alpha, 
          complex *A, integer *lda, 
          complex *X, integer *incX, 
          complex *beta, 
          complex *Y, integer *incY)
{
    _starpu_cgbmv_(trans, M, N, KL, KU,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_ctrmv(char* uplo, char *trans, char* diag, integer *N,  
          complex *A, integer *lda, 
          complex *X, integer *incX)
{
    _starpu_ctrmv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_ctbmv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          complex* A, integer* lda,
          complex* X, integer* incX)
{
    _starpu_ctbmv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_ctpmv(char* uplo, char* trans, char* diag, integer* N, 
          complex* Ap, 
          complex* X, integer* incX)
{
    _starpu_ctpmv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
}

int
f2c_ctrsv(char* uplo, char* trans, char* diag, integer* N,
          complex* A, integer* lda,
          complex* X, integer* incX)
{
    _starpu_ctrsv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_ctbsv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          complex* A, integer* lda, 
          complex* X, integer* incX)
{
    _starpu_ctbsv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_ctpsv(char* uplo, char* trans, char* diag, integer* N, 
          complex* Ap, 
          complex* X, integer* incX)
{
    _starpu_ctpsv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
} 



int
f2c_zgemv(char* trans, integer* M, integer* N,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* X, integer* incX,
          doublecomplex* beta,
          doublecomplex* Y, integer* incY)
{
    _starpu_zgemv_(trans, M, N,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_zgbmv(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
          doublecomplex *alpha, 
          doublecomplex *A, integer *lda, 
          doublecomplex *X, integer *incX, 
          doublecomplex *beta, 
          doublecomplex *Y, integer *incY)
{
    _starpu_zgbmv_(trans, M, N, KL, KU,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_ztrmv(char* uplo, char *trans, char* diag, integer *N,  
          doublecomplex *A, integer *lda, 
          doublecomplex *X, integer *incX)
{
    _starpu_ztrmv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_ztbmv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          doublecomplex* A, integer* lda,
          doublecomplex* X, integer* incX)
{
    _starpu_ztbmv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_ztpmv(char* uplo, char* trans, char* diag, integer* N, 
          doublecomplex* Ap, 
          doublecomplex* X, integer* incX)
{
    _starpu_ztpmv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
}

int
f2c_ztrsv(char* uplo, char* trans, char* diag, integer* N,
          doublecomplex* A, integer* lda,
          doublecomplex* X, integer* incX)
{
    _starpu_ztrsv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_ztbsv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          doublecomplex* A, integer* lda, 
          doublecomplex* X, integer* incX)
{
    _starpu_ztbsv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_ztpsv(char* uplo, char* trans, char* diag, integer* N, 
          doublecomplex* Ap, 
          doublecomplex* X, integer* incX)
{
    _starpu_ztpsv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
} 


/*
 * Routines with S and D prefixes only
 */

int
f2c_ssymv(char* uplo, integer* N,
          real* alpha,
          real* A, integer* lda,
          real* X, integer* incX,
          real* beta,
          real* Y, integer* incY)
{
    _starpu_ssymv_(uplo, N, alpha, A, lda, 
           X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_ssbmv(char* uplo, integer* N, integer* K,
          real* alpha,
          real* A, integer* lda,
          real* X, integer* incX,
          real* beta,
          real* Y, integer* incY)
{
    _starpu_ssbmv_(uplo, N, K, alpha, A, lda, 
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_sspmv(char* uplo, integer* N,
          real* alpha,
          real* Ap,
          real* X, integer* incX,
          real* beta,
          real* Y, integer* incY)
{
    _starpu_sspmv_(uplo, N, alpha, Ap,  
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_sger(integer* M, integer* N,
         real* alpha,
         real* X, integer* incX,
         real* Y, integer* incY,
         real* A, integer* lda)
{
    _starpu_sger_(M, N, alpha,
          X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_ssyr(char* uplo, integer* N,
         real* alpha,
         real* X, integer* incX,
         real* A, integer* lda)
{
    _starpu_ssyr_(uplo, N, alpha, X, incX, A, lda);
    return 0;
}

int
f2c_sspr(char* uplo, integer* N,
         real* alpha,
         real* X, integer* incX,
         real* Ap)
{
    _starpu_sspr_(uplo, N, alpha, X, incX, Ap);
    return 0;
}

int
f2c_ssyr2(char* uplo, integer* N,
          real* alpha,
          real* X, integer* incX,
          real* Y, integer* incY,
          real* A, integer* lda)
{
    _starpu_ssyr2_(uplo, N, alpha,
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_sspr2(char* uplo, integer* N,
          real* alpha, 
          real* X, integer* incX,
          real* Y, integer* incY,
          real* A)
{
    _starpu_sspr2_(uplo, N, alpha,
           X, incX, Y, incY, A);
    return 0;
}



int
f2c_dsymv(char* uplo, integer* N,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* X, integer* incX,
          doublereal* beta,
          doublereal* Y, integer* incY)
{
    _starpu_dsymv_(uplo, N, alpha, A, lda, 
           X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_dsbmv(char* uplo, integer* N, integer* K,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* X, integer* incX,
          doublereal* beta,
          doublereal* Y, integer* incY)
{
    _starpu_dsbmv_(uplo, N, K, alpha, A, lda, 
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_dspmv(char* uplo, integer* N,
          doublereal* alpha,
          doublereal* Ap,
          doublereal* X, integer* incX,
          doublereal* beta,
          doublereal* Y, integer* incY)
{
    _starpu_dspmv_(uplo, N, alpha, Ap,  
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_dger(integer* M, integer* N,
         doublereal* alpha,
         doublereal* X, integer* incX,
         doublereal* Y, integer* incY,
         doublereal* A, integer* lda)
{
    _starpu_dger_(M, N, alpha,
          X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_dsyr(char* uplo, integer* N,
         doublereal* alpha,
         doublereal* X, integer* incX,
         doublereal* A, integer* lda)
{
    _starpu_dsyr_(uplo, N, alpha, X, incX, A, lda);
    return 0;
}

int
f2c_dspr(char* uplo, integer* N,
         doublereal* alpha,
         doublereal* X, integer* incX,
         doublereal* Ap)
{
    _starpu_dspr_(uplo, N, alpha, X, incX, Ap);
    return 0;
}

int
f2c_dsyr2(char* uplo, integer* N,
          doublereal* alpha,
          doublereal* X, integer* incX,
          doublereal* Y, integer* incY,
          doublereal* A, integer* lda)
{
    _starpu_dsyr2_(uplo, N, alpha,
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_dspr2(char* uplo, integer* N,
          doublereal* alpha, 
          doublereal* X, integer* incX,
          doublereal* Y, integer* incY,
          doublereal* A)
{
    _starpu_dspr2_(uplo, N, alpha,
           X, incX, Y, incY, A);
    return 0;
}



/*
 * Routines with C and Z prefixes only
 */

int
f2c_chemv(char* uplo, integer* N,
          complex* alpha,
          complex* A, integer* lda,
          complex* X, integer* incX,
          complex* beta,
          complex* Y, integer* incY)
{
    _starpu_chemv_(uplo, N, alpha, A, lda,
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_chbmv(char* uplo, integer* N, integer* K,
          complex* alpha,
          complex* A, integer* lda,
          complex* X, integer* incX,
          complex* beta,
          complex* Y, integer* incY)
{
    _starpu_chbmv_(uplo, N, K, alpha, A, lda,
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_chpmv(char* uplo, integer* N, 
          complex* alpha,
          complex* Ap, 
          complex* X, integer* incX,
          complex* beta,
          complex* Y, integer* incY)
{
    _starpu_chpmv_(uplo, N, alpha, Ap, 
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_cgeru(integer* M, integer* N,
          complex* alpha,
          complex* X, integer* incX,
          complex* Y, integer* incY,
          complex* A, integer* lda)
{
    _starpu_cgeru_(M, N, alpha, 
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_cgerc(integer* M, integer* N,
          complex* alpha,
          complex* X, integer* incX,
          complex* Y, integer* incY,
          complex* A, integer* lda)
{
    _starpu_cgerc_(M, N, alpha, 
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_cher(char* uplo, integer* N,
         real* alpha,
         complex* X, integer* incX,
         complex* A, integer* lda)
{
    _starpu_cher_(uplo, N, alpha,
          X, incX, A, lda);
    return 0;
}

int
f2c_chpr(char* uplo, integer* N,
         real* alpha,
         complex* X, integer* incX,
         complex* Ap)
{
    _starpu_chpr_(uplo, N, alpha,
          X, incX, Ap);
    return 0;
}

int
f2c_cher2(char* uplo, integer* N,
          complex* alpha,
          complex* X, integer* incX,
          complex* Y, integer* incY,
          complex* A, integer* lda)
{
    _starpu_cher2_(uplo, N, alpha,
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_chpr2(char* uplo, integer* N,
          complex* alpha,
          complex* X, integer* incX,
          complex* Y, integer* incY,
          complex* Ap)
{
    _starpu_chpr2_(uplo, N, alpha,
           X, incX, Y, incY, Ap);
    return 0;
}



int
f2c_zhemv(char* uplo, integer* N,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* X, integer* incX,
          doublecomplex* beta,
          doublecomplex* Y, integer* incY)
{
    _starpu_zhemv_(uplo, N, alpha, A, lda,
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_zhbmv(char* uplo, integer* N, integer* K,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* X, integer* incX,
          doublecomplex* beta,
          doublecomplex* Y, integer* incY)
{
    _starpu_zhbmv_(uplo, N, K, alpha, A, lda,
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_zhpmv(char* uplo, integer* N, 
          doublecomplex* alpha,
          doublecomplex* Ap, 
          doublecomplex* X, integer* incX,
          doublecomplex* beta,
          doublecomplex* Y, integer* incY)
{
    _starpu_zhpmv_(uplo, N, alpha, Ap, 
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_zgeru(integer* M, integer* N,
          doublecomplex* alpha,
          doublecomplex* X, integer* incX,
          doublecomplex* Y, integer* incY,
          doublecomplex* A, integer* lda)
{
    _starpu_zgeru_(M, N, alpha, 
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_zgerc(integer* M, integer* N,
          doublecomplex* alpha,
          doublecomplex* X, integer* incX,
          doublecomplex* Y, integer* incY,
          doublecomplex* A, integer* lda)
{
    _starpu_zgerc_(M, N, alpha, 
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_zher(char* uplo, integer* N,
         doublereal* alpha,
         doublecomplex* X, integer* incX,
         doublecomplex* A, integer* lda)
{
    _starpu_zher_(uplo, N, alpha,
          X, incX, A, lda);
    return 0;
}

int
f2c_zhpr(char* uplo, integer* N,
         doublereal* alpha,
         doublecomplex* X, integer* incX,
         doublecomplex* Ap)
{
    _starpu_zhpr_(uplo, N, alpha,
          X, incX, Ap);
    return 0;
}

int
f2c_zher2(char* uplo, integer* N,
          doublecomplex* alpha,
          doublecomplex* X, integer* incX,
          doublecomplex* Y, integer* incY,
          doublecomplex* A, integer* lda)
{
    _starpu_zher2_(uplo, N, alpha,
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_zhpr2(char* uplo, integer* N,
          doublecomplex* alpha,
          doublecomplex* X, integer* incX,
          doublecomplex* Y, integer* incY,
          doublecomplex* Ap)
{
    _starpu_zhpr2_(uplo, N, alpha,
           X, incX, Y, incY, Ap);
    return 0;
}



/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */

int
f2c_sgemm(char* transA, char* transB, integer* M, integer* N, integer* K,
          real* alpha,
          real* A, integer* lda,
          real* B, integer* ldb,
          real* beta,
          real* C, integer* ldc)
{
    _starpu_sgemm_(transA, transB, M, N, K,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_ssymm(char* side, char* uplo, integer* M, integer* N,
          real* alpha,
          real* A, integer* lda,
          real* B, integer* ldb,
          real* beta,
          real* C, integer* ldc)
{
    _starpu_ssymm_(side, uplo, M, N,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_ssyrk(char* uplo, char* trans, integer* N, integer* K,
          real* alpha,
          real* A, integer* lda,
          real* beta,
          real* C, integer* ldc)
{
    _starpu_ssyrk_(uplo, trans, N, K,
           alpha, A, lda, beta, C, ldc);
    return 0;
}

int
f2c_ssyr2k(char* uplo, char* trans, integer* N, integer* K,
           real* alpha,
           real* A, integer* lda,
           real* B, integer* ldb,
           real* beta,
           real* C, integer* ldc)
{
    _starpu_ssyr2k_(uplo, trans, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_strmm(char* side, char* uplo, char* trans, char* diag, 
          integer* M, integer* N,
          real* alpha,
          real* A, integer* lda,
          real* B, integer* ldb)
{
    _starpu_strmm_(side, uplo, 
           trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}

int 
f2c_strsm(char* side, char* uplo, char* trans, char* diag,
          integer* M, integer* N,
          real* alpha,
          real* A, integer* lda,
          real* B, integer* ldb)
{
    _starpu_strsm_(side, uplo, 
           trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}



int
f2c_dgemm(char* transA, char* transB, integer* M, integer* N, integer* K,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* B, integer* ldb,
          doublereal* beta,
          doublereal* C, integer* ldc)
{
    _starpu_dgemm_(transA, transB, M, N, K,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_dsymm(char* side, char* uplo, integer* M, integer* N,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* B, integer* ldb,
          doublereal* beta,
          doublereal* C, integer* ldc)
{
    _starpu_dsymm_(side, uplo, M, N,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_dsyrk(char* uplo, char* trans, integer* N, integer* K,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* beta,
          doublereal* C, integer* ldc)
{
    _starpu_dsyrk_(uplo, trans, N, K,
           alpha, A, lda, beta, C, ldc);
    return 0;
}

int
f2c_dsyr2k(char* uplo, char* trans, integer* N, integer* K,
           doublereal* alpha,
           doublereal* A, integer* lda,
           doublereal* B, integer* ldb,
           doublereal* beta,
           doublereal* C, integer* ldc)
{
    _starpu_dsyr2k_(uplo, trans, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_dtrmm(char* side, char* uplo, char* trans, char* diag, 
          integer* M, integer* N,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* B, integer* ldb)
{
    _starpu_dtrmm_(side, uplo, trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}

int 
f2c_dtrsm(char* side, char* uplo, char* trans, char* diag,
          integer* M, integer* N,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* B, integer* ldb)
{
    _starpu_dtrsm_(side, uplo, trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}



int
f2c_cgemm(char* transA, char* transB, integer* M, integer* N, integer* K,
          complex* alpha,
          complex* A, integer* lda,
          complex* B, integer* ldb,
          complex* beta,
          complex* C, integer* ldc)
{
    _starpu_cgemm_(transA, transB, M, N, K,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_csymm(char* side, char* uplo, integer* M, integer* N,
          complex* alpha,
          complex* A, integer* lda,
          complex* B, integer* ldb,
          complex* beta,
          complex* C, integer* ldc)
{
    _starpu_csymm_(side, uplo, M, N,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_csyrk(char* uplo, char* trans, integer* N, integer* K,
          complex* alpha,
          complex* A, integer* lda,
          complex* beta,
          complex* C, integer* ldc)
{
    _starpu_csyrk_(uplo, trans, N, K,
           alpha, A, lda, beta, C, ldc);
    return 0;
}

int
f2c_csyr2k(char* uplo, char* trans, integer* N, integer* K,
           complex* alpha,
           complex* A, integer* lda,
           complex* B, integer* ldb,
           complex* beta,
           complex* C, integer* ldc)
{
    _starpu_csyr2k_(uplo, trans, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_ctrmm(char* side, char* uplo, char* trans, char* diag, 
          integer* M, integer* N,
          complex* alpha,
          complex* A, integer* lda,
          complex* B, integer* ldb)
{
    _starpu_ctrmm_(side, uplo, trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}

int 
f2c_ctrsm(char* side, char* uplo, char* trans, char* diag,
          integer* M, integer* N,
          complex* alpha,
          complex* A, integer* lda,
          complex* B, integer* ldb)
{
    _starpu_ctrsm_(side, uplo, trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}



int
f2c_zgemm(char* transA, char* transB, integer* M, integer* N, integer* K,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* B, integer* ldb,
          doublecomplex* beta,
          doublecomplex* C, integer* ldc)
{
    _starpu_zgemm_(transA, transB, M, N, K,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_zsymm(char* side, char* uplo, integer* M, integer* N,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* B, integer* ldb,
          doublecomplex* beta,
          doublecomplex* C, integer* ldc)
{
    _starpu_zsymm_(side, uplo, M, N,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_zsyrk(char* uplo, char* trans, integer* N, integer* K,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* beta,
          doublecomplex* C, integer* ldc)
{
    _starpu_zsyrk_(uplo, trans, N, K,
           alpha, A, lda, beta, C, ldc);
    return 0;
}

int
f2c_zsyr2k(char* uplo, char* trans, integer* N, integer* K,
           doublecomplex* alpha,
           doublecomplex* A, integer* lda,
           doublecomplex* B, integer* ldb,
           doublecomplex* beta,
           doublecomplex* C, integer* ldc)
{
    _starpu_zsyr2k_(uplo, trans, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_ztrmm(char* side, char* uplo, char* trans, char* diag, 
          integer* M, integer* N,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* B, integer* ldb)
{
    _starpu_ztrmm_(side, uplo, trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}

int 
f2c_ztrsm(char* side, char* uplo, char* trans, char* diag,
          integer* M, integer* N,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* B, integer* ldb)
{
    _starpu_ztrsm_(side, uplo, trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}



/*
 * Routines with prefixes C and Z only
 */

int
f2c_chemm(char* side, char* uplo, integer* M, integer* N,
          complex* alpha,
          complex* A, integer* lda,
          complex* B, integer* ldb,
          complex* beta,
          complex* C, integer* ldc)
{
    _starpu_chemm_(side, uplo, M, N,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_cherk(char* uplo, char* trans, integer* N, integer* K,
          real* alpha,
          complex* A, integer* lda,
          real* beta,
          complex* C, integer* ldc)
{
    _starpu_cherk_(uplo, trans, N, K,
           alpha, A, lda, beta, C, ldc);
    return 0;
}

int
f2c_cher2k(char* uplo, char* trans, integer* N, integer* K,
           complex* alpha,
           complex* A, integer* lda,
           complex* B, integer* ldb,
           real* beta,
           complex* C, integer* ldc)
{
    _starpu_cher2k_(uplo, trans, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}



int
f2c_zhemm(char* side, char* uplo, integer* M, integer* N,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* B, integer* ldb,
          doublecomplex* beta,
          doublecomplex* C, integer* ldc)
{
    _starpu_zhemm_(side, uplo, M, N,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_zherk(char* uplo, char* trans, integer* N, integer* K,
          doublereal* alpha,
          doublecomplex* A, integer* lda,
          doublereal* beta,
          doublecomplex* C, integer* ldc)
{
    _starpu_zherk_(uplo, trans, N, K,
           alpha, A, lda, beta, C, ldc);
    return 0;
}

int
f2c_zher2k(char* uplo, char* trans, integer* N, integer* K,
           doublecomplex* alpha,
           doublecomplex* A, integer* lda,
           doublecomplex* B, integer* ldb,
           doublereal* beta,
           doublecomplex* C, integer* ldc)
{
    _starpu_zher2k_(uplo, trans, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

