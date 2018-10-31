real 
_starpu_sdot_(integer* N, 
      real* X, integer* incX, 
      real* Y, integer* incY);

doublereal
_starpu_ddot_(integer* N, 
      doublereal* X, integer* incX, 
      doublereal* Y, integer* incY);

void 
_starpu_cdotu_(complex* retval,
       integer* N, 
       complex* X, integer* incX, 
       complex* Y, integer* incY);

void
_starpu_cdotc_(complex* retval,
       integer* N, 
       complex* X, integer* incX, 
       complex* Y, integer* incY);

void
_starpu_zdotu_(doublecomplex* retval,
       integer* N, 
       doublecomplex* X, integer* incX, 
       doublecomplex* Y, integer* incY);

void
_starpu_zdotc_(doublecomplex* retval,
       integer* N, 
       doublecomplex* X, integer* incX, 
       doublecomplex* Y, integer* incY);

real 
_starpu_snrm2_(integer* N, 
       real* X, integer* incX);

real
_starpu_sasum_(integer* N, 
       real* X, integer* incX);

doublereal
_starpu_dnrm2_(integer* N, 
       doublereal* X, integer* incX);

doublereal
_starpu_dasum_(integer* N, 
       doublereal* X, integer* incX);

real 
_starpu_scnrm2_(integer* N, 
        complex* X, integer* incX);

real
_starpu_scasum_(integer* N, 
        complex* X, integer* incX);

doublereal 
_starpu_dznrm2_(integer* N, 
        doublecomplex* X, integer* incX);

doublereal
_starpu_dzasum_(integer* N, 
        doublecomplex* X, integer* incX);

integer
_starpu_isamax_(integer* N,
        real* X, integer* incX);

integer
_starpu_idamax_(integer* N,
        doublereal* X, integer* incX);

integer
_starpu_icamax_(integer* N,
        complex* X, integer* incX);

integer
_starpu_izamax_(integer* N,
        doublecomplex* X, integer* incX);

int
_starpu_sswap_(integer* N,
       real* X, integer* incX,
       real* Y, integer* incY);

int
_starpu_scopy_(integer* N,
       real* X, integer* incX,
       real* Y, integer* incY);

int
_starpu_saxpy_(integer* N,
       real* alpha,
       real* X, integer* incX,
       real* Y, integer* incY);

int
_starpu_dswap_(integer* N,
       doublereal* X, integer* incX,
       doublereal* Y, integer* incY);

int
_starpu_dcopy_(integer* N,
       doublereal* X, integer* incX,
       doublereal* Y, integer* incY);

int
_starpu_daxpy_(integer* N,
       doublereal* alpha,
       doublereal* X, integer* incX,
       doublereal* Y, integer* incY);

int
_starpu_cswap_(integer* N,
       complex* X, integer* incX,
       complex* Y, integer* incY);

int
_starpu_ccopy_(integer* N,
       complex* X, integer* incX,
       complex* Y, integer* incY);

int
_starpu_caxpy_(integer* N,
      complex* alpha,
      complex* X, integer* incX,
      complex* Y, integer* incY);

int
_starpu_zswap_(integer* N,
       doublecomplex* X, integer* incX,
       doublecomplex* Y, integer* incY);

int
_starpu_zcopy_(integer* N,
       doublecomplex* X, integer* incX,
       doublecomplex* Y, integer* incY);

int
_starpu_zaxpy_(integer* N,
       doublecomplex* alpha,
       doublecomplex* X, integer* incX,
       doublecomplex* Y, integer* incY);

int
_starpu_srotg_(real* a, real* b, real* c, real* s);

int
_starpu_srot_(integer* N,
      real* X, integer* incX,
      real* Y, integer* incY,
      real* c, real* s);

int
_starpu_crotg_(complex* a, complex* b, complex* c, complex* s);

int
_starpu_drotg_(doublereal* a, doublereal* b, doublereal* c, doublereal* s);

int
_starpu_drot_(integer* N,
      doublereal* X, integer* incX,
      doublereal* Y, integer* incY,
      doublereal* c, doublereal* s);

int
_starpu_zrotg_(doublecomplex* a, doublecomplex* b, doublecomplex* c, doublecomplex* s);

int
_starpu_sscal_(integer* N,
       real* alpha,
       real* X, integer* incX);

int
_starpu_dscal_(integer* N,
       doublereal* alpha,
       doublereal* X, integer* incX);

int
_starpu_cscal_(integer* N,
       complex* alpha,
       complex* X, integer* incX);

int
_starpu_zscal_(integer* N,
       doublecomplex* alpha,
       doublecomplex* X, integer* incX);

int
_starpu_csscal_(integer* N,
        real* alpha,
        complex* X, integer* incX);

int
_starpu_zdscal_(integer* N,
        doublereal* alpha,
        doublecomplex* X, integer* incX);

int
_starpu_sgemv_(char* trans, integer* M, integer* N,
       real* alpha,
       real* A, integer* lda,
       real* X, integer* incX,
       real* beta,
       real* Y, integer* incY);

int
_starpu_sgbmv_(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
       real *alpha, 
       real *A, integer *lda, 
       real *X, integer *incX, 
       real *beta, 
       real *Y, integer *incY);

int 
_starpu_strmv_(char* uplo, char *trans, char* diag, integer *N,  
       real *A, integer *lda, 
       real *X, integer *incX);

int
_starpu_stbmv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       real* A, integer* lda,
       real* X, integer* incX);

int
_starpu_stpmv_(char* uplo, char* trans, char* diag, integer* N, 
       real* Ap, 
       real* X, integer* incX);

int
_starpu_strsv_(char* uplo, char* trans, char* diag, integer* N,
       real* A, integer* lda,
       real* X, integer* incX);

int
_starpu_stbsv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       real* A, integer* lda, 
       real* X, integer* incX);

int
_starpu_stpsv_(char* uplo, char* trans, char* diag, integer* N, 
       real* Ap, 
       real* X, integer* incX);

int
_starpu_dgemv_(char* trans, integer* M, integer* N,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* X, integer* incX,
       doublereal* beta,
       doublereal* Y, integer* incY);

int 
_starpu_dgbmv_(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
       doublereal *alpha, 
       doublereal *A, integer *lda, 
       doublereal *X, integer *incX, 
       doublereal *beta, 
       doublereal *Y, integer *incY);

int 
_starpu_dtrmv_(char* uplo, char *trans, char* diag, integer *N,  
       doublereal *A, integer *lda, 
       doublereal *X, integer *incX);

int
_starpu_dtbmv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       doublereal* A, integer* lda,
       doublereal* X, integer* incX);

int
_starpu_dtpmv_(char* uplo, char* trans, char* diag, integer* N, 
       doublereal* Ap, 
       doublereal* X, integer* incX);

int
_starpu_dtrsv_(char* uplo, char* trans, char* diag, integer* N,
       doublereal* A, integer* lda,
       doublereal* X, integer* incX);

int
_starpu_dtbsv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       doublereal* A, integer* lda, 
       doublereal* X, integer* incX);

int
_starpu_dtpsv_(char* uplo, char* trans, char* diag, integer* N, 
       doublereal* Ap, 
       doublereal* X, integer* incX);

int
_starpu_cgemv_(char* trans, integer* M, integer* N,
       complex* alpha,
       complex* A, integer* lda,
       complex* X, integer* incX,
       complex* beta,
       complex* Y, integer* incY);

int 
_starpu_cgbmv_(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
       complex *alpha, 
       complex *A, integer *lda, 
       complex *X, integer *incX, 
       complex *beta, 
       complex *Y, integer *incY);

int 
_starpu_ctrmv_(char* uplo, char *trans, char* diag, integer *N,  
       complex *A, integer *lda, 
       complex *X, integer *incX);

int
_starpu_ctbmv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       complex* A, integer* lda,
       complex* X, integer* incX);

int
_starpu_ctpmv_(char* uplo, char* trans, char* diag, integer* N, 
       complex* Ap, 
       complex* X, integer* incX);

int
_starpu_ctrsv_(char* uplo, char* trans, char* diag, integer* N,
       complex* A, integer* lda,
       complex* X, integer* incX);

int
_starpu_ctbsv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       complex* A, integer* lda, 
       complex* X, integer* incX);

int
_starpu_ctpsv_(char* uplo, char* trans, char* diag, integer* N, 
       complex* Ap, 
       complex* X, integer* incX);

int
_starpu_zgemv_(char* trans, integer* M, integer* N,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* X, integer* incX,
       doublecomplex* beta,
       doublecomplex* Y, integer* incY);

int 
_starpu_zgbmv_(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
       doublecomplex *alpha, 
       doublecomplex *A, integer *lda, 
       doublecomplex *X, integer *incX, 
       doublecomplex *beta, 
       doublecomplex *Y, integer *incY);

int 
_starpu_ztrmv_(char* uplo, char *trans, char* diag, integer *N,  
       doublecomplex *A, integer *lda, 
       doublecomplex *X, integer *incX);

int
_starpu_ztbmv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       doublecomplex* A, integer* lda,
       doublecomplex* X, integer* incX);

 void  
_starpu_ztpmv_(char* uplo, char* trans, char* diag, integer* N, 
      doublecomplex* Ap, 
      doublecomplex* X, integer* incX);

int
_starpu_ztrsv_(char* uplo, char* trans, char* diag, integer* N,
       doublecomplex* A, integer* lda,
       doublecomplex* X, integer* incX);

int
_starpu_ztbsv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       doublecomplex* A, integer* lda, 
       doublecomplex* X, integer* incX);

int
_starpu_ztpsv_(char* uplo, char* trans, char* diag, integer* N, 
       doublecomplex* Ap, 
       doublecomplex* X, integer* incX);

int
_starpu_ssymv_(char* uplo, integer* N,
       real* alpha,
       real* A, integer* lda,
       real* X, integer* incX,
       real* beta,
       real* Y, integer* incY);

int 
_starpu_ssbmv_(char* uplo, integer* N, integer* K,
       real* alpha,
       real* A, integer* lda,
       real* X, integer* incX,
       real* beta,
       real* Y, integer* incY);

int
_starpu_sspmv_(char* uplo, integer* N,
       real* alpha,
       real* Ap,
       real* X, integer* incX,
       real* beta,
       real* Y, integer* incY);

int
_starpu_sger_(integer* M, integer* N,
      real* alpha,
      real* X, integer* incX,
      real* Y, integer* incY,
      real* A, integer* lda);

int
_starpu_ssyr_(char* uplo, integer* N,
      real* alpha,
      real* X, integer* incX,
      real* A, integer* lda);

int
_starpu_sspr_(char* uplo, integer* N,
      real* alpha,
      real* X, integer* incX,
      real* Ap);

int
_starpu_ssyr2_(char* uplo, integer* N,
       real* alpha,
       real* X, integer* incX,
       real* Y, integer* incY,
       real* A, integer* lda);

int
_starpu_sspr2_(char* uplo, integer* N,
       real* alpha, 
       real* X, integer* incX,
       real* Y, integer* incY,
       real* A);

int
_starpu_dsymv_(char* uplo, integer* N,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* X, integer* incX,
       doublereal* beta,
       doublereal* Y, integer* incY);

int 
_starpu_dsbmv_(char* uplo, integer* N, integer* K,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* X, integer* incX,
       doublereal* beta,
       doublereal* Y, integer* incY);

int
_starpu_dspmv_(char* uplo, integer* N,
       doublereal* alpha,
       doublereal* Ap,
       doublereal* X, integer* incX,
       doublereal* beta,
       doublereal* Y, integer* incY);

int
_starpu_dger_(integer* M, integer* N,
      doublereal* alpha,
      doublereal* X, integer* incX,
      doublereal* Y, integer* incY,
      doublereal* A, integer* lda);

int
_starpu_dsyr_(char* uplo, integer* N,
      doublereal* alpha,
      doublereal* X, integer* incX,
      doublereal* A, integer* lda);

int
_starpu_dspr_(char* uplo, integer* N,
      doublereal* alpha,
      doublereal* X, integer* incX,
      doublereal* Ap);

int
_starpu_dsyr2_(char* uplo, integer* N,
       doublereal* alpha,
       doublereal* X, integer* incX,
       doublereal* Y, integer* incY,
       doublereal* A, integer* lda);

int
_starpu_dspr2_(char* uplo, integer* N,
       doublereal* alpha, 
       doublereal* X, integer* incX,
       doublereal* Y, integer* incY,
       doublereal* A);

int
_starpu_chemv_(char* uplo, integer* N,
       complex* alpha,
       complex* A, integer* lda,
       complex* X, integer* incX,
       complex* beta,
       complex* Y, integer* incY);

int
_starpu_chbmv_(char* uplo, integer* N, integer* K,
       complex* alpha,
       complex* A, integer* lda,
       complex* X, integer* incX,
       complex* beta,
       complex* Y, integer* incY);

int
_starpu_chpmv_(char* uplo, integer* N, 
       complex* alpha,
       complex* Ap, 
       complex* X, integer* incX,
       complex* beta,
       complex* Y, integer* incY);

int
_starpu_cgeru_(integer* M, integer* N,
       complex* alpha,
       complex* X, integer* incX,
       complex* Y, integer* incY,
       complex* A, integer* lda);

int
_starpu_cgerc_(integer* M, integer* N,
       complex* alpha,
       complex* X, integer* incX,
       complex* Y, integer* incY,
       complex* A, integer* lda);

int
_starpu_cher_(char* uplo, integer* N,
      real* alpha,
      complex* X, integer* incX,
      complex* A, integer* lda);

int
_starpu_chpr_(char* uplo, integer* N,
      real* alpha,
      complex* X, integer* incX,
      complex* Ap);

int
_starpu_cher2_(char* uplo, integer* N,
       complex* alpha,
       complex* X, integer* incX,
       complex* Y, integer* incY,
       complex* A, integer* lda);

int
_starpu_chpr2_(char* uplo, integer* N,
       complex* alpha,
       complex* X, integer* incX,
       complex* Y, integer* incY,
       complex* Ap);

int
_starpu_zhemv_(char* uplo, integer* N,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* X, integer* incX,
       doublecomplex* beta,
       doublecomplex* Y, integer* incY);

int
_starpu_zhbmv_(char* uplo, integer* N, integer* K,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* X, integer* incX,
       doublecomplex* beta,
       doublecomplex* Y, integer* incY);

int
_starpu_zhpmv_(char* uplo, integer* N, 
       doublecomplex* alpha,
       doublecomplex* Ap, 
       doublecomplex* X, integer* incX,
       doublecomplex* beta,
       doublecomplex* Y, integer* incY);

int
_starpu_zgeru_(integer* M, integer* N,
       doublecomplex* alpha,
       doublecomplex* X, integer* incX,
       doublecomplex* Y, integer* incY,
       doublecomplex* A, integer* lda);

int
_starpu_zgerc_(integer* M, integer* N,
       doublecomplex* alpha,
       doublecomplex* X, integer* incX,
       doublecomplex* Y, integer* incY,
       doublecomplex* A, integer* lda);

int
_starpu_zher_(char* uplo, integer* N,
      doublereal* alpha,
      doublecomplex* X, integer* incX,
      doublecomplex* A, integer* lda);

int
_starpu_zhpr_(char* uplo, integer* N,
      doublereal* alpha,
      doublecomplex* X, integer* incX,
      doublecomplex* Ap);

int
_starpu_zher2_(char* uplo, integer* N,
       doublecomplex* alpha,
       doublecomplex* X, integer* incX,
       doublecomplex* Y, integer* incY,
       doublecomplex* A, integer* lda);

int
_starpu_zhpr2_(char* uplo, integer* N,
       doublecomplex* alpha,
       doublecomplex* X, integer* incX,
       doublecomplex* Y, integer* incY,
       doublecomplex* Ap);

int
_starpu_sgemm_(char* transA, char* transB, integer* M, integer* N, integer* K,
       real* alpha,
       real* A, integer* lda,
       real* B, integer* ldb,
       real* beta,
       real* C, integer* ldc);

int
_starpu_ssymm_(char* side, char* uplo, integer* M, integer* N,
       real* alpha,
       real* A, integer* lda,
       real* B, integer* ldb,
       real* beta,
       real* C, integer* ldc);

int
_starpu_ssyrk_(char* uplo, char* trans, integer* N, integer* K,
       real* alpha,
       real* A, integer* lda,
       real* beta,
       real* C, integer* ldc);

int
_starpu_ssyr2k_(char* uplo, char* trans, integer* N, integer* K,
        real* alpha,
        real* A, integer* lda,
        real* B, integer* ldb,
        real* beta,
        real* C, integer* ldc);

int
_starpu_strmm_(char* side, char* uplo, char* trans, char* diag, 
       integer* M, integer* N,
       real* alpha,
       real* A, integer* lda,
       real* B, integer* ldb);

int 
_starpu_strsm_(char* side, char* uplo, char* trans, char* diag,
       integer* M, integer* N,
       real* alpha,
       real* A, integer* lda,
       real* B, integer* ldb);

int
_starpu_dgemm_(char* transA, char* transB, integer* M, integer* N, integer* K,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* B, integer* ldb,
       doublereal* beta,
       doublereal* C, integer* ldc);

int
_starpu_dsymm_(char* side, char* uplo, integer* M, integer* N,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* B, integer* ldb,
       doublereal* beta,
       doublereal* C, integer* ldc);

int
_starpu_dsyrk_(char* uplo, char* trans, integer* N, integer* K,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* beta,
       doublereal* C, integer* ldc);

int
_starpu_dsyr2k_(char* uplo, char* trans, integer* N, integer* K,
        doublereal* alpha,
        doublereal* A, integer* lda,
        doublereal* B, integer* ldb,
        doublereal* beta,
        doublereal* C, integer* ldc);

int
_starpu_dtrmm_(char* side, char* uplo, char* trans, char* diag, 
       integer* M, integer* N,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* B, integer* ldb);

int 
_starpu_dtrsm_(char* side, char* uplo, char* trans, char* diag,
       integer* M, integer* N,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* B, integer* ldb);

int
_starpu_cgemm_(char* transA, char* transB, integer* M, integer* N, integer* K,
       complex* alpha,
       complex* A, integer* lda,
       complex* B, integer* ldb,
       complex* beta,
       complex* C, integer* ldc);

int
_starpu_csymm_(char* side, char* uplo, integer* M, integer* N,
       complex* alpha,
       complex* A, integer* lda,
       complex* B, integer* ldb,
       complex* beta,
       complex* C, integer* ldc);

int
_starpu_csyrk_(char* uplo, char* trans, integer* N, integer* K,
       complex* alpha,
       complex* A, integer* lda,
       complex* beta,
       complex* C, integer* ldc);

int
_starpu_csyr2k_(char* uplo, char* trans, integer* N, integer* K,
        complex* alpha,
        complex* A, integer* lda,
        complex* B, integer* ldb,
        complex* beta,
        complex* C, integer* ldc);

int
_starpu_ctrmm_(char* side, char* uplo, char* trans, char* diag, 
       integer* M, integer* N,
       complex* alpha,
       complex* A, integer* lda,
       complex* B, integer* ldb);

int 
_starpu_ctrsm_(char* side, char* uplo, char* trans, char* diag,
       integer* M, integer* N,
       complex* alpha,
       complex* A, integer* lda,
       complex* B, integer* ldb);

int
_starpu_zgemm_(char* transA, char* transB, integer* M, integer* N, integer* K,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* B, integer* ldb,
       doublecomplex* beta,
       doublecomplex* C, integer* ldc);

int
_starpu_zsymm_(char* side, char* uplo, integer* M, integer* N,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* B, integer* ldb,
       doublecomplex* beta,
       doublecomplex* C, integer* ldc);

int
_starpu_zsyrk_(char* uplo, char* trans, integer* N, integer* K,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* beta,
       doublecomplex* C, integer* ldc);

int
_starpu_zsyr2k_(char* uplo, char* trans, integer* N, integer* K,
        doublecomplex* alpha,
        doublecomplex* A, integer* lda,
        doublecomplex* B, integer* ldb,
        doublecomplex* beta,
        doublecomplex* C, integer* ldc);

int
_starpu_ztrmm_(char* side, char* uplo, char* trans, char* diag, 
       integer* M, integer* N,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* B, integer* ldb);

int 
_starpu_ztrsm_(char* side, char* uplo, char* trans, char* diag,
       integer* M, integer* N,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* B, integer* ldb);

int
_starpu_chemm_(char* side, char* uplo, integer* M, integer* N,
       complex* alpha,
       complex* A, integer* lda,
       complex* B, integer* ldb,
       complex* beta,
       complex* C, integer* ldc);

int
_starpu_cherk_(char* uplo, char* trans, integer* N, integer* K,
       real* alpha,
       complex* A, integer* lda,
       real* beta,
       complex* C, integer* ldc);

int
_starpu_cher2k_(char* uplo, char* trans, integer* N, integer* K,
        complex* alpha,
        complex* A, integer* lda,
        complex* B, integer* ldb,
        real* beta,
        complex* C, integer* ldc);

int
_starpu_zhemm_(char* side, char* uplo, integer* M, integer* N,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* B, integer* ldb,
       doublecomplex* beta,
       doublecomplex* C, integer* ldc);

int
_starpu_zherk_(char* uplo, char* trans, integer* N, integer* K,
       doublereal* alpha,
       doublecomplex* A, integer* lda,
       doublereal* beta,
       doublecomplex* C, integer* ldc);

int
_starpu_zher2k_(char* uplo, char* trans, integer* N, integer* K,
        doublecomplex* alpha,
        doublecomplex* A, integer* lda,
        doublecomplex* B, integer* ldb,
        doublereal* beta,
        doublecomplex* C, integer* ldc);
