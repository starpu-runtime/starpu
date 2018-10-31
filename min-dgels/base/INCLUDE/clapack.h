/* header file for clapack 3.2.1 */

#ifndef __CLAPACK_H
#define __CLAPACK_H

#ifdef __cplusplus 	
extern "C" {	
#endif		

/* Subroutine */ int _starpu_caxpy_(integer *n, complex *ca, complex *cx, integer *
	incx, complex *cy, integer *incy);

/* Subroutine */ int _starpu_ccopy_(integer *n, complex *cx, integer *incx, complex *
	cy, integer *incy);

/* Complex */ VOID _starpu_cdotc_(complex * ret_val, integer *n, complex *cx, integer 
	*incx, complex *cy, integer *incy);

/* Complex */ VOID _starpu_cdotu_(complex * ret_val, integer *n, complex *cx, integer 
	*incx, complex *cy, integer *incy);

/* Subroutine */ int _starpu_cgbmv_(char *trans, integer *m, integer *n, integer *kl, 
	integer *ku, complex *alpha, complex *a, integer *lda, complex *x, 
	integer *incx, complex *beta, complex *y, integer *incy);

/* Subroutine */ int _starpu_cgemm_(char *transa, char *transb, integer *m, integer *
	n, integer *k, complex *alpha, complex *a, integer *lda, complex *b, 
	integer *ldb, complex *beta, complex *c__, integer *ldc);

/* Subroutine */ int _starpu_cgemv_(char *trans, integer *m, integer *n, complex *
	alpha, complex *a, integer *lda, complex *x, integer *incx, complex *
	beta, complex *y, integer *incy);

/* Subroutine */ int _starpu_cgerc_(integer *m, integer *n, complex *alpha, complex *
	x, integer *incx, complex *y, integer *incy, complex *a, integer *lda);

/* Subroutine */ int _starpu_cgeru_(integer *m, integer *n, complex *alpha, complex *
	x, integer *incx, complex *y, integer *incy, complex *a, integer *lda);

/* Subroutine */ int _starpu_chbmv_(char *uplo, integer *n, integer *k, complex *
	alpha, complex *a, integer *lda, complex *x, integer *incx, complex *
	beta, complex *y, integer *incy);

/* Subroutine */ int _starpu_chemm_(char *side, char *uplo, integer *m, integer *n, 
	complex *alpha, complex *a, integer *lda, complex *b, integer *ldb, 
	complex *beta, complex *c__, integer *ldc);

/* Subroutine */ int _starpu_chemv_(char *uplo, integer *n, complex *alpha, complex *
	a, integer *lda, complex *x, integer *incx, complex *beta, complex *y, 
	 integer *incy);

/* Subroutine */ int _starpu_cher_(char *uplo, integer *n, real *alpha, complex *x, 
	integer *incx, complex *a, integer *lda);

/* Subroutine */ int _starpu_cher2_(char *uplo, integer *n, complex *alpha, complex *
	x, integer *incx, complex *y, integer *incy, complex *a, integer *lda);

/* Subroutine */ int _starpu_cher2k_(char *uplo, char *trans, integer *n, integer *k, 
	complex *alpha, complex *a, integer *lda, complex *b, integer *ldb, 
	real *beta, complex *c__, integer *ldc);

/* Subroutine */ int _starpu_cherk_(char *uplo, char *trans, integer *n, integer *k, 
	real *alpha, complex *a, integer *lda, real *beta, complex *c__, 
	integer *ldc);

/* Subroutine */ int _starpu_chpmv_(char *uplo, integer *n, complex *alpha, complex *
	ap, complex *x, integer *incx, complex *beta, complex *y, integer *
	incy);

/* Subroutine */ int _starpu_chpr_(char *uplo, integer *n, real *alpha, complex *x, 
	integer *incx, complex *ap);

/* Subroutine */ int _starpu_chpr2_(char *uplo, integer *n, complex *alpha, complex *
	x, integer *incx, complex *y, integer *incy, complex *ap);

/* Subroutine */ int _starpu_crotg_(complex *ca, complex *cb, real *c__, complex *s);

/* Subroutine */ int _starpu_cscal_(integer *n, complex *ca, complex *cx, integer *
	incx);

/* Subroutine */ int _starpu__starpu_csrot_(integer *n, complex *cx, integer *incx, complex *
	cy, integer *incy, real *c__, real *s);

/* Subroutine */ int _starpu_csscal_(integer *n, real *sa, complex *cx, integer *incx);

/* Subroutine */ int _starpu_cswap_(integer *n, complex *cx, integer *incx, complex *
	cy, integer *incy);

/* Subroutine */ int _starpu_csymm_(char *side, char *uplo, integer *m, integer *n, 
	complex *alpha, complex *a, integer *lda, complex *b, integer *ldb, 
	complex *beta, complex *c__, integer *ldc);

/* Subroutine */ int _starpu_csyr2k_(char *uplo, char *trans, integer *n, integer *k, 
	complex *alpha, complex *a, integer *lda, complex *b, integer *ldb, 
	complex *beta, complex *c__, integer *ldc);

/* Subroutine */ int _starpu_csyrk_(char *uplo, char *trans, integer *n, integer *k, 
	complex *alpha, complex *a, integer *lda, complex *beta, complex *c__, 
	 integer *ldc);

/* Subroutine */ int _starpu_ctbmv_(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, complex *a, integer *lda, complex *x, integer *incx);

/* Subroutine */ int _starpu_ctbsv_(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, complex *a, integer *lda, complex *x, integer *incx);

/* Subroutine */ int _starpu_ctpmv_(char *uplo, char *trans, char *diag, integer *n, 
	complex *ap, complex *x, integer *incx);

/* Subroutine */ int _starpu_ctpsv_(char *uplo, char *trans, char *diag, integer *n, 
	complex *ap, complex *x, integer *incx);

/* Subroutine */ int _starpu_ctrmm_(char *side, char *uplo, char *transa, char *diag, 
	integer *m, integer *n, complex *alpha, complex *a, integer *lda, 
	complex *b, integer *ldb);

/* Subroutine */ int _starpu_ctrmv_(char *uplo, char *trans, char *diag, integer *n, 
	complex *a, integer *lda, complex *x, integer *incx);

/* Subroutine */ int _starpu_ctrsm_(char *side, char *uplo, char *transa, char *diag, 
	integer *m, integer *n, complex *alpha, complex *a, integer *lda, 
	complex *b, integer *ldb);

/* Subroutine */ int _starpu_ctrsv_(char *uplo, char *trans, char *diag, integer *n, 
	complex *a, integer *lda, complex *x, integer *incx);

doublereal _starpu_dasum_(integer *n, doublereal *dx, integer *incx);

/* Subroutine */ int _starpu_daxpy_(integer *n, doublereal *da, doublereal *dx, 
	integer *incx, doublereal *dy, integer *incy);

doublereal _starpu_dcabs1_(doublecomplex *z__);

/* Subroutine */ int _starpu_dcopy_(integer *n, doublereal *dx, integer *incx, 
	doublereal *dy, integer *incy);

doublereal _starpu_ddot_(integer *n, doublereal *dx, integer *incx, doublereal *dy, 
	integer *incy);

/* Subroutine */ int _starpu_dgbmv_(char *trans, integer *m, integer *n, integer *kl, 
	integer *ku, doublereal *alpha, doublereal *a, integer *lda, 
	doublereal *x, integer *incx, doublereal *beta, doublereal *y, 
	integer *incy);

/* Subroutine */ int _starpu_dgemm_(char *transa, char *transb, integer *m, integer *
	n, integer *k, doublereal *alpha, doublereal *a, integer *lda, 
	doublereal *b, integer *ldb, doublereal *beta, doublereal *c__, 
	integer *ldc);

/* Subroutine */ int _starpu_dgemv_(char *trans, integer *m, integer *n, doublereal *
	alpha, doublereal *a, integer *lda, doublereal *x, integer *incx, 
	doublereal *beta, doublereal *y, integer *incy);

/* Subroutine */ int _starpu_dger_(integer *m, integer *n, doublereal *alpha, 
	doublereal *x, integer *incx, doublereal *y, integer *incy, 
	doublereal *a, integer *lda);

doublereal _starpu_dnrm2_(integer *n, doublereal *x, integer *incx);

/* Subroutine */ int _starpu_drot_(integer *n, doublereal *dx, integer *incx, 
	doublereal *dy, integer *incy, doublereal *c__, doublereal *s);

/* Subroutine */ int _starpu_drotg_(doublereal *da, doublereal *db, doublereal *c__, 
	doublereal *s);

/* Subroutine */ int _starpu_drotm_(integer *n, doublereal *dx, integer *incx, 
	doublereal *dy, integer *incy, doublereal *dparam);

/* Subroutine */ int _starpu_drotmg_(doublereal *dd1, doublereal *dd2, doublereal *
	dx1, doublereal *dy1, doublereal *dparam);

/* Subroutine */ int _starpu_dsbmv_(char *uplo, integer *n, integer *k, doublereal *
	alpha, doublereal *a, integer *lda, doublereal *x, integer *incx, 
	doublereal *beta, doublereal *y, integer *incy);

/* Subroutine */ int _starpu_dscal_(integer *n, doublereal *da, doublereal *dx, 
	integer *incx);

doublereal _starpu_dsdot_(integer *n, real *sx, integer *incx, real *sy, integer *
	incy);

/* Subroutine */ int _starpu_dspmv_(char *uplo, integer *n, doublereal *alpha, 
	doublereal *ap, doublereal *x, integer *incx, doublereal *beta, 
	doublereal *y, integer *incy);

/* Subroutine */ int _starpu_dspr_(char *uplo, integer *n, doublereal *alpha, 
	doublereal *x, integer *incx, doublereal *ap);

/* Subroutine */ int _starpu_dspr2_(char *uplo, integer *n, doublereal *alpha, 
	doublereal *x, integer *incx, doublereal *y, integer *incy, 
	doublereal *ap);

/* Subroutine */ int _starpu_dswap_(integer *n, doublereal *dx, integer *incx, 
	doublereal *dy, integer *incy);

/* Subroutine */ int _starpu_dsymm_(char *side, char *uplo, integer *m, integer *n, 
	doublereal *alpha, doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *beta, doublereal *c__, integer *ldc);

/* Subroutine */ int _starpu_dsymv_(char *uplo, integer *n, doublereal *alpha, 
	doublereal *a, integer *lda, doublereal *x, integer *incx, doublereal 
	*beta, doublereal *y, integer *incy);

/* Subroutine */ int _starpu_dsyr_(char *uplo, integer *n, doublereal *alpha, 
	doublereal *x, integer *incx, doublereal *a, integer *lda);

/* Subroutine */ int _starpu_dsyr2_(char *uplo, integer *n, doublereal *alpha, 
	doublereal *x, integer *incx, doublereal *y, integer *incy, 
	doublereal *a, integer *lda);

/* Subroutine */ int _starpu_dsyr2k_(char *uplo, char *trans, integer *n, integer *k, 
	doublereal *alpha, doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *beta, doublereal *c__, integer *ldc);

/* Subroutine */ int _starpu_dsyrk_(char *uplo, char *trans, integer *n, integer *k, 
	doublereal *alpha, doublereal *a, integer *lda, doublereal *beta, 
	doublereal *c__, integer *ldc);

/* Subroutine */ int _starpu_dtbmv_(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *x, integer *incx);

/* Subroutine */ int _starpu_dtbsv_(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *x, integer *incx);

/* Subroutine */ int _starpu_dtpmv_(char *uplo, char *trans, char *diag, integer *n, 
	doublereal *ap, doublereal *x, integer *incx);

/* Subroutine */ int _starpu_dtpsv_(char *uplo, char *trans, char *diag, integer *n, 
	doublereal *ap, doublereal *x, integer *incx);

/* Subroutine */ int _starpu_dtrmm_(char *side, char *uplo, char *transa, char *diag, 
	integer *m, integer *n, doublereal *alpha, doublereal *a, integer *
	lda, doublereal *b, integer *ldb);

/* Subroutine */ int _starpu_dtrmv_(char *uplo, char *trans, char *diag, integer *n, 
	doublereal *a, integer *lda, doublereal *x, integer *incx);

/* Subroutine */ int _starpu_dtrsm_(char *side, char *uplo, char *transa, char *diag, 
	integer *m, integer *n, doublereal *alpha, doublereal *a, integer *
	lda, doublereal *b, integer *ldb);

/* Subroutine */ int _starpu_dtrsv_(char *uplo, char *trans, char *diag, integer *n, 
	doublereal *a, integer *lda, doublereal *x, integer *incx);

doublereal _starpu_dzasum_(integer *n, doublecomplex *zx, integer *incx);

doublereal _starpu_dznrm2_(integer *n, doublecomplex *x, integer *incx);

integer _starpu_icamax_(integer *n, complex *cx, integer *incx);

integer _starpu_idamax_(integer *n, doublereal *dx, integer *incx);

integer _starpu_isamax_(integer *n, real *sx, integer *incx);

integer _starpu_izamax_(integer *n, doublecomplex *zx, integer *incx);

logical _starpu_lsame_(char *ca, char *cb);

doublereal _starpu_sasum_(integer *n, real *sx, integer *incx);

/* Subroutine */ int _starpu_saxpy_(integer *n, real *sa, real *sx, integer *incx, 
	real *sy, integer *incy);

doublereal _starpu_scabs1_(complex *z__);

doublereal _starpu_scasum_(integer *n, complex *cx, integer *incx);

doublereal _starpu_scnrm2_(integer *n, complex *x, integer *incx);

/* Subroutine */ int _starpu_scopy_(integer *n, real *sx, integer *incx, real *sy, 
	integer *incy);

doublereal _starpu_sdot_(integer *n, real *sx, integer *incx, real *sy, integer *incy);

doublereal _starpu_sdsdot_(integer *n, real *sb, real *sx, integer *incx, real *sy, 
	integer *incy);

/* Subroutine */ int _starpu_sgbmv_(char *trans, integer *m, integer *n, integer *kl, 
	integer *ku, real *alpha, real *a, integer *lda, real *x, integer *
	incx, real *beta, real *y, integer *incy);

/* Subroutine */ int _starpu_sgemm_(char *transa, char *transb, integer *m, integer *
	n, integer *k, real *alpha, real *a, integer *lda, real *b, integer *
	ldb, real *beta, real *c__, integer *ldc);

/* Subroutine */ int _starpu_sgemv_(char *trans, integer *m, integer *n, real *alpha, 
	real *a, integer *lda, real *x, integer *incx, real *beta, real *y, 
	integer *incy);

/* Subroutine */ int _starpu_sger_(integer *m, integer *n, real *alpha, real *x, 
	integer *incx, real *y, integer *incy, real *a, integer *lda);

doublereal _starpu_snrm2_(integer *n, real *x, integer *incx);

/* Subroutine */ int _starpu_srot_(integer *n, real *sx, integer *incx, real *sy, 
	integer *incy, real *c__, real *s);

/* Subroutine */ int _starpu_srotg_(real *sa, real *sb, real *c__, real *s);

/* Subroutine */ int _starpu_srotm_(integer *n, real *sx, integer *incx, real *sy, 
	integer *incy, real *sparam);

/* Subroutine */ int _starpu_srotmg_(real *sd1, real *sd2, real *sx1, real *sy1, real 
	*sparam);

/* Subroutine */ int _starpu_ssbmv_(char *uplo, integer *n, integer *k, real *alpha, 
	real *a, integer *lda, real *x, integer *incx, real *beta, real *y, 
	integer *incy);

/* Subroutine */ int _starpu_sscal_(integer *n, real *sa, real *sx, integer *incx);

/* Subroutine */ int _starpu_sspmv_(char *uplo, integer *n, real *alpha, real *ap, 
	real *x, integer *incx, real *beta, real *y, integer *incy);

/* Subroutine */ int _starpu_sspr_(char *uplo, integer *n, real *alpha, real *x, 
	integer *incx, real *ap);

/* Subroutine */ int _starpu_sspr2_(char *uplo, integer *n, real *alpha, real *x, 
	integer *incx, real *y, integer *incy, real *ap);

/* Subroutine */ int _starpu_sswap_(integer *n, real *sx, integer *incx, real *sy, 
	integer *incy);

/* Subroutine */ int _starpu_ssymm_(char *side, char *uplo, integer *m, integer *n, 
	real *alpha, real *a, integer *lda, real *b, integer *ldb, real *beta, 
	 real *c__, integer *ldc);

/* Subroutine */ int _starpu_ssymv_(char *uplo, integer *n, real *alpha, real *a, 
	integer *lda, real *x, integer *incx, real *beta, real *y, integer *
	incy);

/* Subroutine */ int _starpu_ssyr_(char *uplo, integer *n, real *alpha, real *x, 
	integer *incx, real *a, integer *lda);

/* Subroutine */ int _starpu_ssyr2_(char *uplo, integer *n, real *alpha, real *x, 
	integer *incx, real *y, integer *incy, real *a, integer *lda);

/* Subroutine */ int _starpu_ssyr2k_(char *uplo, char *trans, integer *n, integer *k, 
	real *alpha, real *a, integer *lda, real *b, integer *ldb, real *beta, 
	 real *c__, integer *ldc);

/* Subroutine */ int _starpu_ssyrk_(char *uplo, char *trans, integer *n, integer *k, 
	real *alpha, real *a, integer *lda, real *beta, real *c__, integer *
	ldc);

/* Subroutine */ int _starpu_stbmv_(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, real *a, integer *lda, real *x, integer *incx);

/* Subroutine */ int _starpu_stbsv_(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, real *a, integer *lda, real *x, integer *incx);

/* Subroutine */ int _starpu_stpmv_(char *uplo, char *trans, char *diag, integer *n, 
	real *ap, real *x, integer *incx);

/* Subroutine */ int _starpu_stpsv_(char *uplo, char *trans, char *diag, integer *n, 
	real *ap, real *x, integer *incx);

/* Subroutine */ int _starpu_strmm_(char *side, char *uplo, char *transa, char *diag, 
	integer *m, integer *n, real *alpha, real *a, integer *lda, real *b, 
	integer *ldb);

/* Subroutine */ int _starpu_strmv_(char *uplo, char *trans, char *diag, integer *n, 
	real *a, integer *lda, real *x, integer *incx);

/* Subroutine */ int _starpu_strsm_(char *side, char *uplo, char *transa, char *diag, 
	integer *m, integer *n, real *alpha, real *a, integer *lda, real *b, 
	integer *ldb);

/* Subroutine */ int _starpu_strsv_(char *uplo, char *trans, char *diag, integer *n, 
	real *a, integer *lda, real *x, integer *incx);

/* Subroutine */ int _starpu_xerbla_(char *srname, integer *info);

/* Subroutine */ int _starpu_xerbla_array__(char *srname_array__, integer *
	srname_len__, integer *info, ftnlen srname_array_len);

/* Subroutine */ int _starpu_zaxpy_(integer *n, doublecomplex *za, doublecomplex *zx, 
	integer *incx, doublecomplex *zy, integer *incy);

/* Subroutine */ int _starpu_zcopy_(integer *n, doublecomplex *zx, integer *incx, 
	doublecomplex *zy, integer *incy);

/* Double Complex */ VOID _starpu_zdotc_(doublecomplex * ret_val, integer *n, 
	doublecomplex *zx, integer *incx, doublecomplex *zy, integer *incy);

/* Double Complex */ VOID _starpu_zdotu_(doublecomplex * ret_val, integer *n, 
	doublecomplex *zx, integer *incx, doublecomplex *zy, integer *incy);

/* Subroutine */ int _starpu_zdrot_(integer *n, doublecomplex *cx, integer *incx, 
	doublecomplex *cy, integer *incy, doublereal *c__, doublereal *s);

/* Subroutine */ int _starpu_zdscal_(integer *n, doublereal *da, doublecomplex *zx, 
	integer *incx);

/* Subroutine */ int _starpu_zgbmv_(char *trans, integer *m, integer *n, integer *kl, 
	integer *ku, doublecomplex *alpha, doublecomplex *a, integer *lda, 
	doublecomplex *x, integer *incx, doublecomplex *beta, doublecomplex *
	y, integer *incy);

/* Subroutine */ int _starpu_zgemm_(char *transa, char *transb, integer *m, integer *
	n, integer *k, doublecomplex *alpha, doublecomplex *a, integer *lda, 
	doublecomplex *b, integer *ldb, doublecomplex *beta, doublecomplex *
	c__, integer *ldc);

/* Subroutine */ int _starpu_zgemv_(char *trans, integer *m, integer *n, 
	doublecomplex *alpha, doublecomplex *a, integer *lda, doublecomplex *
	x, integer *incx, doublecomplex *beta, doublecomplex *y, integer *
	incy);

/* Subroutine */ int _starpu_zgerc_(integer *m, integer *n, doublecomplex *alpha, 
	doublecomplex *x, integer *incx, doublecomplex *y, integer *incy, 
	doublecomplex *a, integer *lda);

/* Subroutine */ int _starpu_zgeru_(integer *m, integer *n, doublecomplex *alpha, 
	doublecomplex *x, integer *incx, doublecomplex *y, integer *incy, 
	doublecomplex *a, integer *lda);

/* Subroutine */ int _starpu_zhbmv_(char *uplo, integer *n, integer *k, doublecomplex 
	*alpha, doublecomplex *a, integer *lda, doublecomplex *x, integer *
	incx, doublecomplex *beta, doublecomplex *y, integer *incy);

/* Subroutine */ int _starpu_zhemm_(char *side, char *uplo, integer *m, integer *n, 
	doublecomplex *alpha, doublecomplex *a, integer *lda, doublecomplex *
	b, integer *ldb, doublecomplex *beta, doublecomplex *c__, integer *
	ldc);

/* Subroutine */ int _starpu_zhemv_(char *uplo, integer *n, doublecomplex *alpha, 
	doublecomplex *a, integer *lda, doublecomplex *x, integer *incx, 
	doublecomplex *beta, doublecomplex *y, integer *incy);

/* Subroutine */ int _starpu_zher_(char *uplo, integer *n, doublereal *alpha, 
	doublecomplex *x, integer *incx, doublecomplex *a, integer *lda);

/* Subroutine */ int _starpu_zher2_(char *uplo, integer *n, doublecomplex *alpha, 
	doublecomplex *x, integer *incx, doublecomplex *y, integer *incy, 
	doublecomplex *a, integer *lda);

/* Subroutine */ int _starpu_zher2k_(char *uplo, char *trans, integer *n, integer *k, 
	doublecomplex *alpha, doublecomplex *a, integer *lda, doublecomplex *
	b, integer *ldb, doublereal *beta, doublecomplex *c__, integer *ldc);

/* Subroutine */ int _starpu_zherk_(char *uplo, char *trans, integer *n, integer *k, 
	doublereal *alpha, doublecomplex *a, integer *lda, doublereal *beta, 
	doublecomplex *c__, integer *ldc);

/* Subroutine */ int _starpu_zhpmv_(char *uplo, integer *n, doublecomplex *alpha, 
	doublecomplex *ap, doublecomplex *x, integer *incx, doublecomplex *
	beta, doublecomplex *y, integer *incy);

/* Subroutine */ int _starpu_zhpr_(char *uplo, integer *n, doublereal *alpha, 
	doublecomplex *x, integer *incx, doublecomplex *ap);

/* Subroutine */ int _starpu_zhpr2_(char *uplo, integer *n, doublecomplex *alpha, 
	doublecomplex *x, integer *incx, doublecomplex *y, integer *incy, 
	doublecomplex *ap);

/* Subroutine */ int _starpu_zrotg_(doublecomplex *ca, doublecomplex *cb, doublereal *
	c__, doublecomplex *s);

/* Subroutine */ int _starpu_zscal_(integer *n, doublecomplex *za, doublecomplex *zx, 
	integer *incx);

/* Subroutine */ int _starpu_zswap_(integer *n, doublecomplex *zx, integer *incx, 
	doublecomplex *zy, integer *incy);

/* Subroutine */ int _starpu_zsymm_(char *side, char *uplo, integer *m, integer *n, 
	doublecomplex *alpha, doublecomplex *a, integer *lda, doublecomplex *
	b, integer *ldb, doublecomplex *beta, doublecomplex *c__, integer *
	ldc);

/* Subroutine */ int _starpu_zsyr2k_(char *uplo, char *trans, integer *n, integer *k, 
	doublecomplex *alpha, doublecomplex *a, integer *lda, doublecomplex *
	b, integer *ldb, doublecomplex *beta, doublecomplex *c__, integer *
	ldc);

/* Subroutine */ int _starpu_zsyrk_(char *uplo, char *trans, integer *n, integer *k, 
	doublecomplex *alpha, doublecomplex *a, integer *lda, doublecomplex *
	beta, doublecomplex *c__, integer *ldc);

/* Subroutine */ int _starpu_ztbmv_(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, doublecomplex *a, integer *lda, doublecomplex *x, integer 
	*incx);

/* Subroutine */ int _starpu_ztbsv_(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, doublecomplex *a, integer *lda, doublecomplex *x, integer 
	*incx);

/* Subroutine */ int _starpu_ztpmv_(char *uplo, char *trans, char *diag, integer *n, 
	doublecomplex *ap, doublecomplex *x, integer *incx);

/* Subroutine */ int _starpu_ztpsv_(char *uplo, char *trans, char *diag, integer *n, 
	doublecomplex *ap, doublecomplex *x, integer *incx);

/* Subroutine */ int _starpu_ztrmm_(char *side, char *uplo, char *transa, char *diag, 
	integer *m, integer *n, doublecomplex *alpha, doublecomplex *a, 
	integer *lda, doublecomplex *b, integer *ldb);

/* Subroutine */ int _starpu_ztrmv_(char *uplo, char *trans, char *diag, integer *n, 
	doublecomplex *a, integer *lda, doublecomplex *x, integer *incx);

/* Subroutine */ int _starpu_ztrsm_(char *side, char *uplo, char *transa, char *diag, 
	integer *m, integer *n, doublecomplex *alpha, doublecomplex *a, 
	integer *lda, doublecomplex *b, integer *ldb);

/* Subroutine */ int _starpu_ztrsv_(char *uplo, char *trans, char *diag, integer *n, 
	doublecomplex *a, integer *lda, doublecomplex *x, integer *incx);

/* Subroutine */ int _starpu_cbdsqr_(char *uplo, integer *n, integer *ncvt, integer *
	nru, integer *ncc, real *d__, real *e, complex *vt, integer *ldvt, 
	complex *u, integer *ldu, complex *c__, integer *ldc, real *rwork, 
	integer *info);

/* Subroutine */ int _starpu_cgbbrd_(char *vect, integer *m, integer *n, integer *ncc, 
	 integer *kl, integer *ku, complex *ab, integer *ldab, real *d__, 
	real *e, complex *q, integer *ldq, complex *pt, integer *ldpt, 
	complex *c__, integer *ldc, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cgbcon_(char *norm, integer *n, integer *kl, integer *ku, 
	 complex *ab, integer *ldab, integer *ipiv, real *anorm, real *rcond, 
	complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cgbequ_(integer *m, integer *n, integer *kl, integer *ku, 
	 complex *ab, integer *ldab, real *r__, real *c__, real *rowcnd, real 
	*colcnd, real *amax, integer *info);

/* Subroutine */ int _starpu_cgbequb_(integer *m, integer *n, integer *kl, integer *
	ku, complex *ab, integer *ldab, real *r__, real *c__, real *rowcnd, 
	real *colcnd, real *amax, integer *info);

/* Subroutine */ int _starpu_cgbrfs_(char *trans, integer *n, integer *kl, integer *
	ku, integer *nrhs, complex *ab, integer *ldab, complex *afb, integer *
	ldafb, integer *ipiv, complex *b, integer *ldb, complex *x, integer *
	ldx, real *ferr, real *berr, complex *work, real *rwork, integer *
	info);

/* Subroutine */ int _starpu_cgbrfsx_(char *trans, char *equed, integer *n, integer *
	kl, integer *ku, integer *nrhs, complex *ab, integer *ldab, complex *
	afb, integer *ldafb, integer *ipiv, real *r__, real *c__, complex *b, 
	integer *ldb, complex *x, integer *ldx, real *rcond, real *berr, 
	integer *n_err_bnds__, real *err_bnds_norm__, real *err_bnds_comp__, 
	integer *nparams, real *params, complex *work, real *rwork, integer *
	info);

/* Subroutine */ int _starpu_cgbsv_(integer *n, integer *kl, integer *ku, integer *
	nrhs, complex *ab, integer *ldab, integer *ipiv, complex *b, integer *
	ldb, integer *info);

/* Subroutine */ int _starpu_cgbsvx_(char *fact, char *trans, integer *n, integer *kl, 
	 integer *ku, integer *nrhs, complex *ab, integer *ldab, complex *afb, 
	 integer *ldafb, integer *ipiv, char *equed, real *r__, real *c__, 
	complex *b, integer *ldb, complex *x, integer *ldx, real *rcond, real 
	*ferr, real *berr, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cgbsvxx_(char *fact, char *trans, integer *n, integer *
	kl, integer *ku, integer *nrhs, complex *ab, integer *ldab, complex *
	afb, integer *ldafb, integer *ipiv, char *equed, real *r__, real *c__, 
	 complex *b, integer *ldb, complex *x, integer *ldx, real *rcond, 
	real *rpvgrw, real *berr, integer *n_err_bnds__, real *
	err_bnds_norm__, real *err_bnds_comp__, integer *nparams, real *
	params, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cgbtf2_(integer *m, integer *n, integer *kl, integer *ku, 
	 complex *ab, integer *ldab, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_cgbtrf_(integer *m, integer *n, integer *kl, integer *ku, 
	 complex *ab, integer *ldab, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_cgbtrs_(char *trans, integer *n, integer *kl, integer *
	ku, integer *nrhs, complex *ab, integer *ldab, integer *ipiv, complex 
	*b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_cgebak_(char *job, char *side, integer *n, integer *ilo, 
	integer *ihi, real *scale, integer *m, complex *v, integer *ldv, 
	integer *info);

/* Subroutine */ int _starpu_cgebal_(char *job, integer *n, complex *a, integer *lda, 
	integer *ilo, integer *ihi, real *scale, integer *info);

/* Subroutine */ int _starpu_cgebd2_(integer *m, integer *n, complex *a, integer *lda, 
	 real *d__, real *e, complex *tauq, complex *taup, complex *work, 
	integer *info);

/* Subroutine */ int _starpu_cgebrd_(integer *m, integer *n, complex *a, integer *lda, 
	 real *d__, real *e, complex *tauq, complex *taup, complex *work, 
	integer *lwork, integer *info);

/* Subroutine */ int _starpu_cgecon_(char *norm, integer *n, complex *a, integer *lda, 
	 real *anorm, real *rcond, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cgeequ_(integer *m, integer *n, complex *a, integer *lda, 
	 real *r__, real *c__, real *rowcnd, real *colcnd, real *amax, 
	integer *info);

/* Subroutine */ int _starpu_cgeequb_(integer *m, integer *n, complex *a, integer *
	lda, real *r__, real *c__, real *rowcnd, real *colcnd, real *amax, 
	integer *info);

/* Subroutine */ int _starpu_cgees_(char *jobvs, char *sort, L_fp select, integer *n, 
	complex *a, integer *lda, integer *sdim, complex *w, complex *vs, 
	integer *ldvs, complex *work, integer *lwork, real *rwork, logical *
	bwork, integer *info);

/* Subroutine */ int _starpu_cgeesx_(char *jobvs, char *sort, L_fp select, char *
	sense, integer *n, complex *a, integer *lda, integer *sdim, complex *
	w, complex *vs, integer *ldvs, real *rconde, real *rcondv, complex *
	work, integer *lwork, real *rwork, logical *bwork, integer *info);

/* Subroutine */ int _starpu_cgeev_(char *jobvl, char *jobvr, integer *n, complex *a, 
	integer *lda, complex *w, complex *vl, integer *ldvl, complex *vr, 
	integer *ldvr, complex *work, integer *lwork, real *rwork, integer *
	info);

/* Subroutine */ int _starpu_cgeevx_(char *balanc, char *jobvl, char *jobvr, char *
	sense, integer *n, complex *a, integer *lda, complex *w, complex *vl, 
	integer *ldvl, complex *vr, integer *ldvr, integer *ilo, integer *ihi, 
	 real *scale, real *abnrm, real *rconde, real *rcondv, complex *work, 
	integer *lwork, real *rwork, integer *info);

/* Subroutine */ int _starpu_cgegs_(char *jobvsl, char *jobvsr, integer *n, complex *
	a, integer *lda, complex *b, integer *ldb, complex *alpha, complex *
	beta, complex *vsl, integer *ldvsl, complex *vsr, integer *ldvsr, 
	complex *work, integer *lwork, real *rwork, integer *info);

/* Subroutine */ int _starpu_cgegv_(char *jobvl, char *jobvr, integer *n, complex *a, 
	integer *lda, complex *b, integer *ldb, complex *alpha, complex *beta, 
	 complex *vl, integer *ldvl, complex *vr, integer *ldvr, complex *
	work, integer *lwork, real *rwork, integer *info);

/* Subroutine */ int _starpu_cgehd2_(integer *n, integer *ilo, integer *ihi, complex *
	a, integer *lda, complex *tau, complex *work, integer *info);

/* Subroutine */ int _starpu_cgehrd_(integer *n, integer *ilo, integer *ihi, complex *
	a, integer *lda, complex *tau, complex *work, integer *lwork, integer 
	*info);

/* Subroutine */ int _starpu_cgelq2_(integer *m, integer *n, complex *a, integer *lda, 
	 complex *tau, complex *work, integer *info);

/* Subroutine */ int _starpu_cgelqf_(integer *m, integer *n, complex *a, integer *lda, 
	 complex *tau, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cgels_(char *trans, integer *m, integer *n, integer *
	nrhs, complex *a, integer *lda, complex *b, integer *ldb, complex *
	work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cgelsd_(integer *m, integer *n, integer *nrhs, complex *
	a, integer *lda, complex *b, integer *ldb, real *s, real *rcond, 
	integer *rank, complex *work, integer *lwork, real *rwork, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_cgelss_(integer *m, integer *n, integer *nrhs, complex *
	a, integer *lda, complex *b, integer *ldb, real *s, real *rcond, 
	integer *rank, complex *work, integer *lwork, real *rwork, integer *
	info);

/* Subroutine */ int _starpu_cgelsx_(integer *m, integer *n, integer *nrhs, complex *
	a, integer *lda, complex *b, integer *ldb, integer *jpvt, real *rcond, 
	 integer *rank, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cgelsy_(integer *m, integer *n, integer *nrhs, complex *
	a, integer *lda, complex *b, integer *ldb, integer *jpvt, real *rcond, 
	 integer *rank, complex *work, integer *lwork, real *rwork, integer *
	info);

/* Subroutine */ int _starpu_cgeql2_(integer *m, integer *n, complex *a, integer *lda, 
	 complex *tau, complex *work, integer *info);

/* Subroutine */ int _starpu_cgeqlf_(integer *m, integer *n, complex *a, integer *lda, 
	 complex *tau, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cgeqp3_(integer *m, integer *n, complex *a, integer *lda, 
	 integer *jpvt, complex *tau, complex *work, integer *lwork, real *
	rwork, integer *info);

/* Subroutine */ int _starpu_cgeqpf_(integer *m, integer *n, complex *a, integer *lda, 
	 integer *jpvt, complex *tau, complex *work, real *rwork, integer *
	info);

/* Subroutine */ int _starpu_cgeqr2_(integer *m, integer *n, complex *a, integer *lda, 
	 complex *tau, complex *work, integer *info);

/* Subroutine */ int _starpu_cgeqrf_(integer *m, integer *n, complex *a, integer *lda, 
	 complex *tau, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cgerfs_(char *trans, integer *n, integer *nrhs, complex *
	a, integer *lda, complex *af, integer *ldaf, integer *ipiv, complex *
	b, integer *ldb, complex *x, integer *ldx, real *ferr, real *berr, 
	complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cgerfsx_(char *trans, char *equed, integer *n, integer *
	nrhs, complex *a, integer *lda, complex *af, integer *ldaf, integer *
	ipiv, real *r__, real *c__, complex *b, integer *ldb, complex *x, 
	integer *ldx, real *rcond, real *berr, integer *n_err_bnds__, real *
	err_bnds_norm__, real *err_bnds_comp__, integer *nparams, real *
	params, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cgerq2_(integer *m, integer *n, complex *a, integer *lda, 
	 complex *tau, complex *work, integer *info);

/* Subroutine */ int _starpu_cgerqf_(integer *m, integer *n, complex *a, integer *lda, 
	 complex *tau, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cgesc2_(integer *n, complex *a, integer *lda, complex *
	rhs, integer *ipiv, integer *jpiv, real *scale);

/* Subroutine */ int _starpu_cgesdd_(char *jobz, integer *m, integer *n, complex *a, 
	integer *lda, real *s, complex *u, integer *ldu, complex *vt, integer 
	*ldvt, complex *work, integer *lwork, real *rwork, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_cgesv_(integer *n, integer *nrhs, complex *a, integer *
	lda, integer *ipiv, complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_cgesvd_(char *jobu, char *jobvt, integer *m, integer *n, 
	complex *a, integer *lda, real *s, complex *u, integer *ldu, complex *
	vt, integer *ldvt, complex *work, integer *lwork, real *rwork, 
	integer *info);

/* Subroutine */ int _starpu_cgesvx_(char *fact, char *trans, integer *n, integer *
	nrhs, complex *a, integer *lda, complex *af, integer *ldaf, integer *
	ipiv, char *equed, real *r__, real *c__, complex *b, integer *ldb, 
	complex *x, integer *ldx, real *rcond, real *ferr, real *berr, 
	complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cgesvxx_(char *fact, char *trans, integer *n, integer *
	nrhs, complex *a, integer *lda, complex *af, integer *ldaf, integer *
	ipiv, char *equed, real *r__, real *c__, complex *b, integer *ldb, 
	complex *x, integer *ldx, real *rcond, real *rpvgrw, real *berr, 
	integer *n_err_bnds__, real *err_bnds_norm__, real *err_bnds_comp__, 
	integer *nparams, real *params, complex *work, real *rwork, integer *
	info);

/* Subroutine */ int _starpu_cgetc2_(integer *n, complex *a, integer *lda, integer *
	ipiv, integer *jpiv, integer *info);

/* Subroutine */ int _starpu_cgetf2_(integer *m, integer *n, complex *a, integer *lda, 
	 integer *ipiv, integer *info);

/* Subroutine */ int _starpu_cgetrf_(integer *m, integer *n, complex *a, integer *lda, 
	 integer *ipiv, integer *info);

/* Subroutine */ int _starpu_cgetri_(integer *n, complex *a, integer *lda, integer *
	ipiv, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cgetrs_(char *trans, integer *n, integer *nrhs, complex *
	a, integer *lda, integer *ipiv, complex *b, integer *ldb, integer *
	info);

/* Subroutine */ int _starpu_cggbak_(char *job, char *side, integer *n, integer *ilo, 
	integer *ihi, real *lscale, real *rscale, integer *m, complex *v, 
	integer *ldv, integer *info);

/* Subroutine */ int _starpu_cggbal_(char *job, integer *n, complex *a, integer *lda, 
	complex *b, integer *ldb, integer *ilo, integer *ihi, real *lscale, 
	real *rscale, real *work, integer *info);

/* Subroutine */ int _starpu_cgges_(char *jobvsl, char *jobvsr, char *sort, L_fp 
	selctg, integer *n, complex *a, integer *lda, complex *b, integer *
	ldb, integer *sdim, complex *alpha, complex *beta, complex *vsl, 
	integer *ldvsl, complex *vsr, integer *ldvsr, complex *work, integer *
	lwork, real *rwork, logical *bwork, integer *info);

/* Subroutine */ int _starpu_cggesx_(char *jobvsl, char *jobvsr, char *sort, L_fp 
	selctg, char *sense, integer *n, complex *a, integer *lda, complex *b, 
	 integer *ldb, integer *sdim, complex *alpha, complex *beta, complex *
	vsl, integer *ldvsl, complex *vsr, integer *ldvsr, real *rconde, real 
	*rcondv, complex *work, integer *lwork, real *rwork, integer *iwork, 
	integer *liwork, logical *bwork, integer *info);

/* Subroutine */ int _starpu_cggev_(char *jobvl, char *jobvr, integer *n, complex *a, 
	integer *lda, complex *b, integer *ldb, complex *alpha, complex *beta, 
	 complex *vl, integer *ldvl, complex *vr, integer *ldvr, complex *
	work, integer *lwork, real *rwork, integer *info);

/* Subroutine */ int _starpu_cggevx_(char *balanc, char *jobvl, char *jobvr, char *
	sense, integer *n, complex *a, integer *lda, complex *b, integer *ldb, 
	 complex *alpha, complex *beta, complex *vl, integer *ldvl, complex *
	vr, integer *ldvr, integer *ilo, integer *ihi, real *lscale, real *
	rscale, real *abnrm, real *bbnrm, real *rconde, real *rcondv, complex 
	*work, integer *lwork, real *rwork, integer *iwork, logical *bwork, 
	integer *info);

/* Subroutine */ int _starpu_cggglm_(integer *n, integer *m, integer *p, complex *a, 
	integer *lda, complex *b, integer *ldb, complex *d__, complex *x, 
	complex *y, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cgghrd_(char *compq, char *compz, integer *n, integer *
	ilo, integer *ihi, complex *a, integer *lda, complex *b, integer *ldb, 
	 complex *q, integer *ldq, complex *z__, integer *ldz, integer *info);

/* Subroutine */ int _starpu_cgglse_(integer *m, integer *n, integer *p, complex *a, 
	integer *lda, complex *b, integer *ldb, complex *c__, complex *d__, 
	complex *x, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cggqrf_(integer *n, integer *m, integer *p, complex *a, 
	integer *lda, complex *taua, complex *b, integer *ldb, complex *taub, 
	complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cggrqf_(integer *m, integer *p, integer *n, complex *a, 
	integer *lda, complex *taua, complex *b, integer *ldb, complex *taub, 
	complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cggsvd_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *n, integer *p, integer *k, integer *l, complex *a, integer *
	lda, complex *b, integer *ldb, real *alpha, real *beta, complex *u, 
	integer *ldu, complex *v, integer *ldv, complex *q, integer *ldq, 
	complex *work, real *rwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_cggsvp_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *p, integer *n, complex *a, integer *lda, complex *b, integer 
	*ldb, real *tola, real *tolb, integer *k, integer *l, complex *u, 
	integer *ldu, complex *v, integer *ldv, complex *q, integer *ldq, 
	integer *iwork, real *rwork, complex *tau, complex *work, integer *
	info);

/* Subroutine */ int _starpu_cgtcon_(char *norm, integer *n, complex *dl, complex *
	d__, complex *du, complex *du2, integer *ipiv, real *anorm, real *
	rcond, complex *work, integer *info);

/* Subroutine */ int _starpu_cgtrfs_(char *trans, integer *n, integer *nrhs, complex *
	dl, complex *d__, complex *du, complex *dlf, complex *df, complex *
	duf, complex *du2, integer *ipiv, complex *b, integer *ldb, complex *
	x, integer *ldx, real *ferr, real *berr, complex *work, real *rwork, 
	integer *info);

/* Subroutine */ int _starpu_cgtsv_(integer *n, integer *nrhs, complex *dl, complex *
	d__, complex *du, complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_cgtsvx_(char *fact, char *trans, integer *n, integer *
	nrhs, complex *dl, complex *d__, complex *du, complex *dlf, complex *
	df, complex *duf, complex *du2, integer *ipiv, complex *b, integer *
	ldb, complex *x, integer *ldx, real *rcond, real *ferr, real *berr, 
	complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cgttrf_(integer *n, complex *dl, complex *d__, complex *
	du, complex *du2, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_cgttrs_(char *trans, integer *n, integer *nrhs, complex *
	dl, complex *d__, complex *du, complex *du2, integer *ipiv, complex *
	b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_cgtts2_(integer *itrans, integer *n, integer *nrhs, 
	complex *dl, complex *d__, complex *du, complex *du2, integer *ipiv, 
	complex *b, integer *ldb);

/* Subroutine */ int _starpu_chbev_(char *jobz, char *uplo, integer *n, integer *kd, 
	complex *ab, integer *ldab, real *w, complex *z__, integer *ldz, 
	complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_chbevd_(char *jobz, char *uplo, integer *n, integer *kd, 
	complex *ab, integer *ldab, real *w, complex *z__, integer *ldz, 
	complex *work, integer *lwork, real *rwork, integer *lrwork, integer *
	iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_chbevx_(char *jobz, char *range, char *uplo, integer *n, 
	integer *kd, complex *ab, integer *ldab, complex *q, integer *ldq, 
	real *vl, real *vu, integer *il, integer *iu, real *abstol, integer *
	m, real *w, complex *z__, integer *ldz, complex *work, real *rwork, 
	integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu_chbgst_(char *vect, char *uplo, integer *n, integer *ka, 
	integer *kb, complex *ab, integer *ldab, complex *bb, integer *ldbb, 
	complex *x, integer *ldx, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_chbgv_(char *jobz, char *uplo, integer *n, integer *ka, 
	integer *kb, complex *ab, integer *ldab, complex *bb, integer *ldbb, 
	real *w, complex *z__, integer *ldz, complex *work, real *rwork, 
	integer *info);

/* Subroutine */ int _starpu_chbgvd_(char *jobz, char *uplo, integer *n, integer *ka, 
	integer *kb, complex *ab, integer *ldab, complex *bb, integer *ldbb, 
	real *w, complex *z__, integer *ldz, complex *work, integer *lwork, 
	real *rwork, integer *lrwork, integer *iwork, integer *liwork, 
	integer *info);

/* Subroutine */ int _starpu_chbgvx_(char *jobz, char *range, char *uplo, integer *n, 
	integer *ka, integer *kb, complex *ab, integer *ldab, complex *bb, 
	integer *ldbb, complex *q, integer *ldq, real *vl, real *vu, integer *
	il, integer *iu, real *abstol, integer *m, real *w, complex *z__, 
	integer *ldz, complex *work, real *rwork, integer *iwork, integer *
	ifail, integer *info);

/* Subroutine */ int _starpu_chbtrd_(char *vect, char *uplo, integer *n, integer *kd, 
	complex *ab, integer *ldab, real *d__, real *e, complex *q, integer *
	ldq, complex *work, integer *info);

/* Subroutine */ int _starpu_checon_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *ipiv, real *anorm, real *rcond, complex *work, integer *
	info);

/* Subroutine */ int _starpu_cheequb_(char *uplo, integer *n, complex *a, integer *
	lda, real *s, real *scond, real *amax, complex *work, integer *info);

/* Subroutine */ int _starpu_cheev_(char *jobz, char *uplo, integer *n, complex *a, 
	integer *lda, real *w, complex *work, integer *lwork, real *rwork, 
	integer *info);

/* Subroutine */ int _starpu_cheevd_(char *jobz, char *uplo, integer *n, complex *a, 
	integer *lda, real *w, complex *work, integer *lwork, real *rwork, 
	integer *lrwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_cheevr_(char *jobz, char *range, char *uplo, integer *n, 
	complex *a, integer *lda, real *vl, real *vu, integer *il, integer *
	iu, real *abstol, integer *m, real *w, complex *z__, integer *ldz, 
	integer *isuppz, complex *work, integer *lwork, real *rwork, integer *
	lrwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_cheevx_(char *jobz, char *range, char *uplo, integer *n, 
	complex *a, integer *lda, real *vl, real *vu, integer *il, integer *
	iu, real *abstol, integer *m, real *w, complex *z__, integer *ldz, 
	complex *work, integer *lwork, real *rwork, integer *iwork, integer *
	ifail, integer *info);

/* Subroutine */ int _starpu_chegs2_(integer *itype, char *uplo, integer *n, complex *
	a, integer *lda, complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_chegst_(integer *itype, char *uplo, integer *n, complex *
	a, integer *lda, complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_chegv_(integer *itype, char *jobz, char *uplo, integer *
	n, complex *a, integer *lda, complex *b, integer *ldb, real *w, 
	complex *work, integer *lwork, real *rwork, integer *info);

/* Subroutine */ int _starpu_chegvd_(integer *itype, char *jobz, char *uplo, integer *
	n, complex *a, integer *lda, complex *b, integer *ldb, real *w, 
	complex *work, integer *lwork, real *rwork, integer *lrwork, integer *
	iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_chegvx_(integer *itype, char *jobz, char *range, char *
	uplo, integer *n, complex *a, integer *lda, complex *b, integer *ldb, 
	real *vl, real *vu, integer *il, integer *iu, real *abstol, integer *
	m, real *w, complex *z__, integer *ldz, complex *work, integer *lwork, 
	 real *rwork, integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu_cherfs_(char *uplo, integer *n, integer *nrhs, complex *
	a, integer *lda, complex *af, integer *ldaf, integer *ipiv, complex *
	b, integer *ldb, complex *x, integer *ldx, real *ferr, real *berr, 
	complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cherfsx_(char *uplo, char *equed, integer *n, integer *
	nrhs, complex *a, integer *lda, complex *af, integer *ldaf, integer *
	ipiv, real *s, complex *b, integer *ldb, complex *x, integer *ldx, 
	real *rcond, real *berr, integer *n_err_bnds__, real *err_bnds_norm__, 
	 real *err_bnds_comp__, integer *nparams, real *params, complex *work, 
	 real *rwork, integer *info);

/* Subroutine */ int _starpu_chesv_(char *uplo, integer *n, integer *nrhs, complex *a, 
	 integer *lda, integer *ipiv, complex *b, integer *ldb, complex *work, 
	 integer *lwork, integer *info);

/* Subroutine */ int _starpu_chesvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, complex *a, integer *lda, complex *af, integer *ldaf, integer *
	ipiv, complex *b, integer *ldb, complex *x, integer *ldx, real *rcond, 
	 real *ferr, real *berr, complex *work, integer *lwork, real *rwork, 
	integer *info);

/* Subroutine */ int _starpu_chesvxx_(char *fact, char *uplo, integer *n, integer *
	nrhs, complex *a, integer *lda, complex *af, integer *ldaf, integer *
	ipiv, char *equed, real *s, complex *b, integer *ldb, complex *x, 
	integer *ldx, real *rcond, real *rpvgrw, real *berr, integer *
	n_err_bnds__, real *err_bnds_norm__, real *err_bnds_comp__, integer *
	nparams, real *params, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_chetd2_(char *uplo, integer *n, complex *a, integer *lda, 
	 real *d__, real *e, complex *tau, integer *info);

/* Subroutine */ int _starpu_chetf2_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *ipiv, integer *info);

/* Subroutine */ int _starpu_chetrd_(char *uplo, integer *n, complex *a, integer *lda, 
	 real *d__, real *e, complex *tau, complex *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_chetrf_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *ipiv, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_chetri_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *ipiv, complex *work, integer *info);

/* Subroutine */ int _starpu_chetrs_(char *uplo, integer *n, integer *nrhs, complex *
	a, integer *lda, integer *ipiv, complex *b, integer *ldb, integer *
	info);

/* Subroutine */ int _starpu_chfrk_(char *transr, char *uplo, char *trans, integer *n, 
	 integer *k, real *alpha, complex *a, integer *lda, real *beta, 
	complex *c__);

/* Subroutine */ int _starpu_chgeqz_(char *job, char *compq, char *compz, integer *n, 
	integer *ilo, integer *ihi, complex *h__, integer *ldh, complex *t, 
	integer *ldt, complex *alpha, complex *beta, complex *q, integer *ldq, 
	 complex *z__, integer *ldz, complex *work, integer *lwork, real *
	rwork, integer *info);

/* Character */ VOID _starpu_chla_transtype__(char *ret_val, ftnlen ret_val_len, 
	integer *trans);

/* Subroutine */ int _starpu_chpcon_(char *uplo, integer *n, complex *ap, integer *
	ipiv, real *anorm, real *rcond, complex *work, integer *info);

/* Subroutine */ int _starpu_chpev_(char *jobz, char *uplo, integer *n, complex *ap, 
	real *w, complex *z__, integer *ldz, complex *work, real *rwork, 
	integer *info);

/* Subroutine */ int _starpu_chpevd_(char *jobz, char *uplo, integer *n, complex *ap, 
	real *w, complex *z__, integer *ldz, complex *work, integer *lwork, 
	real *rwork, integer *lrwork, integer *iwork, integer *liwork, 
	integer *info);

/* Subroutine */ int _starpu_chpevx_(char *jobz, char *range, char *uplo, integer *n, 
	complex *ap, real *vl, real *vu, integer *il, integer *iu, real *
	abstol, integer *m, real *w, complex *z__, integer *ldz, complex *
	work, real *rwork, integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu_chpgst_(integer *itype, char *uplo, integer *n, complex *
	ap, complex *bp, integer *info);

/* Subroutine */ int _starpu_chpgv_(integer *itype, char *jobz, char *uplo, integer *
	n, complex *ap, complex *bp, real *w, complex *z__, integer *ldz, 
	complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_chpgvd_(integer *itype, char *jobz, char *uplo, integer *
	n, complex *ap, complex *bp, real *w, complex *z__, integer *ldz, 
	complex *work, integer *lwork, real *rwork, integer *lrwork, integer *
	iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_chpgvx_(integer *itype, char *jobz, char *range, char *
	uplo, integer *n, complex *ap, complex *bp, real *vl, real *vu, 
	integer *il, integer *iu, real *abstol, integer *m, real *w, complex *
	z__, integer *ldz, complex *work, real *rwork, integer *iwork, 
	integer *ifail, integer *info);

/* Subroutine */ int _starpu_chprfs_(char *uplo, integer *n, integer *nrhs, complex *
	ap, complex *afp, integer *ipiv, complex *b, integer *ldb, complex *x, 
	 integer *ldx, real *ferr, real *berr, complex *work, real *rwork, 
	integer *info);

/* Subroutine */ int _starpu_chpsv_(char *uplo, integer *n, integer *nrhs, complex *
	ap, integer *ipiv, complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_chpsvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, complex *ap, complex *afp, integer *ipiv, complex *b, integer *
	ldb, complex *x, integer *ldx, real *rcond, real *ferr, real *berr, 
	complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_chptrd_(char *uplo, integer *n, complex *ap, real *d__, 
	real *e, complex *tau, integer *info);

/* Subroutine */ int _starpu_chptrf_(char *uplo, integer *n, complex *ap, integer *
	ipiv, integer *info);

/* Subroutine */ int _starpu_chptri_(char *uplo, integer *n, complex *ap, integer *
	ipiv, complex *work, integer *info);

/* Subroutine */ int _starpu_chptrs_(char *uplo, integer *n, integer *nrhs, complex *
	ap, integer *ipiv, complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_chsein_(char *side, char *eigsrc, char *initv, logical *
	select, integer *n, complex *h__, integer *ldh, complex *w, complex *
	vl, integer *ldvl, complex *vr, integer *ldvr, integer *mm, integer *
	m, complex *work, real *rwork, integer *ifaill, integer *ifailr, 
	integer *info);

/* Subroutine */ int _starpu_chseqr_(char *job, char *compz, integer *n, integer *ilo, 
	 integer *ihi, complex *h__, integer *ldh, complex *w, complex *z__, 
	integer *ldz, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cla_gbamv__(integer *trans, integer *m, integer *n, 
	integer *kl, integer *ku, real *alpha, complex *ab, integer *ldab, 
	complex *x, integer *incx, real *beta, real *y, integer *incy);

doublereal _starpu_cla_gbrcond_c__(char *trans, integer *n, integer *kl, integer *ku, 
	complex *ab, integer *ldab, complex *afb, integer *ldafb, integer *
	ipiv, real *c__, logical *capply, integer *info, complex *work, real *
	rwork, ftnlen trans_len);

doublereal _starpu_cla_gbrcond_x__(char *trans, integer *n, integer *kl, integer *ku, 
	complex *ab, integer *ldab, complex *afb, integer *ldafb, integer *
	ipiv, complex *x, integer *info, complex *work, real *rwork, ftnlen 
	trans_len);

/* Subroutine */ int _starpu_cla_gbrfsx_extended__(integer *prec_type__, integer *
	trans_type__, integer *n, integer *kl, integer *ku, integer *nrhs, 
	complex *ab, integer *ldab, complex *afb, integer *ldafb, integer *
	ipiv, logical *colequ, real *c__, complex *b, integer *ldb, complex *
	y, integer *ldy, real *berr_out__, integer *n_norms__, real *errs_n__,
	 real *errs_c__, complex *res, real *ayb, complex *dy, complex *
	y_tail__, real *rcond, integer *ithresh, real *rthresh, real *dz_ub__,
	 logical *ignore_cwise__, integer *info);

doublereal _starpu_cla_gbrpvgrw__(integer *n, integer *kl, integer *ku, integer *
	ncols, complex *ab, integer *ldab, complex *afb, integer *ldafb);

/* Subroutine */ int _starpu_cla_geamv__(integer *trans, integer *m, integer *n, real 
	*alpha, complex *a, integer *lda, complex *x, integer *incx, real *
	beta, real *y, integer *incy);

doublereal _starpu_cla_gercond_c__(char *trans, integer *n, complex *a, integer *lda, 
	complex *af, integer *ldaf, integer *ipiv, real *c__, logical *capply,
	 integer *info, complex *work, real *rwork, ftnlen trans_len);

doublereal _starpu_cla_gercond_x__(char *trans, integer *n, complex *a, integer *lda, 
	complex *af, integer *ldaf, integer *ipiv, complex *x, integer *info, 
	complex *work, real *rwork, ftnlen trans_len);

/* Subroutine */ int _starpu_cla_gerfsx_extended__(integer *prec_type__, integer *
	trans_type__, integer *n, integer *nrhs, complex *a, integer *lda, 
	complex *af, integer *ldaf, integer *ipiv, logical *colequ, real *c__,
	 complex *b, integer *ldb, complex *y, integer *ldy, real *berr_out__,
	 integer *n_norms__, real *errs_n__, real *errs_c__, complex *res, 
	real *ayb, complex *dy, complex *y_tail__, real *rcond, integer *
	ithresh, real *rthresh, real *dz_ub__, logical *ignore_cwise__, 
	integer *info);

/* Subroutine */ int _starpu_cla_heamv__(integer *uplo, integer *n, real *alpha, 
	complex *a, integer *lda, complex *x, integer *incx, real *beta, real 
	*y, integer *incy);

doublereal _starpu_cla_hercond_c__(char *uplo, integer *n, complex *a, integer *lda, 
	complex *af, integer *ldaf, integer *ipiv, real *c__, logical *capply,
	 integer *info, complex *work, real *rwork, ftnlen uplo_len);

doublereal _starpu_cla_hercond_x__(char *uplo, integer *n, complex *a, integer *lda, 
	complex *af, integer *ldaf, integer *ipiv, complex *x, integer *info, 
	complex *work, real *rwork, ftnlen uplo_len);

/* Subroutine */ int _starpu_cla_herfsx_extended__(integer *prec_type__, char *uplo, 
	integer *n, integer *nrhs, complex *a, integer *lda, complex *af, 
	integer *ldaf, integer *ipiv, logical *colequ, real *c__, complex *b, 
	integer *ldb, complex *y, integer *ldy, real *berr_out__, integer *
	n_norms__, real *errs_n__, real *errs_c__, complex *res, real *ayb, 
	complex *dy, complex *y_tail__, real *rcond, integer *ithresh, real *
	rthresh, real *dz_ub__, logical *ignore_cwise__, integer *info, 
	ftnlen uplo_len);

doublereal _starpu_cla_herpvgrw__(char *uplo, integer *n, integer *info, complex *a, 
	integer *lda, complex *af, integer *ldaf, integer *ipiv, real *work, 
	ftnlen uplo_len);

/* Subroutine */ int _starpu_cla_lin_berr__(integer *n, integer *nz, integer *nrhs, 
	complex *res, real *ayb, real *berr);

doublereal _starpu_cla_porcond_c__(char *uplo, integer *n, complex *a, integer *lda, 
	complex *af, integer *ldaf, real *c__, logical *capply, integer *info,
	 complex *work, real *rwork, ftnlen uplo_len);

doublereal _starpu_cla_porcond_x__(char *uplo, integer *n, complex *a, integer *lda, 
	complex *af, integer *ldaf, complex *x, integer *info, complex *work, 
	real *rwork, ftnlen uplo_len);

/* Subroutine */ int _starpu_cla_porfsx_extended__(integer *prec_type__, char *uplo, 
	integer *n, integer *nrhs, complex *a, integer *lda, complex *af, 
	integer *ldaf, logical *colequ, real *c__, complex *b, integer *ldb, 
	complex *y, integer *ldy, real *berr_out__, integer *n_norms__, real *
	errs_n__, real *errs_c__, complex *res, real *ayb, complex *dy, 
	complex *y_tail__, real *rcond, integer *ithresh, real *rthresh, real 
	*dz_ub__, logical *ignore_cwise__, integer *info, ftnlen uplo_len);

doublereal _starpu_cla_porpvgrw__(char *uplo, integer *ncols, complex *a, integer *
	lda, complex *af, integer *ldaf, real *work, ftnlen uplo_len);

doublereal _starpu_cla_rpvgrw__(integer *n, integer *ncols, complex *a, integer *lda, 
	complex *af, integer *ldaf);

/* Subroutine */ int _starpu_cla_syamv__(integer *uplo, integer *n, real *alpha, 
	complex *a, integer *lda, complex *x, integer *incx, real *beta, real 
	*y, integer *incy);

doublereal _starpu_cla_syrcond_c__(char *uplo, integer *n, complex *a, integer *lda, 
	complex *af, integer *ldaf, integer *ipiv, real *c__, logical *capply,
	 integer *info, complex *work, real *rwork, ftnlen uplo_len);

doublereal _starpu_cla_syrcond_x__(char *uplo, integer *n, complex *a, integer *lda, 
	complex *af, integer *ldaf, integer *ipiv, complex *x, integer *info, 
	complex *work, real *rwork, ftnlen uplo_len);

/* Subroutine */ int _starpu_cla_syrfsx_extended__(integer *prec_type__, char *uplo, 
	integer *n, integer *nrhs, complex *a, integer *lda, complex *af, 
	integer *ldaf, integer *ipiv, logical *colequ, real *c__, complex *b, 
	integer *ldb, complex *y, integer *ldy, real *berr_out__, integer *
	n_norms__, real *errs_n__, real *errs_c__, complex *res, real *ayb, 
	complex *dy, complex *y_tail__, real *rcond, integer *ithresh, real *
	rthresh, real *dz_ub__, logical *ignore_cwise__, integer *info, 
	ftnlen uplo_len);

doublereal _starpu_cla_syrpvgrw__(char *uplo, integer *n, integer *info, complex *a, 
	integer *lda, complex *af, integer *ldaf, integer *ipiv, real *work, 
	ftnlen uplo_len);

/* Subroutine */ int _starpu_cla_wwaddw__(integer *n, complex *x, complex *y, complex 
	*w);

/* Subroutine */ int _starpu_clabrd_(integer *m, integer *n, integer *nb, complex *a, 
	integer *lda, real *d__, real *e, complex *tauq, complex *taup, 
	complex *x, integer *ldx, complex *y, integer *ldy);

/* Subroutine */ int _starpu_clacgv_(integer *n, complex *x, integer *incx);

/* Subroutine */ int _starpu_clacn2_(integer *n, complex *v, complex *x, real *est, 
	integer *kase, integer *isave);

/* Subroutine */ int _starpu_clacon_(integer *n, complex *v, complex *x, real *est, 
	integer *kase);

/* Subroutine */ int _starpu_clacp2_(char *uplo, integer *m, integer *n, real *a, 
	integer *lda, complex *b, integer *ldb);

/* Subroutine */ int _starpu_clacpy_(char *uplo, integer *m, integer *n, complex *a, 
	integer *lda, complex *b, integer *ldb);

/* Subroutine */ int _starpu_clacrm_(integer *m, integer *n, complex *a, integer *lda, 
	 real *b, integer *ldb, complex *c__, integer *ldc, real *rwork);

/* Subroutine */ int _starpu_clacrt_(integer *n, complex *cx, integer *incx, complex *
	cy, integer *incy, complex *c__, complex *s);

/* Complex */ VOID _starpu_cladiv_(complex * ret_val, complex *x, complex *y);

/* Subroutine */ int _starpu_claed0_(integer *qsiz, integer *n, real *d__, real *e, 
	complex *q, integer *ldq, complex *qstore, integer *ldqs, real *rwork, 
	 integer *iwork, integer *info);

/* Subroutine */ int _starpu_claed7_(integer *n, integer *cutpnt, integer *qsiz, 
	integer *tlvls, integer *curlvl, integer *curpbm, real *d__, complex *
	q, integer *ldq, real *rho, integer *indxq, real *qstore, integer *
	qptr, integer *prmptr, integer *perm, integer *givptr, integer *
	givcol, real *givnum, complex *work, real *rwork, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_claed8_(integer *k, integer *n, integer *qsiz, complex *
	q, integer *ldq, real *d__, real *rho, integer *cutpnt, real *z__, 
	real *dlamda, complex *q2, integer *ldq2, real *w, integer *indxp, 
	integer *indx, integer *indxq, integer *perm, integer *givptr, 
	integer *givcol, real *givnum, integer *info);

/* Subroutine */ int _starpu_claein_(logical *rightv, logical *noinit, integer *n, 
	complex *h__, integer *ldh, complex *w, complex *v, complex *b, 
	integer *ldb, real *rwork, real *eps3, real *smlnum, integer *info);

/* Subroutine */ int _starpu_claesy_(complex *a, complex *b, complex *c__, complex *
	rt1, complex *rt2, complex *evscal, complex *cs1, complex *sn1);

/* Subroutine */ int _starpu_claev2_(complex *a, complex *b, complex *c__, real *rt1, 
	real *rt2, real *cs1, complex *sn1);

/* Subroutine */ int _starpu_clag2z_(integer *m, integer *n, complex *sa, integer *
	ldsa, doublecomplex *a, integer *lda, integer *info);

/* Subroutine */ int _starpu_clags2_(logical *upper, real *a1, complex *a2, real *a3, 
	real *b1, complex *b2, real *b3, real *csu, complex *snu, real *csv, 
	complex *snv, real *csq, complex *snq);

/* Subroutine */ int _starpu_clagtm_(char *trans, integer *n, integer *nrhs, real *
	alpha, complex *dl, complex *d__, complex *du, complex *x, integer *
	ldx, real *beta, complex *b, integer *ldb);

/* Subroutine */ int _starpu_clahef_(char *uplo, integer *n, integer *nb, integer *kb, 
	 complex *a, integer *lda, integer *ipiv, complex *w, integer *ldw, 
	integer *info);

/* Subroutine */ int _starpu_clahqr_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, complex *h__, integer *ldh, complex *w, 
	integer *iloz, integer *ihiz, complex *z__, integer *ldz, integer *
	info);

/* Subroutine */ int _starpu_clahr2_(integer *n, integer *k, integer *nb, complex *a, 
	integer *lda, complex *tau, complex *t, integer *ldt, complex *y, 
	integer *ldy);

/* Subroutine */ int _starpu_clahrd_(integer *n, integer *k, integer *nb, complex *a, 
	integer *lda, complex *tau, complex *t, integer *ldt, complex *y, 
	integer *ldy);

/* Subroutine */ int _starpu_claic1_(integer *job, integer *j, complex *x, real *sest, 
	 complex *w, complex *gamma, real *sestpr, complex *s, complex *c__);

/* Subroutine */ int _starpu_clals0_(integer *icompq, integer *nl, integer *nr, 
	integer *sqre, integer *nrhs, complex *b, integer *ldb, complex *bx, 
	integer *ldbx, integer *perm, integer *givptr, integer *givcol, 
	integer *ldgcol, real *givnum, integer *ldgnum, real *poles, real *
	difl, real *difr, real *z__, integer *k, real *c__, real *s, real *
	rwork, integer *info);

/* Subroutine */ int _starpu_clalsa_(integer *icompq, integer *smlsiz, integer *n, 
	integer *nrhs, complex *b, integer *ldb, complex *bx, integer *ldbx, 
	real *u, integer *ldu, real *vt, integer *k, real *difl, real *difr, 
	real *z__, real *poles, integer *givptr, integer *givcol, integer *
	ldgcol, integer *perm, real *givnum, real *c__, real *s, real *rwork, 
	integer *iwork, integer *info);

/* Subroutine */ int _starpu_clalsd_(char *uplo, integer *smlsiz, integer *n, integer 
	*nrhs, real *d__, real *e, complex *b, integer *ldb, real *rcond, 
	integer *rank, complex *work, real *rwork, integer *iwork, integer *
	info);

doublereal _starpu_clangb_(char *norm, integer *n, integer *kl, integer *ku, complex *
	ab, integer *ldab, real *work);

doublereal _starpu_clange_(char *norm, integer *m, integer *n, complex *a, integer *
	lda, real *work);

doublereal _starpu_clangt_(char *norm, integer *n, complex *dl, complex *d__, complex 
	*du);

doublereal _starpu_clanhb_(char *norm, char *uplo, integer *n, integer *k, complex *
	ab, integer *ldab, real *work);

doublereal _starpu_clanhe_(char *norm, char *uplo, integer *n, complex *a, integer *
	lda, real *work);

doublereal _starpu_clanhf_(char *norm, char *transr, char *uplo, integer *n, complex *
	a, real *work);

doublereal _starpu_clanhp_(char *norm, char *uplo, integer *n, complex *ap, real *
	work);

doublereal _starpu_clanhs_(char *norm, integer *n, complex *a, integer *lda, real *
	work);

doublereal _starpu_clanht_(char *norm, integer *n, real *d__, complex *e);

doublereal _starpu_clansb_(char *norm, char *uplo, integer *n, integer *k, complex *
	ab, integer *ldab, real *work);

doublereal _starpu_clansp_(char *norm, char *uplo, integer *n, complex *ap, real *
	work);

doublereal _starpu_clansy_(char *norm, char *uplo, integer *n, complex *a, integer *
	lda, real *work);

doublereal _starpu_clantb_(char *norm, char *uplo, char *diag, integer *n, integer *k, 
	 complex *ab, integer *ldab, real *work);

doublereal _starpu_clantp_(char *norm, char *uplo, char *diag, integer *n, complex *
	ap, real *work);

doublereal _starpu_clantr_(char *norm, char *uplo, char *diag, integer *m, integer *n, 
	 complex *a, integer *lda, real *work);

/* Subroutine */ int _starpu_clapll_(integer *n, complex *x, integer *incx, complex *
	y, integer *incy, real *ssmin);

/* Subroutine */ int _starpu_clapmt_(logical *forwrd, integer *m, integer *n, complex 
	*x, integer *ldx, integer *k);

/* Subroutine */ int _starpu_claqgb_(integer *m, integer *n, integer *kl, integer *ku, 
	 complex *ab, integer *ldab, real *r__, real *c__, real *rowcnd, real 
	*colcnd, real *amax, char *equed);

/* Subroutine */ int _starpu_claqge_(integer *m, integer *n, complex *a, integer *lda, 
	 real *r__, real *c__, real *rowcnd, real *colcnd, real *amax, char *
	equed);

/* Subroutine */ int _starpu_claqhb_(char *uplo, integer *n, integer *kd, complex *ab, 
	 integer *ldab, real *s, real *scond, real *amax, char *equed);

/* Subroutine */ int _starpu_claqhe_(char *uplo, integer *n, complex *a, integer *lda, 
	 real *s, real *scond, real *amax, char *equed);

/* Subroutine */ int _starpu_claqhp_(char *uplo, integer *n, complex *ap, real *s, 
	real *scond, real *amax, char *equed);

/* Subroutine */ int _starpu_claqp2_(integer *m, integer *n, integer *offset, complex 
	*a, integer *lda, integer *jpvt, complex *tau, real *vn1, real *vn2, 
	complex *work);

/* Subroutine */ int _starpu_claqps_(integer *m, integer *n, integer *offset, integer 
	*nb, integer *kb, complex *a, integer *lda, integer *jpvt, complex *
	tau, real *vn1, real *vn2, complex *auxv, complex *f, integer *ldf);

/* Subroutine */ int _starpu_claqr0_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, complex *h__, integer *ldh, complex *w, 
	integer *iloz, integer *ihiz, complex *z__, integer *ldz, complex *
	work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_claqr1_(integer *n, complex *h__, integer *ldh, complex *
	s1, complex *s2, complex *v);

/* Subroutine */ int _starpu_claqr2_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, complex *h__, integer *ldh, 
	 integer *iloz, integer *ihiz, complex *z__, integer *ldz, integer *
	ns, integer *nd, complex *sh, complex *v, integer *ldv, integer *nh, 
	complex *t, integer *ldt, integer *nv, complex *wv, integer *ldwv, 
	complex *work, integer *lwork);

/* Subroutine */ int _starpu_claqr3_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, complex *h__, integer *ldh, 
	 integer *iloz, integer *ihiz, complex *z__, integer *ldz, integer *
	ns, integer *nd, complex *sh, complex *v, integer *ldv, integer *nh, 
	complex *t, integer *ldt, integer *nv, complex *wv, integer *ldwv, 
	complex *work, integer *lwork);

/* Subroutine */ int _starpu_claqr4_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, complex *h__, integer *ldh, complex *w, 
	integer *iloz, integer *ihiz, complex *z__, integer *ldz, complex *
	work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_claqr5_(logical *wantt, logical *wantz, integer *kacc22, 
	integer *n, integer *ktop, integer *kbot, integer *nshfts, complex *s, 
	 complex *h__, integer *ldh, integer *iloz, integer *ihiz, complex *
	z__, integer *ldz, complex *v, integer *ldv, complex *u, integer *ldu, 
	 integer *nv, complex *wv, integer *ldwv, integer *nh, complex *wh, 
	integer *ldwh);

/* Subroutine */ int _starpu_claqsb_(char *uplo, integer *n, integer *kd, complex *ab, 
	 integer *ldab, real *s, real *scond, real *amax, char *equed);

/* Subroutine */ int _starpu_claqsp_(char *uplo, integer *n, complex *ap, real *s, 
	real *scond, real *amax, char *equed);

/* Subroutine */ int _starpu_claqsy_(char *uplo, integer *n, complex *a, integer *lda, 
	 real *s, real *scond, real *amax, char *equed);

/* Subroutine */ int _starpu_clar1v_(integer *n, integer *b1, integer *bn, real *
	lambda, real *d__, real *l, real *ld, real *lld, real *pivmin, real *
	gaptol, complex *z__, logical *wantnc, integer *negcnt, real *ztz, 
	real *mingma, integer *r__, integer *isuppz, real *nrminv, real *
	resid, real *rqcorr, real *work);

/* Subroutine */ int _starpu_clar2v_(integer *n, complex *x, complex *y, complex *z__, 
	 integer *incx, real *c__, complex *s, integer *incc);

/* Subroutine */ int _starpu_clarcm_(integer *m, integer *n, real *a, integer *lda, 
	complex *b, integer *ldb, complex *c__, integer *ldc, real *rwork);

/* Subroutine */ int _starpu_clarf_(char *side, integer *m, integer *n, complex *v, 
	integer *incv, complex *tau, complex *c__, integer *ldc, complex *
	work);

/* Subroutine */ int _starpu_clarfb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, complex *v, integer *ldv, 
	complex *t, integer *ldt, complex *c__, integer *ldc, complex *work, 
	integer *ldwork);

/* Subroutine */ int _starpu_clarfg_(integer *n, complex *alpha, complex *x, integer *
	incx, complex *tau);

/* Subroutine */ int _starpu_clarfp_(integer *n, complex *alpha, complex *x, integer *
	incx, complex *tau);

/* Subroutine */ int _starpu_clarft_(char *direct, char *storev, integer *n, integer *
	k, complex *v, integer *ldv, complex *tau, complex *t, integer *ldt);

/* Subroutine */ int _starpu_clarfx_(char *side, integer *m, integer *n, complex *v, 
	complex *tau, complex *c__, integer *ldc, complex *work);

/* Subroutine */ int _starpu_clargv_(integer *n, complex *x, integer *incx, complex *
	y, integer *incy, real *c__, integer *incc);

/* Subroutine */ int _starpu_clarnv_(integer *idist, integer *iseed, integer *n, 
	complex *x);

/* Subroutine */ int _starpu_clarrv_(integer *n, real *vl, real *vu, real *d__, real *
	l, real *pivmin, integer *isplit, integer *m, integer *dol, integer *
	dou, real *minrgp, real *rtol1, real *rtol2, real *w, real *werr, 
	real *wgap, integer *iblock, integer *indexw, real *gers, complex *
	z__, integer *ldz, integer *isuppz, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_clarscl2_(integer *m, integer *n, real *d__, complex *x, 
	integer *ldx);

/* Subroutine */ int _starpu_clartg_(complex *f, complex *g, real *cs, complex *sn, 
	complex *r__);

/* Subroutine */ int _starpu_clartv_(integer *n, complex *x, integer *incx, complex *
	y, integer *incy, real *c__, complex *s, integer *incc);

/* Subroutine */ int _starpu_clarz_(char *side, integer *m, integer *n, integer *l, 
	complex *v, integer *incv, complex *tau, complex *c__, integer *ldc, 
	complex *work);

/* Subroutine */ int _starpu_clarzb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, integer *l, complex *v, 
	integer *ldv, complex *t, integer *ldt, complex *c__, integer *ldc, 
	complex *work, integer *ldwork);

/* Subroutine */ int _starpu_clarzt_(char *direct, char *storev, integer *n, integer *
	k, complex *v, integer *ldv, complex *tau, complex *t, integer *ldt);

/* Subroutine */ int _starpu_clascl_(char *type__, integer *kl, integer *ku, real *
	cfrom, real *cto, integer *m, integer *n, complex *a, integer *lda, 
	integer *info);

/* Subroutine */ int _starpu_clascl2_(integer *m, integer *n, real *d__, complex *x, 
	integer *ldx);

/* Subroutine */ int _starpu_claset_(char *uplo, integer *m, integer *n, complex *
	alpha, complex *beta, complex *a, integer *lda);

/* Subroutine */ int _starpu_clasr_(char *side, char *pivot, char *direct, integer *m, 
	 integer *n, real *c__, real *s, complex *a, integer *lda);

/* Subroutine */ int _starpu_classq_(integer *n, complex *x, integer *incx, real *
	scale, real *sumsq);

/* Subroutine */ int _starpu_claswp_(integer *n, complex *a, integer *lda, integer *
	k1, integer *k2, integer *ipiv, integer *incx);

/* Subroutine */ int _starpu_clasyf_(char *uplo, integer *n, integer *nb, integer *kb, 
	 complex *a, integer *lda, integer *ipiv, complex *w, integer *ldw, 
	integer *info);

/* Subroutine */ int _starpu_clatbs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, integer *kd, complex *ab, integer *ldab, complex *
	x, real *scale, real *cnorm, integer *info);

/* Subroutine */ int _starpu_clatdf_(integer *ijob, integer *n, complex *z__, integer 
	*ldz, complex *rhs, real *rdsum, real *rdscal, integer *ipiv, integer 
	*jpiv);

/* Subroutine */ int _starpu_clatps_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, complex *ap, complex *x, real *scale, real *cnorm, 
	 integer *info);

/* Subroutine */ int _starpu_clatrd_(char *uplo, integer *n, integer *nb, complex *a, 
	integer *lda, real *e, complex *tau, complex *w, integer *ldw);

/* Subroutine */ int _starpu_clatrs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, complex *a, integer *lda, complex *x, real *scale, 
	 real *cnorm, integer *info);

/* Subroutine */ int _starpu_clatrz_(integer *m, integer *n, integer *l, complex *a, 
	integer *lda, complex *tau, complex *work);

/* Subroutine */ int _starpu_clatzm_(char *side, integer *m, integer *n, complex *v, 
	integer *incv, complex *tau, complex *c1, complex *c2, integer *ldc, 
	complex *work);

/* Subroutine */ int _starpu_clauu2_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *info);

/* Subroutine */ int _starpu_clauum_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *info);

/* Subroutine */ int _starpu_cpbcon_(char *uplo, integer *n, integer *kd, complex *ab, 
	 integer *ldab, real *anorm, real *rcond, complex *work, real *rwork, 
	integer *info);

/* Subroutine */ int _starpu_cpbequ_(char *uplo, integer *n, integer *kd, complex *ab, 
	 integer *ldab, real *s, real *scond, real *amax, integer *info);

/* Subroutine */ int _starpu_cpbrfs_(char *uplo, integer *n, integer *kd, integer *
	nrhs, complex *ab, integer *ldab, complex *afb, integer *ldafb, 
	complex *b, integer *ldb, complex *x, integer *ldx, real *ferr, real *
	berr, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cpbstf_(char *uplo, integer *n, integer *kd, complex *ab, 
	 integer *ldab, integer *info);

/* Subroutine */ int _starpu_cpbsv_(char *uplo, integer *n, integer *kd, integer *
	nrhs, complex *ab, integer *ldab, complex *b, integer *ldb, integer *
	info);

/* Subroutine */ int _starpu_cpbsvx_(char *fact, char *uplo, integer *n, integer *kd, 
	integer *nrhs, complex *ab, integer *ldab, complex *afb, integer *
	ldafb, char *equed, real *s, complex *b, integer *ldb, complex *x, 
	integer *ldx, real *rcond, real *ferr, real *berr, complex *work, 
	real *rwork, integer *info);

/* Subroutine */ int _starpu_cpbtf2_(char *uplo, integer *n, integer *kd, complex *ab, 
	 integer *ldab, integer *info);

/* Subroutine */ int _starpu_cpbtrf_(char *uplo, integer *n, integer *kd, complex *ab, 
	 integer *ldab, integer *info);

/* Subroutine */ int _starpu_cpbtrs_(char *uplo, integer *n, integer *kd, integer *
	nrhs, complex *ab, integer *ldab, complex *b, integer *ldb, integer *
	info);

/* Subroutine */ int _starpu_cpftrf_(char *transr, char *uplo, integer *n, complex *a, 
	 integer *info);

/* Subroutine */ int _starpu_cpftri_(char *transr, char *uplo, integer *n, complex *a, 
	 integer *info);

/* Subroutine */ int _starpu_cpftrs_(char *transr, char *uplo, integer *n, integer *
	nrhs, complex *a, complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_cpocon_(char *uplo, integer *n, complex *a, integer *lda, 
	 real *anorm, real *rcond, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cpoequ_(integer *n, complex *a, integer *lda, real *s, 
	real *scond, real *amax, integer *info);

/* Subroutine */ int _starpu_cpoequb_(integer *n, complex *a, integer *lda, real *s, 
	real *scond, real *amax, integer *info);

/* Subroutine */ int _starpu_cporfs_(char *uplo, integer *n, integer *nrhs, complex *
	a, integer *lda, complex *af, integer *ldaf, complex *b, integer *ldb, 
	 complex *x, integer *ldx, real *ferr, real *berr, complex *work, 
	real *rwork, integer *info);

/* Subroutine */ int _starpu_cporfsx_(char *uplo, char *equed, integer *n, integer *
	nrhs, complex *a, integer *lda, complex *af, integer *ldaf, real *s, 
	complex *b, integer *ldb, complex *x, integer *ldx, real *rcond, real 
	*berr, integer *n_err_bnds__, real *err_bnds_norm__, real *
	err_bnds_comp__, integer *nparams, real *params, complex *work, real *
	rwork, integer *info);

/* Subroutine */ int _starpu_cposv_(char *uplo, integer *n, integer *nrhs, complex *a, 
	 integer *lda, complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_cposvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, complex *a, integer *lda, complex *af, integer *ldaf, char *
	equed, real *s, complex *b, integer *ldb, complex *x, integer *ldx, 
	real *rcond, real *ferr, real *berr, complex *work, real *rwork, 
	integer *info);

/* Subroutine */ int _starpu_cposvxx_(char *fact, char *uplo, integer *n, integer *
	nrhs, complex *a, integer *lda, complex *af, integer *ldaf, char *
	equed, real *s, complex *b, integer *ldb, complex *x, integer *ldx, 
	real *rcond, real *rpvgrw, real *berr, integer *n_err_bnds__, real *
	err_bnds_norm__, real *err_bnds_comp__, integer *nparams, real *
	params, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cpotf2_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *info);

/* Subroutine */ int _starpu_cpotrf_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *info);

/* Subroutine */ int _starpu_cpotri_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *info);

/* Subroutine */ int _starpu_cpotrs_(char *uplo, integer *n, integer *nrhs, complex *
	a, integer *lda, complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_cppcon_(char *uplo, integer *n, complex *ap, real *anorm, 
	 real *rcond, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cppequ_(char *uplo, integer *n, complex *ap, real *s, 
	real *scond, real *amax, integer *info);

/* Subroutine */ int _starpu_cpprfs_(char *uplo, integer *n, integer *nrhs, complex *
	ap, complex *afp, complex *b, integer *ldb, complex *x, integer *ldx, 
	real *ferr, real *berr, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cppsv_(char *uplo, integer *n, integer *nrhs, complex *
	ap, complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_cppsvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, complex *ap, complex *afp, char *equed, real *s, complex *b, 
	integer *ldb, complex *x, integer *ldx, real *rcond, real *ferr, real 
	*berr, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_cpptrf_(char *uplo, integer *n, complex *ap, integer *
	info);

/* Subroutine */ int _starpu_cpptri_(char *uplo, integer *n, complex *ap, integer *
	info);

/* Subroutine */ int _starpu_cpptrs_(char *uplo, integer *n, integer *nrhs, complex *
	ap, complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_cpstf2_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *piv, integer *rank, real *tol, real *work, integer *info);

/* Subroutine */ int _starpu_cpstrf_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *piv, integer *rank, real *tol, real *work, integer *info);

/* Subroutine */ int _starpu_cptcon_(integer *n, real *d__, complex *e, real *anorm, 
	real *rcond, real *rwork, integer *info);

/* Subroutine */ int _starpu_cpteqr_(char *compz, integer *n, real *d__, real *e, 
	complex *z__, integer *ldz, real *work, integer *info);

/* Subroutine */ int _starpu_cptrfs_(char *uplo, integer *n, integer *nrhs, real *d__, 
	 complex *e, real *df, complex *ef, complex *b, integer *ldb, complex 
	*x, integer *ldx, real *ferr, real *berr, complex *work, real *rwork, 
	integer *info);

/* Subroutine */ int _starpu_cptsv_(integer *n, integer *nrhs, real *d__, complex *e, 
	complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_cptsvx_(char *fact, integer *n, integer *nrhs, real *d__, 
	 complex *e, real *df, complex *ef, complex *b, integer *ldb, complex 
	*x, integer *ldx, real *rcond, real *ferr, real *berr, complex *work, 
	real *rwork, integer *info);

/* Subroutine */ int _starpu_cpttrf_(integer *n, real *d__, complex *e, integer *info);

/* Subroutine */ int _starpu_cpttrs_(char *uplo, integer *n, integer *nrhs, real *d__, 
	 complex *e, complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_cptts2_(integer *iuplo, integer *n, integer *nrhs, real *
	d__, complex *e, complex *b, integer *ldb);

/* Subroutine */ int _starpu_crot_(integer *n, complex *cx, integer *incx, complex *
	cy, integer *incy, real *c__, complex *s);

/* Subroutine */ int _starpu_cspcon_(char *uplo, integer *n, complex *ap, integer *
	ipiv, real *anorm, real *rcond, complex *work, integer *info);

/* Subroutine */ int _starpu_cspmv_(char *uplo, integer *n, complex *alpha, complex *
	ap, complex *x, integer *incx, complex *beta, complex *y, integer *
	incy);

/* Subroutine */ int _starpu_cspr_(char *uplo, integer *n, complex *alpha, complex *x, 
	 integer *incx, complex *ap);

/* Subroutine */ int _starpu_csprfs_(char *uplo, integer *n, integer *nrhs, complex *
	ap, complex *afp, integer *ipiv, complex *b, integer *ldb, complex *x, 
	 integer *ldx, real *ferr, real *berr, complex *work, real *rwork, 
	integer *info);

/* Subroutine */ int _starpu_cspsv_(char *uplo, integer *n, integer *nrhs, complex *
	ap, integer *ipiv, complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_cspsvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, complex *ap, complex *afp, integer *ipiv, complex *b, integer *
	ldb, complex *x, integer *ldx, real *rcond, real *ferr, real *berr, 
	complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_csptrf_(char *uplo, integer *n, complex *ap, integer *
	ipiv, integer *info);

/* Subroutine */ int _starpu_csptri_(char *uplo, integer *n, complex *ap, integer *
	ipiv, complex *work, integer *info);

/* Subroutine */ int _starpu_csptrs_(char *uplo, integer *n, integer *nrhs, complex *
	ap, integer *ipiv, complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu__starpu_csrscl_(integer *n, real *sa, complex *sx, integer *incx);

/* Subroutine */ int _starpu_cstedc_(char *compz, integer *n, real *d__, real *e, 
	complex *z__, integer *ldz, complex *work, integer *lwork, real *
	rwork, integer *lrwork, integer *iwork, integer *liwork, integer *
	info);

/* Subroutine */ int _starpu_cstegr_(char *jobz, char *range, integer *n, real *d__, 
	real *e, real *vl, real *vu, integer *il, integer *iu, real *abstol, 
	integer *m, real *w, complex *z__, integer *ldz, integer *isuppz, 
	real *work, integer *lwork, integer *iwork, integer *liwork, integer *
	info);

/* Subroutine */ int _starpu_cstein_(integer *n, real *d__, real *e, integer *m, real 
	*w, integer *iblock, integer *isplit, complex *z__, integer *ldz, 
	real *work, integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu_cstemr_(char *jobz, char *range, integer *n, real *d__, 
	real *e, real *vl, real *vu, integer *il, integer *iu, integer *m, 
	real *w, complex *z__, integer *ldz, integer *nzc, integer *isuppz, 
	logical *tryrac, real *work, integer *lwork, integer *iwork, integer *
	liwork, integer *info);

/* Subroutine */ int _starpu_csteqr_(char *compz, integer *n, real *d__, real *e, 
	complex *z__, integer *ldz, real *work, integer *info);

/* Subroutine */ int _starpu_csycon_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *ipiv, real *anorm, real *rcond, complex *work, integer *
	info);

/* Subroutine */ int _starpu_csyequb_(char *uplo, integer *n, complex *a, integer *
	lda, real *s, real *scond, real *amax, complex *work, integer *info);

/* Subroutine */ int _starpu_csymv_(char *uplo, integer *n, complex *alpha, complex *
	a, integer *lda, complex *x, integer *incx, complex *beta, complex *y, 
	 integer *incy);

/* Subroutine */ int _starpu_csyr_(char *uplo, integer *n, complex *alpha, complex *x, 
	 integer *incx, complex *a, integer *lda);

/* Subroutine */ int _starpu_csyrfs_(char *uplo, integer *n, integer *nrhs, complex *
	a, integer *lda, complex *af, integer *ldaf, integer *ipiv, complex *
	b, integer *ldb, complex *x, integer *ldx, real *ferr, real *berr, 
	complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_csyrfsx_(char *uplo, char *equed, integer *n, integer *
	nrhs, complex *a, integer *lda, complex *af, integer *ldaf, integer *
	ipiv, real *s, complex *b, integer *ldb, complex *x, integer *ldx, 
	real *rcond, real *berr, integer *n_err_bnds__, real *err_bnds_norm__, 
	 real *err_bnds_comp__, integer *nparams, real *params, complex *work, 
	 real *rwork, integer *info);

/* Subroutine */ int _starpu_csysv_(char *uplo, integer *n, integer *nrhs, complex *a, 
	 integer *lda, integer *ipiv, complex *b, integer *ldb, complex *work, 
	 integer *lwork, integer *info);

/* Subroutine */ int _starpu_csysvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, complex *a, integer *lda, complex *af, integer *ldaf, integer *
	ipiv, complex *b, integer *ldb, complex *x, integer *ldx, real *rcond, 
	 real *ferr, real *berr, complex *work, integer *lwork, real *rwork, 
	integer *info);

/* Subroutine */ int _starpu_csysvxx_(char *fact, char *uplo, integer *n, integer *
	nrhs, complex *a, integer *lda, complex *af, integer *ldaf, integer *
	ipiv, char *equed, real *s, complex *b, integer *ldb, complex *x, 
	integer *ldx, real *rcond, real *rpvgrw, real *berr, integer *
	n_err_bnds__, real *err_bnds_norm__, real *err_bnds_comp__, integer *
	nparams, real *params, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_csytf2_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *ipiv, integer *info);

/* Subroutine */ int _starpu_csytrf_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *ipiv, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_csytri_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *ipiv, complex *work, integer *info);

/* Subroutine */ int _starpu_csytrs_(char *uplo, integer *n, integer *nrhs, complex *
	a, integer *lda, integer *ipiv, complex *b, integer *ldb, integer *
	info);

/* Subroutine */ int _starpu_ctbcon_(char *norm, char *uplo, char *diag, integer *n, 
	integer *kd, complex *ab, integer *ldab, real *rcond, complex *work, 
	real *rwork, integer *info);

/* Subroutine */ int _starpu_ctbrfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *kd, integer *nrhs, complex *ab, integer *ldab, complex *b, 
	integer *ldb, complex *x, integer *ldx, real *ferr, real *berr, 
	complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_ctbtrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *kd, integer *nrhs, complex *ab, integer *ldab, complex *b, 
	integer *ldb, integer *info);

/* Subroutine */ int _starpu_ctfsm_(char *transr, char *side, char *uplo, char *trans, 
	 char *diag, integer *m, integer *n, complex *alpha, complex *a, 
	complex *b, integer *ldb);

/* Subroutine */ int _starpu_ctftri_(char *transr, char *uplo, char *diag, integer *n, 
	 complex *a, integer *info);

/* Subroutine */ int _starpu_ctfttp_(char *transr, char *uplo, integer *n, complex *
	arf, complex *ap, integer *info);

/* Subroutine */ int _starpu_ctfttr_(char *transr, char *uplo, integer *n, complex *
	arf, complex *a, integer *lda, integer *info);

/* Subroutine */ int _starpu_ctgevc_(char *side, char *howmny, logical *select, 
	integer *n, complex *s, integer *lds, complex *p, integer *ldp, 
	complex *vl, integer *ldvl, complex *vr, integer *ldvr, integer *mm, 
	integer *m, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_ctgex2_(logical *wantq, logical *wantz, integer *n, 
	complex *a, integer *lda, complex *b, integer *ldb, complex *q, 
	integer *ldq, complex *z__, integer *ldz, integer *j1, integer *info);

/* Subroutine */ int _starpu_ctgexc_(logical *wantq, logical *wantz, integer *n, 
	complex *a, integer *lda, complex *b, integer *ldb, complex *q, 
	integer *ldq, complex *z__, integer *ldz, integer *ifst, integer *
	ilst, integer *info);

/* Subroutine */ int _starpu_ctgsen_(integer *ijob, logical *wantq, logical *wantz, 
	logical *select, integer *n, complex *a, integer *lda, complex *b, 
	integer *ldb, complex *alpha, complex *beta, complex *q, integer *ldq, 
	 complex *z__, integer *ldz, integer *m, real *pl, real *pr, real *
	dif, complex *work, integer *lwork, integer *iwork, integer *liwork, 
	integer *info);

/* Subroutine */ int _starpu_ctgsja_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *p, integer *n, integer *k, integer *l, complex *a, integer *
	lda, complex *b, integer *ldb, real *tola, real *tolb, real *alpha, 
	real *beta, complex *u, integer *ldu, complex *v, integer *ldv, 
	complex *q, integer *ldq, complex *work, integer *ncycle, integer *
	info);

/* Subroutine */ int _starpu_ctgsna_(char *job, char *howmny, logical *select, 
	integer *n, complex *a, integer *lda, complex *b, integer *ldb, 
	complex *vl, integer *ldvl, complex *vr, integer *ldvr, real *s, real 
	*dif, integer *mm, integer *m, complex *work, integer *lwork, integer 
	*iwork, integer *info);

/* Subroutine */ int _starpu_ctgsy2_(char *trans, integer *ijob, integer *m, integer *
	n, complex *a, integer *lda, complex *b, integer *ldb, complex *c__, 
	integer *ldc, complex *d__, integer *ldd, complex *e, integer *lde, 
	complex *f, integer *ldf, real *scale, real *rdsum, real *rdscal, 
	integer *info);

/* Subroutine */ int _starpu_ctgsyl_(char *trans, integer *ijob, integer *m, integer *
	n, complex *a, integer *lda, complex *b, integer *ldb, complex *c__, 
	integer *ldc, complex *d__, integer *ldd, complex *e, integer *lde, 
	complex *f, integer *ldf, real *scale, real *dif, complex *work, 
	integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_ctpcon_(char *norm, char *uplo, char *diag, integer *n, 
	complex *ap, real *rcond, complex *work, real *rwork, integer *info);

/* Subroutine */ int _starpu_ctprfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, complex *ap, complex *b, integer *ldb, complex *x, 
	integer *ldx, real *ferr, real *berr, complex *work, real *rwork, 
	integer *info);

/* Subroutine */ int _starpu_ctptri_(char *uplo, char *diag, integer *n, complex *ap, 
	integer *info);

/* Subroutine */ int _starpu_ctptrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, complex *ap, complex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_ctpttf_(char *transr, char *uplo, integer *n, complex *
	ap, complex *arf, integer *info);

/* Subroutine */ int _starpu_ctpttr_(char *uplo, integer *n, complex *ap, complex *a, 
	integer *lda, integer *info);

/* Subroutine */ int _starpu_ctrcon_(char *norm, char *uplo, char *diag, integer *n, 
	complex *a, integer *lda, real *rcond, complex *work, real *rwork, 
	integer *info);

/* Subroutine */ int _starpu_ctrevc_(char *side, char *howmny, logical *select, 
	integer *n, complex *t, integer *ldt, complex *vl, integer *ldvl, 
	complex *vr, integer *ldvr, integer *mm, integer *m, complex *work, 
	real *rwork, integer *info);

/* Subroutine */ int _starpu_ctrexc_(char *compq, integer *n, complex *t, integer *
	ldt, complex *q, integer *ldq, integer *ifst, integer *ilst, integer *
	info);

/* Subroutine */ int _starpu_ctrrfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, complex *a, integer *lda, complex *b, integer *ldb, 
	complex *x, integer *ldx, real *ferr, real *berr, complex *work, real 
	*rwork, integer *info);

/* Subroutine */ int _starpu_ctrsen_(char *job, char *compq, logical *select, integer 
	*n, complex *t, integer *ldt, complex *q, integer *ldq, complex *w, 
	integer *m, real *s, real *sep, complex *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_ctrsna_(char *job, char *howmny, logical *select, 
	integer *n, complex *t, integer *ldt, complex *vl, integer *ldvl, 
	complex *vr, integer *ldvr, real *s, real *sep, integer *mm, integer *
	m, complex *work, integer *ldwork, real *rwork, integer *info);

/* Subroutine */ int _starpu_ctrsyl_(char *trana, char *tranb, integer *isgn, integer 
	*m, integer *n, complex *a, integer *lda, complex *b, integer *ldb, 
	complex *c__, integer *ldc, real *scale, integer *info);

/* Subroutine */ int _starpu_ctrti2_(char *uplo, char *diag, integer *n, complex *a, 
	integer *lda, integer *info);

/* Subroutine */ int _starpu_ctrtri_(char *uplo, char *diag, integer *n, complex *a, 
	integer *lda, integer *info);

/* Subroutine */ int _starpu_ctrtrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, complex *a, integer *lda, complex *b, integer *ldb, 
	integer *info);

/* Subroutine */ int _starpu_ctrttf_(char *transr, char *uplo, integer *n, complex *a, 
	 integer *lda, complex *arf, integer *info);

/* Subroutine */ int _starpu_ctrttp_(char *uplo, integer *n, complex *a, integer *lda, 
	 complex *ap, integer *info);

/* Subroutine */ int _starpu_ctzrqf_(integer *m, integer *n, complex *a, integer *lda, 
	 complex *tau, integer *info);

/* Subroutine */ int _starpu_ctzrzf_(integer *m, integer *n, complex *a, integer *lda, 
	 complex *tau, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cung2l_(integer *m, integer *n, integer *k, complex *a, 
	integer *lda, complex *tau, complex *work, integer *info);

/* Subroutine */ int _starpu_cung2r_(integer *m, integer *n, integer *k, complex *a, 
	integer *lda, complex *tau, complex *work, integer *info);

/* Subroutine */ int _starpu_cungbr_(char *vect, integer *m, integer *n, integer *k, 
	complex *a, integer *lda, complex *tau, complex *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int _starpu_cunghr_(integer *n, integer *ilo, integer *ihi, complex *
	a, integer *lda, complex *tau, complex *work, integer *lwork, integer 
	*info);

/* Subroutine */ int _starpu_cungl2_(integer *m, integer *n, integer *k, complex *a, 
	integer *lda, complex *tau, complex *work, integer *info);

/* Subroutine */ int _starpu_cunglq_(integer *m, integer *n, integer *k, complex *a, 
	integer *lda, complex *tau, complex *work, integer *lwork, integer *
	info);

/* Subroutine */ int _starpu_cungql_(integer *m, integer *n, integer *k, complex *a, 
	integer *lda, complex *tau, complex *work, integer *lwork, integer *
	info);

/* Subroutine */ int _starpu_cungqr_(integer *m, integer *n, integer *k, complex *a, 
	integer *lda, complex *tau, complex *work, integer *lwork, integer *
	info);

/* Subroutine */ int _starpu_cungr2_(integer *m, integer *n, integer *k, complex *a, 
	integer *lda, complex *tau, complex *work, integer *info);

/* Subroutine */ int _starpu_cungrq_(integer *m, integer *n, integer *k, complex *a, 
	integer *lda, complex *tau, complex *work, integer *lwork, integer *
	info);

/* Subroutine */ int _starpu_cungtr_(char *uplo, integer *n, complex *a, integer *lda, 
	 complex *tau, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cunm2l_(char *side, char *trans, integer *m, integer *n, 
	integer *k, complex *a, integer *lda, complex *tau, complex *c__, 
	integer *ldc, complex *work, integer *info);

/* Subroutine */ int _starpu_cunm2r_(char *side, char *trans, integer *m, integer *n, 
	integer *k, complex *a, integer *lda, complex *tau, complex *c__, 
	integer *ldc, complex *work, integer *info);

/* Subroutine */ int _starpu_cunmbr_(char *vect, char *side, char *trans, integer *m, 
	integer *n, integer *k, complex *a, integer *lda, complex *tau, 
	complex *c__, integer *ldc, complex *work, integer *lwork, integer *
	info);

/* Subroutine */ int _starpu_cunmhr_(char *side, char *trans, integer *m, integer *n, 
	integer *ilo, integer *ihi, complex *a, integer *lda, complex *tau, 
	complex *c__, integer *ldc, complex *work, integer *lwork, integer *
	info);

/* Subroutine */ int _starpu_cunml2_(char *side, char *trans, integer *m, integer *n, 
	integer *k, complex *a, integer *lda, complex *tau, complex *c__, 
	integer *ldc, complex *work, integer *info);

/* Subroutine */ int _starpu_cunmlq_(char *side, char *trans, integer *m, integer *n, 
	integer *k, complex *a, integer *lda, complex *tau, complex *c__, 
	integer *ldc, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cunmql_(char *side, char *trans, integer *m, integer *n, 
	integer *k, complex *a, integer *lda, complex *tau, complex *c__, 
	integer *ldc, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cunmqr_(char *side, char *trans, integer *m, integer *n, 
	integer *k, complex *a, integer *lda, complex *tau, complex *c__, 
	integer *ldc, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cunmr2_(char *side, char *trans, integer *m, integer *n, 
	integer *k, complex *a, integer *lda, complex *tau, complex *c__, 
	integer *ldc, complex *work, integer *info);

/* Subroutine */ int _starpu_cunmr3_(char *side, char *trans, integer *m, integer *n, 
	integer *k, integer *l, complex *a, integer *lda, complex *tau, 
	complex *c__, integer *ldc, complex *work, integer *info);

/* Subroutine */ int _starpu_cunmrq_(char *side, char *trans, integer *m, integer *n, 
	integer *k, complex *a, integer *lda, complex *tau, complex *c__, 
	integer *ldc, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cunmrz_(char *side, char *trans, integer *m, integer *n, 
	integer *k, integer *l, complex *a, integer *lda, complex *tau, 
	complex *c__, integer *ldc, complex *work, integer *lwork, integer *
	info);

/* Subroutine */ int _starpu_cunmtr_(char *side, char *uplo, char *trans, integer *m, 
	integer *n, complex *a, integer *lda, complex *tau, complex *c__, 
	integer *ldc, complex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_cupgtr_(char *uplo, integer *n, complex *ap, complex *
	tau, complex *q, integer *ldq, complex *work, integer *info);

/* Subroutine */ int _starpu_cupmtr_(char *side, char *uplo, char *trans, integer *m, 
	integer *n, complex *ap, complex *tau, complex *c__, integer *ldc, 
	complex *work, integer *info);

/* Subroutine */ int _starpu_dbdsdc_(char *uplo, char *compq, integer *n, doublereal *
	d__, doublereal *e, doublereal *u, integer *ldu, doublereal *vt, 
	integer *ldvt, doublereal *q, integer *iq, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_dbdsqr_(char *uplo, integer *n, integer *ncvt, integer *
	nru, integer *ncc, doublereal *d__, doublereal *e, doublereal *vt, 
	integer *ldvt, doublereal *u, integer *ldu, doublereal *c__, integer *
	ldc, doublereal *work, integer *info);

/* Subroutine */ int _starpu_ddisna_(char *job, integer *m, integer *n, doublereal *
	d__, doublereal *sep, integer *info);

/* Subroutine */ int _starpu_dgbbrd_(char *vect, integer *m, integer *n, integer *ncc, 
	 integer *kl, integer *ku, doublereal *ab, integer *ldab, doublereal *
	d__, doublereal *e, doublereal *q, integer *ldq, doublereal *pt, 
	integer *ldpt, doublereal *c__, integer *ldc, doublereal *work, 
	integer *info);

/* Subroutine */ int _starpu_dgbcon_(char *norm, integer *n, integer *kl, integer *ku, 
	 doublereal *ab, integer *ldab, integer *ipiv, doublereal *anorm, 
	doublereal *rcond, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dgbequ_(integer *m, integer *n, integer *kl, integer *ku, 
	 doublereal *ab, integer *ldab, doublereal *r__, doublereal *c__, 
	doublereal *rowcnd, doublereal *colcnd, doublereal *amax, integer *
	info);

/* Subroutine */ int _starpu_dgbequb_(integer *m, integer *n, integer *kl, integer *
	ku, doublereal *ab, integer *ldab, doublereal *r__, doublereal *c__, 
	doublereal *rowcnd, doublereal *colcnd, doublereal *amax, integer *
	info);

/* Subroutine */ int _starpu_dgbrfs_(char *trans, integer *n, integer *kl, integer *
	ku, integer *nrhs, doublereal *ab, integer *ldab, doublereal *afb, 
	integer *ldafb, integer *ipiv, doublereal *b, integer *ldb, 
	doublereal *x, integer *ldx, doublereal *ferr, doublereal *berr, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dgbrfsx_(char *trans, char *equed, integer *n, integer *
	kl, integer *ku, integer *nrhs, doublereal *ab, integer *ldab, 
	doublereal *afb, integer *ldafb, integer *ipiv, doublereal *r__, 
	doublereal *c__, doublereal *b, integer *ldb, doublereal *x, integer *
	ldx, doublereal *rcond, doublereal *berr, integer *n_err_bnds__, 
	doublereal *err_bnds_norm__, doublereal *err_bnds_comp__, integer *
	nparams, doublereal *params, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_dgbsv_(integer *n, integer *kl, integer *ku, integer *
	nrhs, doublereal *ab, integer *ldab, integer *ipiv, doublereal *b, 
	integer *ldb, integer *info);

/* Subroutine */ int _starpu_dgbsvx_(char *fact, char *trans, integer *n, integer *kl, 
	 integer *ku, integer *nrhs, doublereal *ab, integer *ldab, 
	doublereal *afb, integer *ldafb, integer *ipiv, char *equed, 
	doublereal *r__, doublereal *c__, doublereal *b, integer *ldb, 
	doublereal *x, integer *ldx, doublereal *rcond, doublereal *ferr, 
	doublereal *berr, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dgbsvxx_(char *fact, char *trans, integer *n, integer *
	kl, integer *ku, integer *nrhs, doublereal *ab, integer *ldab, 
	doublereal *afb, integer *ldafb, integer *ipiv, char *equed, 
	doublereal *r__, doublereal *c__, doublereal *b, integer *ldb, 
	doublereal *x, integer *ldx, doublereal *rcond, doublereal *rpvgrw, 
	doublereal *berr, integer *n_err_bnds__, doublereal *err_bnds_norm__, 
	doublereal *err_bnds_comp__, integer *nparams, doublereal *params, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dgbtf2_(integer *m, integer *n, integer *kl, integer *ku, 
	 doublereal *ab, integer *ldab, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_dgbtrf_(integer *m, integer *n, integer *kl, integer *ku, 
	 doublereal *ab, integer *ldab, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_dgbtrs_(char *trans, integer *n, integer *kl, integer *
	ku, integer *nrhs, doublereal *ab, integer *ldab, integer *ipiv, 
	doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_dgebak_(char *job, char *side, integer *n, integer *ilo, 
	integer *ihi, doublereal *scale, integer *m, doublereal *v, integer *
	ldv, integer *info);

/* Subroutine */ int _starpu_dgebal_(char *job, integer *n, doublereal *a, integer *
	lda, integer *ilo, integer *ihi, doublereal *scale, integer *info);

/* Subroutine */ int _starpu_dgebd2_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *d__, doublereal *e, doublereal *tauq, doublereal *
	taup, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dgebrd_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *d__, doublereal *e, doublereal *tauq, doublereal *
	taup, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dgecon_(char *norm, integer *n, doublereal *a, integer *
	lda, doublereal *anorm, doublereal *rcond, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_dgeequ_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *r__, doublereal *c__, doublereal *rowcnd, doublereal 
	*colcnd, doublereal *amax, integer *info);

/* Subroutine */ int _starpu_dgeequb_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *r__, doublereal *c__, doublereal *rowcnd, doublereal 
	*colcnd, doublereal *amax, integer *info);

/* Subroutine */ int _starpu_dgees_(char *jobvs, char *sort, L_fp select, integer *n, 
	doublereal *a, integer *lda, integer *sdim, doublereal *wr, 
	doublereal *wi, doublereal *vs, integer *ldvs, doublereal *work, 
	integer *lwork, logical *bwork, integer *info);

/* Subroutine */ int _starpu_dgeesx_(char *jobvs, char *sort, L_fp select, char *
	sense, integer *n, doublereal *a, integer *lda, integer *sdim, 
	doublereal *wr, doublereal *wi, doublereal *vs, integer *ldvs, 
	doublereal *rconde, doublereal *rcondv, doublereal *work, integer *
	lwork, integer *iwork, integer *liwork, logical *bwork, integer *info);

/* Subroutine */ int _starpu_dgeev_(char *jobvl, char *jobvr, integer *n, doublereal *
	a, integer *lda, doublereal *wr, doublereal *wi, doublereal *vl, 
	integer *ldvl, doublereal *vr, integer *ldvr, doublereal *work, 
	integer *lwork, integer *info);

/* Subroutine */ int _starpu_dgeevx_(char *balanc, char *jobvl, char *jobvr, char *
	sense, integer *n, doublereal *a, integer *lda, doublereal *wr, 
	doublereal *wi, doublereal *vl, integer *ldvl, doublereal *vr, 
	integer *ldvr, integer *ilo, integer *ihi, doublereal *scale, 
	doublereal *abnrm, doublereal *rconde, doublereal *rcondv, doublereal 
	*work, integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dgegs_(char *jobvsl, char *jobvsr, integer *n, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *
	alphar, doublereal *alphai, doublereal *beta, doublereal *vsl, 
	integer *ldvsl, doublereal *vsr, integer *ldvsr, doublereal *work, 
	integer *lwork, integer *info);

/* Subroutine */ int _starpu_dgegv_(char *jobvl, char *jobvr, integer *n, doublereal *
	a, integer *lda, doublereal *b, integer *ldb, doublereal *alphar, 
	doublereal *alphai, doublereal *beta, doublereal *vl, integer *ldvl, 
	doublereal *vr, integer *ldvr, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_dgehd2_(integer *n, integer *ilo, integer *ihi, 
	doublereal *a, integer *lda, doublereal *tau, doublereal *work, 
	integer *info);

/* Subroutine */ int _starpu_dgehrd_(integer *n, integer *ilo, integer *ihi, 
	doublereal *a, integer *lda, doublereal *tau, doublereal *work, 
	integer *lwork, integer *info);

/* Subroutine */ int _starpu_dgejsv_(char *joba, char *jobu, char *jobv, char *jobr, 
	char *jobt, char *jobp, integer *m, integer *n, doublereal *a, 
	integer *lda, doublereal *sva, doublereal *u, integer *ldu, 
	doublereal *v, integer *ldv, doublereal *work, integer *lwork, 
	integer *iwork, integer *info);

/* Subroutine */ int _starpu_dgelq2_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dgelqf_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dgels_(char *trans, integer *m, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *b, integer *ldb, 
	doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dgelsd_(integer *m, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *
	s, doublereal *rcond, integer *rank, doublereal *work, integer *lwork, 
	 integer *iwork, integer *info);

/* Subroutine */ int _starpu_dgelss_(integer *m, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *
	s, doublereal *rcond, integer *rank, doublereal *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int _starpu_dgelsx_(integer *m, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, integer *
	jpvt, doublereal *rcond, integer *rank, doublereal *work, integer *
	info);

/* Subroutine */ int _starpu_dgelsy_(integer *m, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, integer *
	jpvt, doublereal *rcond, integer *rank, doublereal *work, integer *
	lwork, integer *info);

/* Subroutine */ int _starpu_dgeql2_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dgeqlf_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dgeqp3_(integer *m, integer *n, doublereal *a, integer *
	lda, integer *jpvt, doublereal *tau, doublereal *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int _starpu_dgeqpf_(integer *m, integer *n, doublereal *a, integer *
	lda, integer *jpvt, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dgeqr2_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dgeqrf_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dgerfs_(char *trans, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *af, integer *ldaf, integer *
	ipiv, doublereal *b, integer *ldb, doublereal *x, integer *ldx, 
	doublereal *ferr, doublereal *berr, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_dgerfsx_(char *trans, char *equed, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	integer *ipiv, doublereal *r__, doublereal *c__, doublereal *b, 
	integer *ldb, doublereal *x, integer *ldx, doublereal *rcond, 
	doublereal *berr, integer *n_err_bnds__, doublereal *err_bnds_norm__, 
	doublereal *err_bnds_comp__, integer *nparams, doublereal *params, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dgerq2_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dgerqf_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dgesc2_(integer *n, doublereal *a, integer *lda, 
	doublereal *rhs, integer *ipiv, integer *jpiv, doublereal *scale);

/* Subroutine */ int _starpu_dgesdd_(char *jobz, integer *m, integer *n, doublereal *
	a, integer *lda, doublereal *s, doublereal *u, integer *ldu, 
	doublereal *vt, integer *ldvt, doublereal *work, integer *lwork, 
	integer *iwork, integer *info);

/* Subroutine */ int _starpu_dgesv_(integer *n, integer *nrhs, doublereal *a, integer 
	*lda, integer *ipiv, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_dgesvd_(char *jobu, char *jobvt, integer *m, integer *n, 
	doublereal *a, integer *lda, doublereal *s, doublereal *u, integer *
	ldu, doublereal *vt, integer *ldvt, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_dgesvj_(char *joba, char *jobu, char *jobv, integer *m, 
	integer *n, doublereal *a, integer *lda, doublereal *sva, integer *mv, 
	 doublereal *v, integer *ldv, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_dgesvx_(char *fact, char *trans, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	integer *ipiv, char *equed, doublereal *r__, doublereal *c__, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	rcond, doublereal *ferr, doublereal *berr, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_dgesvxx_(char *fact, char *trans, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	integer *ipiv, char *equed, doublereal *r__, doublereal *c__, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	rcond, doublereal *rpvgrw, doublereal *berr, integer *n_err_bnds__, 
	doublereal *err_bnds_norm__, doublereal *err_bnds_comp__, integer *
	nparams, doublereal *params, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_dgetc2_(integer *n, doublereal *a, integer *lda, integer 
	*ipiv, integer *jpiv, integer *info);

/* Subroutine */ int _starpu_dgetf2_(integer *m, integer *n, doublereal *a, integer *
	lda, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_dgetrf_(integer *m, integer *n, doublereal *a, integer *
	lda, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_dgetri_(integer *n, doublereal *a, integer *lda, integer 
	*ipiv, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dgetrs_(char *trans, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, integer *ipiv, doublereal *b, integer *
	ldb, integer *info);

/* Subroutine */ int _starpu_dggbak_(char *job, char *side, integer *n, integer *ilo, 
	integer *ihi, doublereal *lscale, doublereal *rscale, integer *m, 
	doublereal *v, integer *ldv, integer *info);

/* Subroutine */ int _starpu_dggbal_(char *job, integer *n, doublereal *a, integer *
	lda, doublereal *b, integer *ldb, integer *ilo, integer *ihi, 
	doublereal *lscale, doublereal *rscale, doublereal *work, integer *
	info);

/* Subroutine */ int _starpu_dgges_(char *jobvsl, char *jobvsr, char *sort, L_fp 
	selctg, integer *n, doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, integer *sdim, doublereal *alphar, doublereal *alphai, 
	doublereal *beta, doublereal *vsl, integer *ldvsl, doublereal *vsr, 
	integer *ldvsr, doublereal *work, integer *lwork, logical *bwork, 
	integer *info);

/* Subroutine */ int _starpu_dggesx_(char *jobvsl, char *jobvsr, char *sort, L_fp 
	selctg, char *sense, integer *n, doublereal *a, integer *lda, 
	doublereal *b, integer *ldb, integer *sdim, doublereal *alphar, 
	doublereal *alphai, doublereal *beta, doublereal *vsl, integer *ldvsl, 
	 doublereal *vsr, integer *ldvsr, doublereal *rconde, doublereal *
	rcondv, doublereal *work, integer *lwork, integer *iwork, integer *
	liwork, logical *bwork, integer *info);

/* Subroutine */ int _starpu_dggev_(char *jobvl, char *jobvr, integer *n, doublereal *
	a, integer *lda, doublereal *b, integer *ldb, doublereal *alphar, 
	doublereal *alphai, doublereal *beta, doublereal *vl, integer *ldvl, 
	doublereal *vr, integer *ldvr, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_dggevx_(char *balanc, char *jobvl, char *jobvr, char *
	sense, integer *n, doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *alphar, doublereal *alphai, doublereal *
	beta, doublereal *vl, integer *ldvl, doublereal *vr, integer *ldvr, 
	integer *ilo, integer *ihi, doublereal *lscale, doublereal *rscale, 
	doublereal *abnrm, doublereal *bbnrm, doublereal *rconde, doublereal *
	rcondv, doublereal *work, integer *lwork, integer *iwork, logical *
	bwork, integer *info);

/* Subroutine */ int _starpu_dggglm_(integer *n, integer *m, integer *p, doublereal *
	a, integer *lda, doublereal *b, integer *ldb, doublereal *d__, 
	doublereal *x, doublereal *y, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_dgghrd_(char *compq, char *compz, integer *n, integer *
	ilo, integer *ihi, doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *q, integer *ldq, doublereal *z__, integer *
	ldz, integer *info);

/* Subroutine */ int _starpu_dgglse_(integer *m, integer *n, integer *p, doublereal *
	a, integer *lda, doublereal *b, integer *ldb, doublereal *c__, 
	doublereal *d__, doublereal *x, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_dggqrf_(integer *n, integer *m, integer *p, doublereal *
	a, integer *lda, doublereal *taua, doublereal *b, integer *ldb, 
	doublereal *taub, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dggrqf_(integer *m, integer *p, integer *n, doublereal *
	a, integer *lda, doublereal *taua, doublereal *b, integer *ldb, 
	doublereal *taub, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dggsvd_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *n, integer *p, integer *k, integer *l, doublereal *a, 
	integer *lda, doublereal *b, integer *ldb, doublereal *alpha, 
	doublereal *beta, doublereal *u, integer *ldu, doublereal *v, integer 
	*ldv, doublereal *q, integer *ldq, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_dggsvp_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *p, integer *n, doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *tola, doublereal *tolb, integer *k, integer 
	*l, doublereal *u, integer *ldu, doublereal *v, integer *ldv, 
	doublereal *q, integer *ldq, integer *iwork, doublereal *tau, 
	doublereal *work, integer *info);

/* Subroutine */ int _starpu_dgsvj0_(char *jobv, integer *m, integer *n, doublereal *
	a, integer *lda, doublereal *d__, doublereal *sva, integer *mv, 
	doublereal *v, integer *ldv, doublereal *eps, doublereal *sfmin, 
	doublereal *tol, integer *nsweep, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_dgsvj1_(char *jobv, integer *m, integer *n, integer *n1, 
	doublereal *a, integer *lda, doublereal *d__, doublereal *sva, 
	integer *mv, doublereal *v, integer *ldv, doublereal *eps, doublereal 
	*sfmin, doublereal *tol, integer *nsweep, doublereal *work, integer *
	lwork, integer *info);

/* Subroutine */ int _starpu_dgtcon_(char *norm, integer *n, doublereal *dl, 
	doublereal *d__, doublereal *du, doublereal *du2, integer *ipiv, 
	doublereal *anorm, doublereal *rcond, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_dgtrfs_(char *trans, integer *n, integer *nrhs, 
	doublereal *dl, doublereal *d__, doublereal *du, doublereal *dlf, 
	doublereal *df, doublereal *duf, doublereal *du2, integer *ipiv, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	ferr, doublereal *berr, doublereal *work, integer *iwork, integer *
	info);

/* Subroutine */ int _starpu_dgtsv_(integer *n, integer *nrhs, doublereal *dl, 
	doublereal *d__, doublereal *du, doublereal *b, integer *ldb, integer 
	*info);

/* Subroutine */ int _starpu_dgtsvx_(char *fact, char *trans, integer *n, integer *
	nrhs, doublereal *dl, doublereal *d__, doublereal *du, doublereal *
	dlf, doublereal *df, doublereal *duf, doublereal *du2, integer *ipiv, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	rcond, doublereal *ferr, doublereal *berr, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_dgttrf_(integer *n, doublereal *dl, doublereal *d__, 
	doublereal *du, doublereal *du2, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_dgttrs_(char *trans, integer *n, integer *nrhs, 
	doublereal *dl, doublereal *d__, doublereal *du, doublereal *du2, 
	integer *ipiv, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_dgtts2_(integer *itrans, integer *n, integer *nrhs, 
	doublereal *dl, doublereal *d__, doublereal *du, doublereal *du2, 
	integer *ipiv, doublereal *b, integer *ldb);

/* Subroutine */ int _starpu_dhgeqz_(char *job, char *compq, char *compz, integer *n, 
	integer *ilo, integer *ihi, doublereal *h__, integer *ldh, doublereal 
	*t, integer *ldt, doublereal *alphar, doublereal *alphai, doublereal *
	beta, doublereal *q, integer *ldq, doublereal *z__, integer *ldz, 
	doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dhsein_(char *side, char *eigsrc, char *initv, logical *
	select, integer *n, doublereal *h__, integer *ldh, doublereal *wr, 
	doublereal *wi, doublereal *vl, integer *ldvl, doublereal *vr, 
	integer *ldvr, integer *mm, integer *m, doublereal *work, integer *
	ifaill, integer *ifailr, integer *info);

/* Subroutine */ int _starpu_dhseqr_(char *job, char *compz, integer *n, integer *ilo, 
	 integer *ihi, doublereal *h__, integer *ldh, doublereal *wr, 
	doublereal *wi, doublereal *z__, integer *ldz, doublereal *work, 
	integer *lwork, integer *info);

logical _starpu_disnan_(doublereal *din);

/* Subroutine */ int _starpu_dla_gbamv__(integer *trans, integer *m, integer *n, 
	integer *kl, integer *ku, doublereal *alpha, doublereal *ab, integer *
	ldab, doublereal *x, integer *incx, doublereal *beta, doublereal *y, 
	integer *incy);

doublereal _starpu_dla_gbrcond__(char *trans, integer *n, integer *kl, integer *ku, 
	doublereal *ab, integer *ldab, doublereal *afb, integer *ldafb, 
	integer *ipiv, integer *cmode, doublereal *c__, integer *info, 
	doublereal *work, integer *iwork, ftnlen trans_len);

/* Subroutine */ int _starpu_dla_gbrfsx_extended__(integer *prec_type__, integer *
	trans_type__, integer *n, integer *kl, integer *ku, integer *nrhs, 
	doublereal *ab, integer *ldab, doublereal *afb, integer *ldafb, 
	integer *ipiv, logical *colequ, doublereal *c__, doublereal *b, 
	integer *ldb, doublereal *y, integer *ldy, doublereal *berr_out__, 
	integer *n_norms__, doublereal *errs_n__, doublereal *errs_c__, 
	doublereal *res, doublereal *ayb, doublereal *dy, doublereal *
	y_tail__, doublereal *rcond, integer *ithresh, doublereal *rthresh, 
	doublereal *dz_ub__, logical *ignore_cwise__, integer *info);

doublereal _starpu_dla_gbrpvgrw__(integer *n, integer *kl, integer *ku, integer *
	ncols, doublereal *ab, integer *ldab, doublereal *afb, integer *ldafb);

/* Subroutine */ int _starpu_dla_geamv__(integer *trans, integer *m, integer *n, 
	doublereal *alpha, doublereal *a, integer *lda, doublereal *x, 
	integer *incx, doublereal *beta, doublereal *y, integer *incy);

doublereal _starpu_dla_gercond__(char *trans, integer *n, doublereal *a, integer *lda,
	 doublereal *af, integer *ldaf, integer *ipiv, integer *cmode, 
	doublereal *c__, integer *info, doublereal *work, integer *iwork, 
	ftnlen trans_len);

/* Subroutine */ int _starpu_dla_gerfsx_extended__(integer *prec_type__, integer *
	trans_type__, integer *n, integer *nrhs, doublereal *a, integer *lda, 
	doublereal *af, integer *ldaf, integer *ipiv, logical *colequ, 
	doublereal *c__, doublereal *b, integer *ldb, doublereal *y, integer *
	ldy, doublereal *berr_out__, integer *n_norms__, doublereal *errs_n__,
	 doublereal *errs_c__, doublereal *res, doublereal *ayb, doublereal *
	dy, doublereal *y_tail__, doublereal *rcond, integer *ithresh, 
	doublereal *rthresh, doublereal *dz_ub__, logical *ignore_cwise__, 
	integer *info);

/* Subroutine */ int _starpu_dla_lin_berr__(integer *n, integer *nz, integer *nrhs, 
	doublereal *res, doublereal *ayb, doublereal *berr);

doublereal _starpu_dla_porcond__(char *uplo, integer *n, doublereal *a, integer *lda, 
	doublereal *af, integer *ldaf, integer *cmode, doublereal *c__, 
	integer *info, doublereal *work, integer *iwork, ftnlen uplo_len);

/* Subroutine */ int _starpu_dla_porfsx_extended__(integer *prec_type__, char *uplo, 
	integer *n, integer *nrhs, doublereal *a, integer *lda, doublereal *
	af, integer *ldaf, logical *colequ, doublereal *c__, doublereal *b, 
	integer *ldb, doublereal *y, integer *ldy, doublereal *berr_out__, 
	integer *n_norms__, doublereal *errs_n__, doublereal *errs_c__, 
	doublereal *res, doublereal *ayb, doublereal *dy, doublereal *
	y_tail__, doublereal *rcond, integer *ithresh, doublereal *rthresh, 
	doublereal *dz_ub__, logical *ignore_cwise__, integer *info, ftnlen 
	uplo_len);

doublereal _starpu_dla_porpvgrw__(char *uplo, integer *ncols, doublereal *a, integer *
	lda, doublereal *af, integer *ldaf, doublereal *work, ftnlen uplo_len);

doublereal _starpu_dla_rpvgrw__(integer *n, integer *ncols, doublereal *a, integer *
	lda, doublereal *af, integer *ldaf);

/* Subroutine */ int _starpu_dla_syamv__(integer *uplo, integer *n, doublereal *alpha,
	 doublereal *a, integer *lda, doublereal *x, integer *incx, 
	doublereal *beta, doublereal *y, integer *incy);

doublereal _starpu_dla_syrcond__(char *uplo, integer *n, doublereal *a, integer *lda, 
	doublereal *af, integer *ldaf, integer *ipiv, integer *cmode, 
	doublereal *c__, integer *info, doublereal *work, integer *iwork, 
	ftnlen uplo_len);

/* Subroutine */ int _starpu_dla_syrfsx_extended__(integer *prec_type__, char *uplo, 
	integer *n, integer *nrhs, doublereal *a, integer *lda, doublereal *
	af, integer *ldaf, integer *ipiv, logical *colequ, doublereal *c__, 
	doublereal *b, integer *ldb, doublereal *y, integer *ldy, doublereal *
	berr_out__, integer *n_norms__, doublereal *errs_n__, doublereal *
	errs_c__, doublereal *res, doublereal *ayb, doublereal *dy, 
	doublereal *y_tail__, doublereal *rcond, integer *ithresh, doublereal 
	*rthresh, doublereal *dz_ub__, logical *ignore_cwise__, integer *info,
	 ftnlen uplo_len);

doublereal _starpu_dla_syrpvgrw__(char *uplo, integer *n, integer *info, doublereal *
	a, integer *lda, doublereal *af, integer *ldaf, integer *ipiv, 
	doublereal *work, ftnlen uplo_len);

/* Subroutine */ int _starpu_dla_wwaddw__(integer *n, doublereal *x, doublereal *y, 
	doublereal *w);

/* Subroutine */ int _starpu_dlabad_(doublereal *small, doublereal *large);

/* Subroutine */ int _starpu_dlabrd_(integer *m, integer *n, integer *nb, doublereal *
	a, integer *lda, doublereal *d__, doublereal *e, doublereal *tauq, 
	doublereal *taup, doublereal *x, integer *ldx, doublereal *y, integer 
	*ldy);

/* Subroutine */ int _starpu_dlacn2_(integer *n, doublereal *v, doublereal *x, 
	integer *isgn, doublereal *est, integer *kase, integer *isave);

/* Subroutine */ int _starpu_dlacon_(integer *n, doublereal *v, doublereal *x, 
	integer *isgn, doublereal *est, integer *kase);

/* Subroutine */ int _starpu_dlacpy_(char *uplo, integer *m, integer *n, doublereal *
	a, integer *lda, doublereal *b, integer *ldb);

/* Subroutine */ int _starpu_dladiv_(doublereal *a, doublereal *b, doublereal *c__, 
	doublereal *d__, doublereal *p, doublereal *q);

/* Subroutine */ int _starpu_dlae2_(doublereal *a, doublereal *b, doublereal *c__, 
	doublereal *rt1, doublereal *rt2);

/* Subroutine */ int _starpu_dlaebz_(integer *ijob, integer *nitmax, integer *n, 
	integer *mmax, integer *minp, integer *nbmin, doublereal *abstol, 
	doublereal *reltol, doublereal *pivmin, doublereal *d__, doublereal *
	e, doublereal *e2, integer *nval, doublereal *ab, doublereal *c__, 
	integer *mout, integer *nab, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_dlaed0_(integer *icompq, integer *qsiz, integer *n, 
	doublereal *d__, doublereal *e, doublereal *q, integer *ldq, 
	doublereal *qstore, integer *ldqs, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_dlaed1_(integer *n, doublereal *d__, doublereal *q, 
	integer *ldq, integer *indxq, doublereal *rho, integer *cutpnt, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dlaed2_(integer *k, integer *n, integer *n1, doublereal *
	d__, doublereal *q, integer *ldq, integer *indxq, doublereal *rho, 
	doublereal *z__, doublereal *dlamda, doublereal *w, doublereal *q2, 
	integer *indx, integer *indxc, integer *indxp, integer *coltyp, 
	integer *info);

/* Subroutine */ int _starpu_dlaed3_(integer *k, integer *n, integer *n1, doublereal *
	d__, doublereal *q, integer *ldq, doublereal *rho, doublereal *dlamda, 
	 doublereal *q2, integer *indx, integer *ctot, doublereal *w, 
	doublereal *s, integer *info);

/* Subroutine */ int _starpu_dlaed4_(integer *n, integer *i__, doublereal *d__, 
	doublereal *z__, doublereal *delta, doublereal *rho, doublereal *dlam, 
	 integer *info);

/* Subroutine */ int _starpu_dlaed5_(integer *i__, doublereal *d__, doublereal *z__, 
	doublereal *delta, doublereal *rho, doublereal *dlam);

/* Subroutine */ int _starpu_dlaed6_(integer *kniter, logical *orgati, doublereal *
	rho, doublereal *d__, doublereal *z__, doublereal *finit, doublereal *
	tau, integer *info);

/* Subroutine */ int _starpu_dlaed7_(integer *icompq, integer *n, integer *qsiz, 
	integer *tlvls, integer *curlvl, integer *curpbm, doublereal *d__, 
	doublereal *q, integer *ldq, integer *indxq, doublereal *rho, integer 
	*cutpnt, doublereal *qstore, integer *qptr, integer *prmptr, integer *
	perm, integer *givptr, integer *givcol, doublereal *givnum, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dlaed8_(integer *icompq, integer *k, integer *n, integer 
	*qsiz, doublereal *d__, doublereal *q, integer *ldq, integer *indxq, 
	doublereal *rho, integer *cutpnt, doublereal *z__, doublereal *dlamda, 
	 doublereal *q2, integer *ldq2, doublereal *w, integer *perm, integer 
	*givptr, integer *givcol, doublereal *givnum, integer *indxp, integer 
	*indx, integer *info);

/* Subroutine */ int _starpu_dlaed9_(integer *k, integer *kstart, integer *kstop, 
	integer *n, doublereal *d__, doublereal *q, integer *ldq, doublereal *
	rho, doublereal *dlamda, doublereal *w, doublereal *s, integer *lds, 
	integer *info);

/* Subroutine */ int _starpu_dlaeda_(integer *n, integer *tlvls, integer *curlvl, 
	integer *curpbm, integer *prmptr, integer *perm, integer *givptr, 
	integer *givcol, doublereal *givnum, doublereal *q, integer *qptr, 
	doublereal *z__, doublereal *ztemp, integer *info);

/* Subroutine */ int _starpu_dlaein_(logical *rightv, logical *noinit, integer *n, 
	doublereal *h__, integer *ldh, doublereal *wr, doublereal *wi, 
	doublereal *vr, doublereal *vi, doublereal *b, integer *ldb, 
	doublereal *work, doublereal *eps3, doublereal *smlnum, doublereal *
	bignum, integer *info);

/* Subroutine */ int _starpu_dlaev2_(doublereal *a, doublereal *b, doublereal *c__, 
	doublereal *rt1, doublereal *rt2, doublereal *cs1, doublereal *sn1);

/* Subroutine */ int _starpu_dlaexc_(logical *wantq, integer *n, doublereal *t, 
	integer *ldt, doublereal *q, integer *ldq, integer *j1, integer *n1, 
	integer *n2, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dlag2_(doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *safmin, doublereal *scale1, doublereal *
	scale2, doublereal *wr1, doublereal *wr2, doublereal *wi);

/* Subroutine */ int _starpu_dlag2s_(integer *m, integer *n, doublereal *a, integer *
	lda, real *sa, integer *ldsa, integer *info);

/* Subroutine */ int _starpu_dlags2_(logical *upper, doublereal *a1, doublereal *a2, 
	doublereal *a3, doublereal *b1, doublereal *b2, doublereal *b3, 
	doublereal *csu, doublereal *snu, doublereal *csv, doublereal *snv, 
	doublereal *csq, doublereal *snq);

/* Subroutine */ int _starpu_dlagtf_(integer *n, doublereal *a, doublereal *lambda, 
	doublereal *b, doublereal *c__, doublereal *tol, doublereal *d__, 
	integer *in, integer *info);

/* Subroutine */ int _starpu_dlagtm_(char *trans, integer *n, integer *nrhs, 
	doublereal *alpha, doublereal *dl, doublereal *d__, doublereal *du, 
	doublereal *x, integer *ldx, doublereal *beta, doublereal *b, integer 
	*ldb);

/* Subroutine */ int _starpu_dlagts_(integer *job, integer *n, doublereal *a, 
	doublereal *b, doublereal *c__, doublereal *d__, integer *in, 
	doublereal *y, doublereal *tol, integer *info);

/* Subroutine */ int _starpu_dlagv2_(doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *alphar, doublereal *alphai, doublereal *
	beta, doublereal *csl, doublereal *snl, doublereal *csr, doublereal *
	snr);

/* Subroutine */ int _starpu_dlahqr_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, doublereal *h__, integer *ldh, doublereal 
	*wr, doublereal *wi, integer *iloz, integer *ihiz, doublereal *z__, 
	integer *ldz, integer *info);

/* Subroutine */ int _starpu_dlahr2_(integer *n, integer *k, integer *nb, doublereal *
	a, integer *lda, doublereal *tau, doublereal *t, integer *ldt, 
	doublereal *y, integer *ldy);

/* Subroutine */ int _starpu_dlahrd_(integer *n, integer *k, integer *nb, doublereal *
	a, integer *lda, doublereal *tau, doublereal *t, integer *ldt, 
	doublereal *y, integer *ldy);

/* Subroutine */ int _starpu_dlaic1_(integer *job, integer *j, doublereal *x, 
	doublereal *sest, doublereal *w, doublereal *gamma, doublereal *
	sestpr, doublereal *s, doublereal *c__);

logical _starpu_dlaisnan_(doublereal *din1, doublereal *din2);

/* Subroutine */ int _starpu_dlaln2_(logical *ltrans, integer *na, integer *nw, 
	doublereal *smin, doublereal *ca, doublereal *a, integer *lda, 
	doublereal *d1, doublereal *d2, doublereal *b, integer *ldb, 
	doublereal *wr, doublereal *wi, doublereal *x, integer *ldx, 
	doublereal *scale, doublereal *xnorm, integer *info);

/* Subroutine */ int _starpu_dlals0_(integer *icompq, integer *nl, integer *nr, 
	integer *sqre, integer *nrhs, doublereal *b, integer *ldb, doublereal 
	*bx, integer *ldbx, integer *perm, integer *givptr, integer *givcol, 
	integer *ldgcol, doublereal *givnum, integer *ldgnum, doublereal *
	poles, doublereal *difl, doublereal *difr, doublereal *z__, integer *
	k, doublereal *c__, doublereal *s, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dlalsa_(integer *icompq, integer *smlsiz, integer *n, 
	integer *nrhs, doublereal *b, integer *ldb, doublereal *bx, integer *
	ldbx, doublereal *u, integer *ldu, doublereal *vt, integer *k, 
	doublereal *difl, doublereal *difr, doublereal *z__, doublereal *
	poles, integer *givptr, integer *givcol, integer *ldgcol, integer *
	perm, doublereal *givnum, doublereal *c__, doublereal *s, doublereal *
	work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dlalsd_(char *uplo, integer *smlsiz, integer *n, integer 
	*nrhs, doublereal *d__, doublereal *e, doublereal *b, integer *ldb, 
	doublereal *rcond, integer *rank, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_dlamrg_(integer *n1, integer *n2, doublereal *a, integer 
	*dtrd1, integer *dtrd2, integer *index);

integer _starpu_dlaneg_(integer *n, doublereal *d__, doublereal *lld, doublereal *
	sigma, doublereal *pivmin, integer *r__);

doublereal _starpu_dlangb_(char *norm, integer *n, integer *kl, integer *ku, 
	doublereal *ab, integer *ldab, doublereal *work);

doublereal _starpu_dlange_(char *norm, integer *m, integer *n, doublereal *a, integer 
	*lda, doublereal *work);

doublereal _starpu_dlangt_(char *norm, integer *n, doublereal *dl, doublereal *d__, 
	doublereal *du);

doublereal _starpu_dlanhs_(char *norm, integer *n, doublereal *a, integer *lda, 
	doublereal *work);

doublereal _starpu_dlansb_(char *norm, char *uplo, integer *n, integer *k, doublereal 
	*ab, integer *ldab, doublereal *work);

doublereal _starpu_dlansf_(char *norm, char *transr, char *uplo, integer *n, 
	doublereal *a, doublereal *work);

doublereal _starpu_dlansp_(char *norm, char *uplo, integer *n, doublereal *ap, 
	doublereal *work);

doublereal _starpu_dlanst_(char *norm, integer *n, doublereal *d__, doublereal *e);

doublereal _starpu_dlansy_(char *norm, char *uplo, integer *n, doublereal *a, integer 
	*lda, doublereal *work);

doublereal _starpu_dlantb_(char *norm, char *uplo, char *diag, integer *n, integer *k, 
	 doublereal *ab, integer *ldab, doublereal *work);

doublereal _starpu_dlantp_(char *norm, char *uplo, char *diag, integer *n, doublereal 
	*ap, doublereal *work);

doublereal _starpu_dlantr_(char *norm, char *uplo, char *diag, integer *m, integer *n, 
	 doublereal *a, integer *lda, doublereal *work);

/* Subroutine */ int _starpu_dlanv2_(doublereal *a, doublereal *b, doublereal *c__, 
	doublereal *d__, doublereal *rt1r, doublereal *rt1i, doublereal *rt2r, 
	 doublereal *rt2i, doublereal *cs, doublereal *sn);

/* Subroutine */ int _starpu_dlapll_(integer *n, doublereal *x, integer *incx, 
	doublereal *y, integer *incy, doublereal *ssmin);

/* Subroutine */ int _starpu_dlapmt_(logical *forwrd, integer *m, integer *n, 
	doublereal *x, integer *ldx, integer *k);

doublereal _starpu_dlapy2_(doublereal *x, doublereal *y);

doublereal _starpu_dlapy3_(doublereal *x, doublereal *y, doublereal *z__);

/* Subroutine */ int _starpu_dlaqgb_(integer *m, integer *n, integer *kl, integer *ku, 
	 doublereal *ab, integer *ldab, doublereal *r__, doublereal *c__, 
	doublereal *rowcnd, doublereal *colcnd, doublereal *amax, char *equed);

/* Subroutine */ int _starpu_dlaqge_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *r__, doublereal *c__, doublereal *rowcnd, doublereal 
	*colcnd, doublereal *amax, char *equed);

/* Subroutine */ int _starpu_dlaqp2_(integer *m, integer *n, integer *offset, 
	doublereal *a, integer *lda, integer *jpvt, doublereal *tau, 
	doublereal *vn1, doublereal *vn2, doublereal *work);

/* Subroutine */ int _starpu_dlaqps_(integer *m, integer *n, integer *offset, integer 
	*nb, integer *kb, doublereal *a, integer *lda, integer *jpvt, 
	doublereal *tau, doublereal *vn1, doublereal *vn2, doublereal *auxv, 
	doublereal *f, integer *ldf);

/* Subroutine */ int _starpu_dlaqr0_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, doublereal *h__, integer *ldh, doublereal 
	*wr, doublereal *wi, integer *iloz, integer *ihiz, doublereal *z__, 
	integer *ldz, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dlaqr1_(integer *n, doublereal *h__, integer *ldh, 
	doublereal *sr1, doublereal *si1, doublereal *sr2, doublereal *si2, 
	doublereal *v);

/* Subroutine */ int _starpu_dlaqr2_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, doublereal *h__, integer *
	ldh, integer *iloz, integer *ihiz, doublereal *z__, integer *ldz, 
	integer *ns, integer *nd, doublereal *sr, doublereal *si, doublereal *
	v, integer *ldv, integer *nh, doublereal *t, integer *ldt, integer *
	nv, doublereal *wv, integer *ldwv, doublereal *work, integer *lwork);

/* Subroutine */ int _starpu_dlaqr3_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, doublereal *h__, integer *
	ldh, integer *iloz, integer *ihiz, doublereal *z__, integer *ldz, 
	integer *ns, integer *nd, doublereal *sr, doublereal *si, doublereal *
	v, integer *ldv, integer *nh, doublereal *t, integer *ldt, integer *
	nv, doublereal *wv, integer *ldwv, doublereal *work, integer *lwork);

/* Subroutine */ int _starpu_dlaqr4_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, doublereal *h__, integer *ldh, doublereal 
	*wr, doublereal *wi, integer *iloz, integer *ihiz, doublereal *z__, 
	integer *ldz, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dlaqr5_(logical *wantt, logical *wantz, integer *kacc22, 
	integer *n, integer *ktop, integer *kbot, integer *nshfts, doublereal 
	*sr, doublereal *si, doublereal *h__, integer *ldh, integer *iloz, 
	integer *ihiz, doublereal *z__, integer *ldz, doublereal *v, integer *
	ldv, doublereal *u, integer *ldu, integer *nv, doublereal *wv, 
	integer *ldwv, integer *nh, doublereal *wh, integer *ldwh);

/* Subroutine */ int _starpu_dlaqsb_(char *uplo, integer *n, integer *kd, doublereal *
	ab, integer *ldab, doublereal *s, doublereal *scond, doublereal *amax, 
	 char *equed);

/* Subroutine */ int _starpu_dlaqsp_(char *uplo, integer *n, doublereal *ap, 
	doublereal *s, doublereal *scond, doublereal *amax, char *equed);

/* Subroutine */ int _starpu_dlaqsy_(char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *s, doublereal *scond, doublereal *amax, char *equed);

/* Subroutine */ int _starpu_dlaqtr_(logical *ltran, logical *lreal, integer *n, 
	doublereal *t, integer *ldt, doublereal *b, doublereal *w, doublereal 
	*scale, doublereal *x, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dlar1v_(integer *n, integer *b1, integer *bn, doublereal 
	*lambda, doublereal *d__, doublereal *l, doublereal *ld, doublereal *
	lld, doublereal *pivmin, doublereal *gaptol, doublereal *z__, logical 
	*wantnc, integer *negcnt, doublereal *ztz, doublereal *mingma, 
	integer *r__, integer *isuppz, doublereal *nrminv, doublereal *resid, 
	doublereal *rqcorr, doublereal *work);

/* Subroutine */ int _starpu_dlar2v_(integer *n, doublereal *x, doublereal *y, 
	doublereal *z__, integer *incx, doublereal *c__, doublereal *s, 
	integer *incc);

/* Subroutine */ int _starpu_dlarf_(char *side, integer *m, integer *n, doublereal *v, 
	 integer *incv, doublereal *tau, doublereal *c__, integer *ldc, 
	doublereal *work);

/* Subroutine */ int _starpu_dlarfb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, doublereal *v, integer *
	ldv, doublereal *t, integer *ldt, doublereal *c__, integer *ldc, 
	doublereal *work, integer *ldwork);

/* Subroutine */ int _starpu_dlarfg_(integer *n, doublereal *alpha, doublereal *x, 
	integer *incx, doublereal *tau);

/* Subroutine */ int _starpu_dlarfp_(integer *n, doublereal *alpha, doublereal *x, 
	integer *incx, doublereal *tau);

/* Subroutine */ int _starpu_dlarft_(char *direct, char *storev, integer *n, integer *
	k, doublereal *v, integer *ldv, doublereal *tau, doublereal *t, 
	integer *ldt);

/* Subroutine */ int _starpu_dlarfx_(char *side, integer *m, integer *n, doublereal *
	v, doublereal *tau, doublereal *c__, integer *ldc, doublereal *work);

/* Subroutine */ int _starpu_dlargv_(integer *n, doublereal *x, integer *incx, 
	doublereal *y, integer *incy, doublereal *c__, integer *incc);

/* Subroutine */ int _starpu_dlarnv_(integer *idist, integer *iseed, integer *n, 
	doublereal *x);

/* Subroutine */ int _starpu_dlarra_(integer *n, doublereal *d__, doublereal *e, 
	doublereal *e2, doublereal *spltol, doublereal *tnrm, integer *nsplit, 
	 integer *isplit, integer *info);

/* Subroutine */ int _starpu_dlarrb_(integer *n, doublereal *d__, doublereal *lld, 
	integer *ifirst, integer *ilast, doublereal *rtol1, doublereal *rtol2, 
	 integer *offset, doublereal *w, doublereal *wgap, doublereal *werr, 
	doublereal *work, integer *iwork, doublereal *pivmin, doublereal *
	spdiam, integer *twist, integer *info);

/* Subroutine */ int _starpu_dlarrc_(char *jobt, integer *n, doublereal *vl, 
	doublereal *vu, doublereal *d__, doublereal *e, doublereal *pivmin, 
	integer *eigcnt, integer *lcnt, integer *rcnt, integer *info);

/* Subroutine */ int _starpu_dlarrd_(char *range, char *order, integer *n, doublereal 
	*vl, doublereal *vu, integer *il, integer *iu, doublereal *gers, 
	doublereal *reltol, doublereal *d__, doublereal *e, doublereal *e2, 
	doublereal *pivmin, integer *nsplit, integer *isplit, integer *m, 
	doublereal *w, doublereal *werr, doublereal *wl, doublereal *wu, 
	integer *iblock, integer *indexw, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_dlarre_(char *range, integer *n, doublereal *vl, 
	doublereal *vu, integer *il, integer *iu, doublereal *d__, doublereal 
	*e, doublereal *e2, doublereal *rtol1, doublereal *rtol2, doublereal *
	spltol, integer *nsplit, integer *isplit, integer *m, doublereal *w, 
	doublereal *werr, doublereal *wgap, integer *iblock, integer *indexw, 
	doublereal *gers, doublereal *pivmin, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_dlarrf_(integer *n, doublereal *d__, doublereal *l, 
	doublereal *ld, integer *clstrt, integer *clend, doublereal *w, 
	doublereal *wgap, doublereal *werr, doublereal *spdiam, doublereal *
	clgapl, doublereal *clgapr, doublereal *pivmin, doublereal *sigma, 
	doublereal *dplus, doublereal *lplus, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dlarrj_(integer *n, doublereal *d__, doublereal *e2, 
	integer *ifirst, integer *ilast, doublereal *rtol, integer *offset, 
	doublereal *w, doublereal *werr, doublereal *work, integer *iwork, 
	doublereal *pivmin, doublereal *spdiam, integer *info);

/* Subroutine */ int _starpu_dlarrk_(integer *n, integer *iw, doublereal *gl, 
	doublereal *gu, doublereal *d__, doublereal *e2, doublereal *pivmin, 
	doublereal *reltol, doublereal *w, doublereal *werr, integer *info);

/* Subroutine */ int _starpu_dlarrr_(integer *n, doublereal *d__, doublereal *e, 
	integer *info);

/* Subroutine */ int _starpu_dlarrv_(integer *n, doublereal *vl, doublereal *vu, 
	doublereal *d__, doublereal *l, doublereal *pivmin, integer *isplit, 
	integer *m, integer *dol, integer *dou, doublereal *minrgp, 
	doublereal *rtol1, doublereal *rtol2, doublereal *w, doublereal *werr, 
	 doublereal *wgap, integer *iblock, integer *indexw, doublereal *gers, 
	 doublereal *z__, integer *ldz, integer *isuppz, doublereal *work, 
	integer *iwork, integer *info);

/* Subroutine */ int _starpu_dlarscl2_(integer *m, integer *n, doublereal *d__, 
	doublereal *x, integer *ldx);

/* Subroutine */ int _starpu_dlartg_(doublereal *f, doublereal *g, doublereal *cs, 
	doublereal *sn, doublereal *r__);

/* Subroutine */ int _starpu_dlartv_(integer *n, doublereal *x, integer *incx, 
	doublereal *y, integer *incy, doublereal *c__, doublereal *s, integer 
	*incc);

/* Subroutine */ int _starpu_dlaruv_(integer *iseed, integer *n, doublereal *x);

/* Subroutine */ int _starpu_dlarz_(char *side, integer *m, integer *n, integer *l, 
	doublereal *v, integer *incv, doublereal *tau, doublereal *c__, 
	integer *ldc, doublereal *work);

/* Subroutine */ int _starpu_dlarzb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, integer *l, doublereal *v, 
	 integer *ldv, doublereal *t, integer *ldt, doublereal *c__, integer *
	ldc, doublereal *work, integer *ldwork);

/* Subroutine */ int _starpu_dlarzt_(char *direct, char *storev, integer *n, integer *
	k, doublereal *v, integer *ldv, doublereal *tau, doublereal *t, 
	integer *ldt);

/* Subroutine */ int _starpu_dlas2_(doublereal *f, doublereal *g, doublereal *h__, 
	doublereal *ssmin, doublereal *ssmax);

/* Subroutine */ int _starpu_dlascl_(char *type__, integer *kl, integer *ku, 
	doublereal *cfrom, doublereal *cto, integer *m, integer *n, 
	doublereal *a, integer *lda, integer *info);

/* Subroutine */ int _starpu_dlascl2_(integer *m, integer *n, doublereal *d__, 
	doublereal *x, integer *ldx);

/* Subroutine */ int _starpu_dlasd0_(integer *n, integer *sqre, doublereal *d__, 
	doublereal *e, doublereal *u, integer *ldu, doublereal *vt, integer *
	ldvt, integer *smlsiz, integer *iwork, doublereal *work, integer *
	info);

/* Subroutine */ int _starpu_dlasd1_(integer *nl, integer *nr, integer *sqre, 
	doublereal *d__, doublereal *alpha, doublereal *beta, doublereal *u, 
	integer *ldu, doublereal *vt, integer *ldvt, integer *idxq, integer *
	iwork, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dlasd2_(integer *nl, integer *nr, integer *sqre, integer 
	*k, doublereal *d__, doublereal *z__, doublereal *alpha, doublereal *
	beta, doublereal *u, integer *ldu, doublereal *vt, integer *ldvt, 
	doublereal *dsigma, doublereal *u2, integer *ldu2, doublereal *vt2, 
	integer *ldvt2, integer *idxp, integer *idx, integer *idxc, integer *
	idxq, integer *coltyp, integer *info);

/* Subroutine */ int _starpu_dlasd3_(integer *nl, integer *nr, integer *sqre, integer 
	*k, doublereal *d__, doublereal *q, integer *ldq, doublereal *dsigma, 
	doublereal *u, integer *ldu, doublereal *u2, integer *ldu2, 
	doublereal *vt, integer *ldvt, doublereal *vt2, integer *ldvt2, 
	integer *idxc, integer *ctot, doublereal *z__, integer *info);

/* Subroutine */ int _starpu_dlasd4_(integer *n, integer *i__, doublereal *d__, 
	doublereal *z__, doublereal *delta, doublereal *rho, doublereal *
	sigma, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dlasd5_(integer *i__, doublereal *d__, doublereal *z__, 
	doublereal *delta, doublereal *rho, doublereal *dsigma, doublereal *
	work);

/* Subroutine */ int _starpu_dlasd6_(integer *icompq, integer *nl, integer *nr, 
	integer *sqre, doublereal *d__, doublereal *vf, doublereal *vl, 
	doublereal *alpha, doublereal *beta, integer *idxq, integer *perm, 
	integer *givptr, integer *givcol, integer *ldgcol, doublereal *givnum, 
	 integer *ldgnum, doublereal *poles, doublereal *difl, doublereal *
	difr, doublereal *z__, integer *k, doublereal *c__, doublereal *s, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dlasd7_(integer *icompq, integer *nl, integer *nr, 
	integer *sqre, integer *k, doublereal *d__, doublereal *z__, 
	doublereal *zw, doublereal *vf, doublereal *vfw, doublereal *vl, 
	doublereal *vlw, doublereal *alpha, doublereal *beta, doublereal *
	dsigma, integer *idx, integer *idxp, integer *idxq, integer *perm, 
	integer *givptr, integer *givcol, integer *ldgcol, doublereal *givnum, 
	 integer *ldgnum, doublereal *c__, doublereal *s, integer *info);

/* Subroutine */ int _starpu_dlasd8_(integer *icompq, integer *k, doublereal *d__, 
	doublereal *z__, doublereal *vf, doublereal *vl, doublereal *difl, 
	doublereal *difr, integer *lddifr, doublereal *dsigma, doublereal *
	work, integer *info);

/* Subroutine */ int _starpu_dlasda_(integer *icompq, integer *smlsiz, integer *n, 
	integer *sqre, doublereal *d__, doublereal *e, doublereal *u, integer 
	*ldu, doublereal *vt, integer *k, doublereal *difl, doublereal *difr, 
	doublereal *z__, doublereal *poles, integer *givptr, integer *givcol, 
	integer *ldgcol, integer *perm, doublereal *givnum, doublereal *c__, 
	doublereal *s, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dlasdq_(char *uplo, integer *sqre, integer *n, integer *
	ncvt, integer *nru, integer *ncc, doublereal *d__, doublereal *e, 
	doublereal *vt, integer *ldvt, doublereal *u, integer *ldu, 
	doublereal *c__, integer *ldc, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dlasdt_(integer *n, integer *lvl, integer *nd, integer *
	inode, integer *ndiml, integer *ndimr, integer *msub);

/* Subroutine */ int _starpu_dlaset_(char *uplo, integer *m, integer *n, doublereal *
	alpha, doublereal *beta, doublereal *a, integer *lda);

/* Subroutine */ int _starpu_dlasq1_(integer *n, doublereal *d__, doublereal *e, 
	doublereal *work, integer *info);

/* Subroutine */ int _starpu_dlasq2_(integer *n, doublereal *z__, integer *info);

/* Subroutine */ int _starpu_dlasq3_(integer *i0, integer *n0, doublereal *z__, 
	integer *pp, doublereal *dmin__, doublereal *sigma, doublereal *desig, 
	 doublereal *qmax, integer *nfail, integer *iter, integer *ndiv, 
	logical *ieee, integer *ttype, doublereal *dmin1, doublereal *dmin2, 
	doublereal *dn, doublereal *dn1, doublereal *dn2, doublereal *g, 
	doublereal *tau);

/* Subroutine */ int _starpu_dlasq4_(integer *i0, integer *n0, doublereal *z__, 
	integer *pp, integer *n0in, doublereal *dmin__, doublereal *dmin1, 
	doublereal *dmin2, doublereal *dn, doublereal *dn1, doublereal *dn2, 
	doublereal *tau, integer *ttype, doublereal *g);

/* Subroutine */ int _starpu_dlasq5_(integer *i0, integer *n0, doublereal *z__, 
	integer *pp, doublereal *tau, doublereal *dmin__, doublereal *dmin1, 
	doublereal *dmin2, doublereal *dn, doublereal *dnm1, doublereal *dnm2, 
	 logical *ieee);

/* Subroutine */ int _starpu_dlasq6_(integer *i0, integer *n0, doublereal *z__, 
	integer *pp, doublereal *dmin__, doublereal *dmin1, doublereal *dmin2, 
	 doublereal *dn, doublereal *dnm1, doublereal *dnm2);

/* Subroutine */ int _starpu_dlasr_(char *side, char *pivot, char *direct, integer *m, 
	 integer *n, doublereal *c__, doublereal *s, doublereal *a, integer *
	lda);

/* Subroutine */ int _starpu_dlasrt_(char *id, integer *n, doublereal *d__, integer *
	info);

/* Subroutine */ int _starpu_dlassq_(integer *n, doublereal *x, integer *incx, 
	doublereal *scale, doublereal *sumsq);

/* Subroutine */ int _starpu_dlasv2_(doublereal *f, doublereal *g, doublereal *h__, 
	doublereal *ssmin, doublereal *ssmax, doublereal *snr, doublereal *
	csr, doublereal *snl, doublereal *csl);

/* Subroutine */ int _starpu_dlaswp_(integer *n, doublereal *a, integer *lda, integer 
	*k1, integer *k2, integer *ipiv, integer *incx);

/* Subroutine */ int _starpu_dlasy2_(logical *ltranl, logical *ltranr, integer *isgn, 
	integer *n1, integer *n2, doublereal *tl, integer *ldtl, doublereal *
	tr, integer *ldtr, doublereal *b, integer *ldb, doublereal *scale, 
	doublereal *x, integer *ldx, doublereal *xnorm, integer *info);

/* Subroutine */ int _starpu_dlasyf_(char *uplo, integer *n, integer *nb, integer *kb, 
	 doublereal *a, integer *lda, integer *ipiv, doublereal *w, integer *
	ldw, integer *info);

/* Subroutine */ int _starpu_dlat2s_(char *uplo, integer *n, doublereal *a, integer *
	lda, real *sa, integer *ldsa, integer *info);

/* Subroutine */ int _starpu_dlatbs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, integer *kd, doublereal *ab, integer *ldab, 
	doublereal *x, doublereal *scale, doublereal *cnorm, integer *info);

/* Subroutine */ int _starpu_dlatdf_(integer *ijob, integer *n, doublereal *z__, 
	integer *ldz, doublereal *rhs, doublereal *rdsum, doublereal *rdscal, 
	integer *ipiv, integer *jpiv);

/* Subroutine */ int _starpu_dlatps_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, doublereal *ap, doublereal *x, doublereal *scale, 
	doublereal *cnorm, integer *info);

/* Subroutine */ int _starpu_dlatrd_(char *uplo, integer *n, integer *nb, doublereal *
	a, integer *lda, doublereal *e, doublereal *tau, doublereal *w, 
	integer *ldw);

/* Subroutine */ int _starpu_dlatrs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, doublereal *a, integer *lda, doublereal *x, 
	doublereal *scale, doublereal *cnorm, integer *info);

/* Subroutine */ int _starpu_dlatrz_(integer *m, integer *n, integer *l, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work);

/* Subroutine */ int _starpu_dlatzm_(char *side, integer *m, integer *n, doublereal *
	v, integer *incv, doublereal *tau, doublereal *c1, doublereal *c2, 
	integer *ldc, doublereal *work);

/* Subroutine */ int _starpu_dlauu2_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *info);

/* Subroutine */ int _starpu_dlauum_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *info);

/* Subroutine */ int _starpu_dopgtr_(char *uplo, integer *n, doublereal *ap, 
	doublereal *tau, doublereal *q, integer *ldq, doublereal *work, 
	integer *info);

/* Subroutine */ int _starpu_dopmtr_(char *side, char *uplo, char *trans, integer *m, 
	integer *n, doublereal *ap, doublereal *tau, doublereal *c__, integer 
	*ldc, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dorg2l_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dorg2r_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dorgbr_(char *vect, integer *m, integer *n, integer *k, 
	doublereal *a, integer *lda, doublereal *tau, doublereal *work, 
	integer *lwork, integer *info);

/* Subroutine */ int _starpu_dorghr_(integer *n, integer *ilo, integer *ihi, 
	doublereal *a, integer *lda, doublereal *tau, doublereal *work, 
	integer *lwork, integer *info);

/* Subroutine */ int _starpu_dorgl2_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dorglq_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_dorgql_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_dorgqr_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_dorgr2_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dorgrq_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_dorgtr_(char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dorm2l_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dorm2r_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dormbr_(char *vect, char *side, char *trans, integer *m, 
	integer *n, integer *k, doublereal *a, integer *lda, doublereal *tau, 
	doublereal *c__, integer *ldc, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_dormhr_(char *side, char *trans, integer *m, integer *n, 
	integer *ilo, integer *ihi, doublereal *a, integer *lda, doublereal *
	tau, doublereal *c__, integer *ldc, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_dorml2_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dormlq_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dormql_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dormqr_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dormr2_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dormr3_(char *side, char *trans, integer *m, integer *n, 
	integer *k, integer *l, doublereal *a, integer *lda, doublereal *tau, 
	doublereal *c__, integer *ldc, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dormrq_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dormrz_(char *side, char *trans, integer *m, integer *n, 
	integer *k, integer *l, doublereal *a, integer *lda, doublereal *tau, 
	doublereal *c__, integer *ldc, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_dormtr_(char *side, char *uplo, char *trans, integer *m, 
	integer *n, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dpbcon_(char *uplo, integer *n, integer *kd, doublereal *
	ab, integer *ldab, doublereal *anorm, doublereal *rcond, doublereal *
	work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dpbequ_(char *uplo, integer *n, integer *kd, doublereal *
	ab, integer *ldab, doublereal *s, doublereal *scond, doublereal *amax, 
	 integer *info);

/* Subroutine */ int _starpu_dpbrfs_(char *uplo, integer *n, integer *kd, integer *
	nrhs, doublereal *ab, integer *ldab, doublereal *afb, integer *ldafb, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	ferr, doublereal *berr, doublereal *work, integer *iwork, integer *
	info);

/* Subroutine */ int _starpu_dpbstf_(char *uplo, integer *n, integer *kd, doublereal *
	ab, integer *ldab, integer *info);

/* Subroutine */ int _starpu_dpbsv_(char *uplo, integer *n, integer *kd, integer *
	nrhs, doublereal *ab, integer *ldab, doublereal *b, integer *ldb, 
	integer *info);

/* Subroutine */ int _starpu_dpbsvx_(char *fact, char *uplo, integer *n, integer *kd, 
	integer *nrhs, doublereal *ab, integer *ldab, doublereal *afb, 
	integer *ldafb, char *equed, doublereal *s, doublereal *b, integer *
	ldb, doublereal *x, integer *ldx, doublereal *rcond, doublereal *ferr, 
	 doublereal *berr, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dpbtf2_(char *uplo, integer *n, integer *kd, doublereal *
	ab, integer *ldab, integer *info);

/* Subroutine */ int _starpu_dpbtrf_(char *uplo, integer *n, integer *kd, doublereal *
	ab, integer *ldab, integer *info);

/* Subroutine */ int _starpu_dpbtrs_(char *uplo, integer *n, integer *kd, integer *
	nrhs, doublereal *ab, integer *ldab, doublereal *b, integer *ldb, 
	integer *info);

/* Subroutine */ int _starpu_dpftrf_(char *transr, char *uplo, integer *n, doublereal 
	*a, integer *info);

/* Subroutine */ int _starpu_dpftri_(char *transr, char *uplo, integer *n, doublereal 
	*a, integer *info);

/* Subroutine */ int _starpu_dpftrs_(char *transr, char *uplo, integer *n, integer *
	nrhs, doublereal *a, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_dpocon_(char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *anorm, doublereal *rcond, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_dpoequ_(integer *n, doublereal *a, integer *lda, 
	doublereal *s, doublereal *scond, doublereal *amax, integer *info);

/* Subroutine */ int _starpu_dpoequb_(integer *n, doublereal *a, integer *lda, 
	doublereal *s, doublereal *scond, doublereal *amax, integer *info);

/* Subroutine */ int _starpu_dporfs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	ferr, doublereal *berr, doublereal *work, integer *iwork, integer *
	info);

/* Subroutine */ int _starpu_dporfsx_(char *uplo, char *equed, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	doublereal *s, doublereal *b, integer *ldb, doublereal *x, integer *
	ldx, doublereal *rcond, doublereal *berr, integer *n_err_bnds__, 
	doublereal *err_bnds_norm__, doublereal *err_bnds_comp__, integer *
	nparams, doublereal *params, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_dposv_(char *uplo, integer *n, integer *nrhs, doublereal 
	*a, integer *lda, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_dposvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	char *equed, doublereal *s, doublereal *b, integer *ldb, doublereal *
	x, integer *ldx, doublereal *rcond, doublereal *ferr, doublereal *
	berr, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dposvxx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	char *equed, doublereal *s, doublereal *b, integer *ldb, doublereal *
	x, integer *ldx, doublereal *rcond, doublereal *rpvgrw, doublereal *
	berr, integer *n_err_bnds__, doublereal *err_bnds_norm__, doublereal *
	err_bnds_comp__, integer *nparams, doublereal *params, doublereal *
	work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dpotf2_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *info);

/* Subroutine */ int _starpu_dpotrf_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *info);

/* Subroutine */ int _starpu_dpotri_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *info);

/* Subroutine */ int _starpu_dpotrs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, integer *
	info);

/* Subroutine */ int _starpu_dppcon_(char *uplo, integer *n, doublereal *ap, 
	doublereal *anorm, doublereal *rcond, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_dppequ_(char *uplo, integer *n, doublereal *ap, 
	doublereal *s, doublereal *scond, doublereal *amax, integer *info);

/* Subroutine */ int _starpu_dpprfs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *ap, doublereal *afp, doublereal *b, integer *ldb, 
	doublereal *x, integer *ldx, doublereal *ferr, doublereal *berr, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dppsv_(char *uplo, integer *n, integer *nrhs, doublereal 
	*ap, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_dppsvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublereal *ap, doublereal *afp, char *equed, doublereal *s, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	rcond, doublereal *ferr, doublereal *berr, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_dpptrf_(char *uplo, integer *n, doublereal *ap, integer *
	info);

/* Subroutine */ int _starpu_dpptri_(char *uplo, integer *n, doublereal *ap, integer *
	info);

/* Subroutine */ int _starpu_dpptrs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *ap, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_dpstf2_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *piv, integer *rank, doublereal *tol, doublereal *work, 
	integer *info);

/* Subroutine */ int _starpu_dpstrf_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *piv, integer *rank, doublereal *tol, doublereal *work, 
	integer *info);

/* Subroutine */ int _starpu_dptcon_(integer *n, doublereal *d__, doublereal *e, 
	doublereal *anorm, doublereal *rcond, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dpteqr_(char *compz, integer *n, doublereal *d__, 
	doublereal *e, doublereal *z__, integer *ldz, doublereal *work, 
	integer *info);

/* Subroutine */ int _starpu_dptrfs_(integer *n, integer *nrhs, doublereal *d__, 
	doublereal *e, doublereal *df, doublereal *ef, doublereal *b, integer 
	*ldb, doublereal *x, integer *ldx, doublereal *ferr, doublereal *berr, 
	 doublereal *work, integer *info);

/* Subroutine */ int _starpu_dptsv_(integer *n, integer *nrhs, doublereal *d__, 
	doublereal *e, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_dptsvx_(char *fact, integer *n, integer *nrhs, 
	doublereal *d__, doublereal *e, doublereal *df, doublereal *ef, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	rcond, doublereal *ferr, doublereal *berr, doublereal *work, integer *
	info);

/* Subroutine */ int _starpu_dpttrf_(integer *n, doublereal *d__, doublereal *e, 
	integer *info);

/* Subroutine */ int _starpu_dpttrs_(integer *n, integer *nrhs, doublereal *d__, 
	doublereal *e, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_dptts2_(integer *n, integer *nrhs, doublereal *d__, 
	doublereal *e, doublereal *b, integer *ldb);

/* Subroutine */ int _starpu_drscl_(integer *n, doublereal *sa, doublereal *sx, 
	integer *incx);

/* Subroutine */ int _starpu_dsbev_(char *jobz, char *uplo, integer *n, integer *kd, 
	doublereal *ab, integer *ldab, doublereal *w, doublereal *z__, 
	integer *ldz, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dsbevd_(char *jobz, char *uplo, integer *n, integer *kd, 
	doublereal *ab, integer *ldab, doublereal *w, doublereal *z__, 
	integer *ldz, doublereal *work, integer *lwork, integer *iwork, 
	integer *liwork, integer *info);

/* Subroutine */ int _starpu_dsbevx_(char *jobz, char *range, char *uplo, integer *n, 
	integer *kd, doublereal *ab, integer *ldab, doublereal *q, integer *
	ldq, doublereal *vl, doublereal *vu, integer *il, integer *iu, 
	doublereal *abstol, integer *m, doublereal *w, doublereal *z__, 
	integer *ldz, doublereal *work, integer *iwork, integer *ifail, 
	integer *info);

/* Subroutine */ int _starpu_dsbgst_(char *vect, char *uplo, integer *n, integer *ka, 
	integer *kb, doublereal *ab, integer *ldab, doublereal *bb, integer *
	ldbb, doublereal *x, integer *ldx, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dsbgv_(char *jobz, char *uplo, integer *n, integer *ka, 
	integer *kb, doublereal *ab, integer *ldab, doublereal *bb, integer *
	ldbb, doublereal *w, doublereal *z__, integer *ldz, doublereal *work, 
	integer *info);

/* Subroutine */ int _starpu_dsbgvd_(char *jobz, char *uplo, integer *n, integer *ka, 
	integer *kb, doublereal *ab, integer *ldab, doublereal *bb, integer *
	ldbb, doublereal *w, doublereal *z__, integer *ldz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_dsbgvx_(char *jobz, char *range, char *uplo, integer *n, 
	integer *ka, integer *kb, doublereal *ab, integer *ldab, doublereal *
	bb, integer *ldbb, doublereal *q, integer *ldq, doublereal *vl, 
	doublereal *vu, integer *il, integer *iu, doublereal *abstol, integer 
	*m, doublereal *w, doublereal *z__, integer *ldz, doublereal *work, 
	integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu_dsbtrd_(char *vect, char *uplo, integer *n, integer *kd, 
	doublereal *ab, integer *ldab, doublereal *d__, doublereal *e, 
	doublereal *q, integer *ldq, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dsfrk_(char *transr, char *uplo, char *trans, integer *n, 
	 integer *k, doublereal *alpha, doublereal *a, integer *lda, 
	doublereal *beta, doublereal *c__);

/* Subroutine */ int _starpu__starpu_dsgesv_(integer *n, integer *nrhs, doublereal *a, 
	integer *lda, integer *ipiv, doublereal *b, integer *ldb, doublereal *
	x, integer *ldx, doublereal *work, real *swork, integer *iter, 
	integer *info);

/* Subroutine */ int _starpu_dspcon_(char *uplo, integer *n, doublereal *ap, integer *
	ipiv, doublereal *anorm, doublereal *rcond, doublereal *work, integer 
	*iwork, integer *info);

/* Subroutine */ int _starpu_dspev_(char *jobz, char *uplo, integer *n, doublereal *
	ap, doublereal *w, doublereal *z__, integer *ldz, doublereal *work, 
	integer *info);

/* Subroutine */ int _starpu_dspevd_(char *jobz, char *uplo, integer *n, doublereal *
	ap, doublereal *w, doublereal *z__, integer *ldz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_dspevx_(char *jobz, char *range, char *uplo, integer *n, 
	doublereal *ap, doublereal *vl, doublereal *vu, integer *il, integer *
	iu, doublereal *abstol, integer *m, doublereal *w, doublereal *z__, 
	integer *ldz, doublereal *work, integer *iwork, integer *ifail, 
	integer *info);

/* Subroutine */ int _starpu_dspgst_(integer *itype, char *uplo, integer *n, 
	doublereal *ap, doublereal *bp, integer *info);

/* Subroutine */ int _starpu_dspgv_(integer *itype, char *jobz, char *uplo, integer *
	n, doublereal *ap, doublereal *bp, doublereal *w, doublereal *z__, 
	integer *ldz, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dspgvd_(integer *itype, char *jobz, char *uplo, integer *
	n, doublereal *ap, doublereal *bp, doublereal *w, doublereal *z__, 
	integer *ldz, doublereal *work, integer *lwork, integer *iwork, 
	integer *liwork, integer *info);

/* Subroutine */ int _starpu_dspgvx_(integer *itype, char *jobz, char *range, char *
	uplo, integer *n, doublereal *ap, doublereal *bp, doublereal *vl, 
	doublereal *vu, integer *il, integer *iu, doublereal *abstol, integer 
	*m, doublereal *w, doublereal *z__, integer *ldz, doublereal *work, 
	integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu__starpu_dsposv_(char *uplo, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *
	x, integer *ldx, doublereal *work, real *swork, integer *iter, 
	integer *info);

/* Subroutine */ int _starpu_dsprfs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *ap, doublereal *afp, integer *ipiv, doublereal *b, 
	integer *ldb, doublereal *x, integer *ldx, doublereal *ferr, 
	doublereal *berr, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dspsv_(char *uplo, integer *n, integer *nrhs, doublereal 
	*ap, integer *ipiv, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_dspsvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublereal *ap, doublereal *afp, integer *ipiv, doublereal *b, 
	integer *ldb, doublereal *x, integer *ldx, doublereal *rcond, 
	doublereal *ferr, doublereal *berr, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_dsptrd_(char *uplo, integer *n, doublereal *ap, 
	doublereal *d__, doublereal *e, doublereal *tau, integer *info);

/* Subroutine */ int _starpu_dsptrf_(char *uplo, integer *n, doublereal *ap, integer *
	ipiv, integer *info);

/* Subroutine */ int _starpu_dsptri_(char *uplo, integer *n, doublereal *ap, integer *
	ipiv, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dsptrs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *ap, integer *ipiv, doublereal *b, integer *ldb, integer *
	info);

/* Subroutine */ int _starpu_dstebz_(char *range, char *order, integer *n, doublereal 
	*vl, doublereal *vu, integer *il, integer *iu, doublereal *abstol, 
	doublereal *d__, doublereal *e, integer *m, integer *nsplit, 
	doublereal *w, integer *iblock, integer *isplit, doublereal *work, 
	integer *iwork, integer *info);

/* Subroutine */ int _starpu_dstedc_(char *compz, integer *n, doublereal *d__, 
	doublereal *e, doublereal *z__, integer *ldz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_dstegr_(char *jobz, char *range, integer *n, doublereal *
	d__, doublereal *e, doublereal *vl, doublereal *vu, integer *il, 
	integer *iu, doublereal *abstol, integer *m, doublereal *w, 
	doublereal *z__, integer *ldz, integer *isuppz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_dstein_(integer *n, doublereal *d__, doublereal *e, 
	integer *m, doublereal *w, integer *iblock, integer *isplit, 
	doublereal *z__, integer *ldz, doublereal *work, integer *iwork, 
	integer *ifail, integer *info);

/* Subroutine */ int _starpu_dstemr_(char *jobz, char *range, integer *n, doublereal *
	d__, doublereal *e, doublereal *vl, doublereal *vu, integer *il, 
	integer *iu, integer *m, doublereal *w, doublereal *z__, integer *ldz, 
	 integer *nzc, integer *isuppz, logical *tryrac, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_dsteqr_(char *compz, integer *n, doublereal *d__, 
	doublereal *e, doublereal *z__, integer *ldz, doublereal *work, 
	integer *info);

/* Subroutine */ int _starpu_dsterf_(integer *n, doublereal *d__, doublereal *e, 
	integer *info);

/* Subroutine */ int _starpu_dstev_(char *jobz, integer *n, doublereal *d__, 
	doublereal *e, doublereal *z__, integer *ldz, doublereal *work, 
	integer *info);

/* Subroutine */ int _starpu_dstevd_(char *jobz, integer *n, doublereal *d__, 
	doublereal *e, doublereal *z__, integer *ldz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_dstevr_(char *jobz, char *range, integer *n, doublereal *
	d__, doublereal *e, doublereal *vl, doublereal *vu, integer *il, 
	integer *iu, doublereal *abstol, integer *m, doublereal *w, 
	doublereal *z__, integer *ldz, integer *isuppz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_dstevx_(char *jobz, char *range, integer *n, doublereal *
	d__, doublereal *e, doublereal *vl, doublereal *vu, integer *il, 
	integer *iu, doublereal *abstol, integer *m, doublereal *w, 
	doublereal *z__, integer *ldz, doublereal *work, integer *iwork, 
	integer *ifail, integer *info);

/* Subroutine */ int _starpu_dsycon_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *ipiv, doublereal *anorm, doublereal *rcond, doublereal *
	work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dsyequb_(char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *s, doublereal *scond, doublereal *amax, doublereal *
	work, integer *info);

/* Subroutine */ int _starpu_dsyev_(char *jobz, char *uplo, integer *n, doublereal *a, 
	 integer *lda, doublereal *w, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_dsyevd_(char *jobz, char *uplo, integer *n, doublereal *
	a, integer *lda, doublereal *w, doublereal *work, integer *lwork, 
	integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_dsyevr_(char *jobz, char *range, char *uplo, integer *n, 
	doublereal *a, integer *lda, doublereal *vl, doublereal *vu, integer *
	il, integer *iu, doublereal *abstol, integer *m, doublereal *w, 
	doublereal *z__, integer *ldz, integer *isuppz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_dsyevx_(char *jobz, char *range, char *uplo, integer *n, 
	doublereal *a, integer *lda, doublereal *vl, doublereal *vu, integer *
	il, integer *iu, doublereal *abstol, integer *m, doublereal *w, 
	doublereal *z__, integer *ldz, doublereal *work, integer *lwork, 
	integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu_dsygs2_(integer *itype, char *uplo, integer *n, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, integer *
	info);

/* Subroutine */ int _starpu_dsygst_(integer *itype, char *uplo, integer *n, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, integer *
	info);

/* Subroutine */ int _starpu_dsygv_(integer *itype, char *jobz, char *uplo, integer *
	n, doublereal *a, integer *lda, doublereal *b, integer *ldb, 
	doublereal *w, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dsygvd_(integer *itype, char *jobz, char *uplo, integer *
	n, doublereal *a, integer *lda, doublereal *b, integer *ldb, 
	doublereal *w, doublereal *work, integer *lwork, integer *iwork, 
	integer *liwork, integer *info);

/* Subroutine */ int _starpu_dsygvx_(integer *itype, char *jobz, char *range, char *
	uplo, integer *n, doublereal *a, integer *lda, doublereal *b, integer 
	*ldb, doublereal *vl, doublereal *vu, integer *il, integer *iu, 
	doublereal *abstol, integer *m, doublereal *w, doublereal *z__, 
	integer *ldz, doublereal *work, integer *lwork, integer *iwork, 
	integer *ifail, integer *info);

/* Subroutine */ int _starpu_dsyrfs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *af, integer *ldaf, integer *
	ipiv, doublereal *b, integer *ldb, doublereal *x, integer *ldx, 
	doublereal *ferr, doublereal *berr, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_dsyrfsx_(char *uplo, char *equed, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	integer *ipiv, doublereal *s, doublereal *b, integer *ldb, doublereal 
	*x, integer *ldx, doublereal *rcond, doublereal *berr, integer *
	n_err_bnds__, doublereal *err_bnds_norm__, doublereal *
	err_bnds_comp__, integer *nparams, doublereal *params, doublereal *
	work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dsysv_(char *uplo, integer *n, integer *nrhs, doublereal 
	*a, integer *lda, integer *ipiv, doublereal *b, integer *ldb, 
	doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dsysvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	integer *ipiv, doublereal *b, integer *ldb, doublereal *x, integer *
	ldx, doublereal *rcond, doublereal *ferr, doublereal *berr, 
	doublereal *work, integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dsysvxx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	integer *ipiv, char *equed, doublereal *s, doublereal *b, integer *
	ldb, doublereal *x, integer *ldx, doublereal *rcond, doublereal *
	rpvgrw, doublereal *berr, integer *n_err_bnds__, doublereal *
	err_bnds_norm__, doublereal *err_bnds_comp__, integer *nparams, 
	doublereal *params, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dsytd2_(char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *d__, doublereal *e, doublereal *tau, integer *info);

/* Subroutine */ int _starpu_dsytf2_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_dsytrd_(char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *d__, doublereal *e, doublereal *tau, doublereal *
	work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dsytrf_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *ipiv, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dsytri_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *ipiv, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dsytrs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, integer *ipiv, doublereal *b, integer *
	ldb, integer *info);

/* Subroutine */ int _starpu_dtbcon_(char *norm, char *uplo, char *diag, integer *n, 
	integer *kd, doublereal *ab, integer *ldab, doublereal *rcond, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dtbrfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *kd, integer *nrhs, doublereal *ab, integer *ldab, doublereal 
	*b, integer *ldb, doublereal *x, integer *ldx, doublereal *ferr, 
	doublereal *berr, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dtbtrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *kd, integer *nrhs, doublereal *ab, integer *ldab, doublereal 
	*b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_dtfsm_(char *transr, char *side, char *uplo, char *trans, 
	 char *diag, integer *m, integer *n, doublereal *alpha, doublereal *a, 
	 doublereal *b, integer *ldb);

/* Subroutine */ int _starpu_dtftri_(char *transr, char *uplo, char *diag, integer *n, 
	 doublereal *a, integer *info);

/* Subroutine */ int _starpu_dtfttp_(char *transr, char *uplo, integer *n, doublereal 
	*arf, doublereal *ap, integer *info);

/* Subroutine */ int _starpu_dtfttr_(char *transr, char *uplo, integer *n, doublereal 
	*arf, doublereal *a, integer *lda, integer *info);

/* Subroutine */ int _starpu_dtgevc_(char *side, char *howmny, logical *select, 
	integer *n, doublereal *s, integer *lds, doublereal *p, integer *ldp, 
	doublereal *vl, integer *ldvl, doublereal *vr, integer *ldvr, integer 
	*mm, integer *m, doublereal *work, integer *info);

/* Subroutine */ int _starpu_dtgex2_(logical *wantq, logical *wantz, integer *n, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *
	q, integer *ldq, doublereal *z__, integer *ldz, integer *j1, integer *
	n1, integer *n2, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dtgexc_(logical *wantq, logical *wantz, integer *n, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *
	q, integer *ldq, doublereal *z__, integer *ldz, integer *ifst, 
	integer *ilst, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_dtgsen_(integer *ijob, logical *wantq, logical *wantz, 
	logical *select, integer *n, doublereal *a, integer *lda, doublereal *
	b, integer *ldb, doublereal *alphar, doublereal *alphai, doublereal *
	beta, doublereal *q, integer *ldq, doublereal *z__, integer *ldz, 
	integer *m, doublereal *pl, doublereal *pr, doublereal *dif, 
	doublereal *work, integer *lwork, integer *iwork, integer *liwork, 
	integer *info);

/* Subroutine */ int _starpu_dtgsja_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *p, integer *n, integer *k, integer *l, doublereal *a, 
	integer *lda, doublereal *b, integer *ldb, doublereal *tola, 
	doublereal *tolb, doublereal *alpha, doublereal *beta, doublereal *u, 
	integer *ldu, doublereal *v, integer *ldv, doublereal *q, integer *
	ldq, doublereal *work, integer *ncycle, integer *info);

/* Subroutine */ int _starpu_dtgsna_(char *job, char *howmny, logical *select, 
	integer *n, doublereal *a, integer *lda, doublereal *b, integer *ldb, 
	doublereal *vl, integer *ldvl, doublereal *vr, integer *ldvr, 
	doublereal *s, doublereal *dif, integer *mm, integer *m, doublereal *
	work, integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dtgsy2_(char *trans, integer *ijob, integer *m, integer *
	n, doublereal *a, integer *lda, doublereal *b, integer *ldb, 
	doublereal *c__, integer *ldc, doublereal *d__, integer *ldd, 
	doublereal *e, integer *lde, doublereal *f, integer *ldf, doublereal *
	scale, doublereal *rdsum, doublereal *rdscal, integer *iwork, integer 
	*pq, integer *info);

/* Subroutine */ int _starpu_dtgsyl_(char *trans, integer *ijob, integer *m, integer *
	n, doublereal *a, integer *lda, doublereal *b, integer *ldb, 
	doublereal *c__, integer *ldc, doublereal *d__, integer *ldd, 
	doublereal *e, integer *lde, doublereal *f, integer *ldf, doublereal *
	scale, doublereal *dif, doublereal *work, integer *lwork, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_dtpcon_(char *norm, char *uplo, char *diag, integer *n, 
	doublereal *ap, doublereal *rcond, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_dtprfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, doublereal *ap, doublereal *b, integer *ldb, 
	doublereal *x, integer *ldx, doublereal *ferr, doublereal *berr, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dtptri_(char *uplo, char *diag, integer *n, doublereal *
	ap, integer *info);

/* Subroutine */ int _starpu_dtptrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, doublereal *ap, doublereal *b, integer *ldb, integer *
	info);

/* Subroutine */ int _starpu_dtpttf_(char *transr, char *uplo, integer *n, doublereal 
	*ap, doublereal *arf, integer *info);

/* Subroutine */ int _starpu_dtpttr_(char *uplo, integer *n, doublereal *ap, 
	doublereal *a, integer *lda, integer *info);

/* Subroutine */ int _starpu_dtrcon_(char *norm, char *uplo, char *diag, integer *n, 
	doublereal *a, integer *lda, doublereal *rcond, doublereal *work, 
	integer *iwork, integer *info);

/* Subroutine */ int _starpu_dtrevc_(char *side, char *howmny, logical *select, 
	integer *n, doublereal *t, integer *ldt, doublereal *vl, integer *
	ldvl, doublereal *vr, integer *ldvr, integer *mm, integer *m, 
	doublereal *work, integer *info);

/* Subroutine */ int _starpu_dtrexc_(char *compq, integer *n, doublereal *t, integer *
	ldt, doublereal *q, integer *ldq, integer *ifst, integer *ilst, 
	doublereal *work, integer *info);

/* Subroutine */ int _starpu_dtrrfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, doublereal *a, integer *lda, doublereal *b, integer *
	ldb, doublereal *x, integer *ldx, doublereal *ferr, doublereal *berr, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_dtrsen_(char *job, char *compq, logical *select, integer 
	*n, doublereal *t, integer *ldt, doublereal *q, integer *ldq, 
	doublereal *wr, doublereal *wi, integer *m, doublereal *s, doublereal 
	*sep, doublereal *work, integer *lwork, integer *iwork, integer *
	liwork, integer *info);

/* Subroutine */ int _starpu_dtrsna_(char *job, char *howmny, logical *select, 
	integer *n, doublereal *t, integer *ldt, doublereal *vl, integer *
	ldvl, doublereal *vr, integer *ldvr, doublereal *s, doublereal *sep, 
	integer *mm, integer *m, doublereal *work, integer *ldwork, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_dtrsyl_(char *trana, char *tranb, integer *isgn, integer 
	*m, integer *n, doublereal *a, integer *lda, doublereal *b, integer *
	ldb, doublereal *c__, integer *ldc, doublereal *scale, integer *info);

/* Subroutine */ int _starpu_dtrti2_(char *uplo, char *diag, integer *n, doublereal *
	a, integer *lda, integer *info);

/* Subroutine */ int _starpu_dtrtri_(char *uplo, char *diag, integer *n, doublereal *
	a, integer *lda, integer *info);

/* Subroutine */ int _starpu_dtrtrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, doublereal *a, integer *lda, doublereal *b, integer *
	ldb, integer *info);

/* Subroutine */ int _starpu_dtrttf_(char *transr, char *uplo, integer *n, doublereal 
	*a, integer *lda, doublereal *arf, integer *info);

/* Subroutine */ int _starpu_dtrttp_(char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *ap, integer *info);

/* Subroutine */ int _starpu_dtzrqf_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, integer *info);

/* Subroutine */ int _starpu_dtzrzf_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *lwork, integer *info);

doublereal _starpu_dzsum1_(integer *n, doublecomplex *cx, integer *incx);

integer _starpu_icmax1_(integer *n, complex *cx, integer *incx);

integer _starpu_ieeeck_(integer *ispec, real *zero, real *one);

integer _starpu_ilaclc_(integer *m, integer *n, complex *a, integer *lda);

integer _starpu_ilaclr_(integer *m, integer *n, complex *a, integer *lda);

integer _starpu_iladiag_(char *diag);

integer _starpu_iladlc_(integer *m, integer *n, doublereal *a, integer *lda);

integer _starpu_iladlr_(integer *m, integer *n, doublereal *a, integer *lda);

integer _starpu_ilaenv_(integer *ispec, char *name__, char *opts, integer *n1, 
	integer *n2, integer *n3, integer *n4);

integer _starpu_ilaprec_(char *prec);

integer _starpu_ilaslc_(integer *m, integer *n, real *a, integer *lda);

integer _starpu_ilaslr_(integer *m, integer *n, real *a, integer *lda);

integer _starpu_ilatrans_(char *trans);

integer _starpu_ilauplo_(char *uplo);

/* Subroutine */ int _starpu_ilaver_(integer *vers_major__, integer *vers_minor__, 
	integer *vers_patch__);

integer _starpu_ilazlc_(integer *m, integer *n, doublecomplex *a, integer *lda);

integer _starpu_ilazlr_(integer *m, integer *n, doublecomplex *a, integer *lda);

integer _starpu_iparmq_(integer *ispec, char *name__, char *opts, integer *n, integer 
	*ilo, integer *ihi, integer *lwork);

integer _starpu_izmax1_(integer *n, doublecomplex *cx, integer *incx);

logical _starpu_lsamen_(integer *n, char *ca, char *cb);

integer _starpu_smaxloc_(real *a, integer *dimm);

/* Subroutine */ int _starpu_sbdsdc_(char *uplo, char *compq, integer *n, real *d__, 
	real *e, real *u, integer *ldu, real *vt, integer *ldvt, real *q, 
	integer *iq, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sbdsqr_(char *uplo, integer *n, integer *ncvt, integer *
	nru, integer *ncc, real *d__, real *e, real *vt, integer *ldvt, real *
	u, integer *ldu, real *c__, integer *ldc, real *work, integer *info);

doublereal _starpu_scsum1_(integer *n, complex *cx, integer *incx);

/* Subroutine */ int _starpu_sdisna_(char *job, integer *m, integer *n, real *d__, 
	real *sep, integer *info);

/* Subroutine */ int _starpu_sgbbrd_(char *vect, integer *m, integer *n, integer *ncc, 
	 integer *kl, integer *ku, real *ab, integer *ldab, real *d__, real *
	e, real *q, integer *ldq, real *pt, integer *ldpt, real *c__, integer 
	*ldc, real *work, integer *info);

/* Subroutine */ int _starpu_sgbcon_(char *norm, integer *n, integer *kl, integer *ku, 
	 real *ab, integer *ldab, integer *ipiv, real *anorm, real *rcond, 
	real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgbequ_(integer *m, integer *n, integer *kl, integer *ku, 
	 real *ab, integer *ldab, real *r__, real *c__, real *rowcnd, real *
	colcnd, real *amax, integer *info);

/* Subroutine */ int _starpu_sgbequb_(integer *m, integer *n, integer *kl, integer *
	ku, real *ab, integer *ldab, real *r__, real *c__, real *rowcnd, real 
	*colcnd, real *amax, integer *info);

/* Subroutine */ int _starpu_sgbrfs_(char *trans, integer *n, integer *kl, integer *
	ku, integer *nrhs, real *ab, integer *ldab, real *afb, integer *ldafb, 
	 integer *ipiv, real *b, integer *ldb, real *x, integer *ldx, real *
	ferr, real *berr, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgbrfsx_(char *trans, char *equed, integer *n, integer *
	kl, integer *ku, integer *nrhs, real *ab, integer *ldab, real *afb, 
	integer *ldafb, integer *ipiv, real *r__, real *c__, real *b, integer 
	*ldb, real *x, integer *ldx, real *rcond, real *berr, integer *
	n_err_bnds__, real *err_bnds_norm__, real *err_bnds_comp__, integer *
	nparams, real *params, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgbsv_(integer *n, integer *kl, integer *ku, integer *
	nrhs, real *ab, integer *ldab, integer *ipiv, real *b, integer *ldb, 
	integer *info);

/* Subroutine */ int _starpu_sgbsvx_(char *fact, char *trans, integer *n, integer *kl, 
	 integer *ku, integer *nrhs, real *ab, integer *ldab, real *afb, 
	integer *ldafb, integer *ipiv, char *equed, real *r__, real *c__, 
	real *b, integer *ldb, real *x, integer *ldx, real *rcond, real *ferr, 
	 real *berr, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgbsvxx_(char *fact, char *trans, integer *n, integer *
	kl, integer *ku, integer *nrhs, real *ab, integer *ldab, real *afb, 
	integer *ldafb, integer *ipiv, char *equed, real *r__, real *c__, 
	real *b, integer *ldb, real *x, integer *ldx, real *rcond, real *
	rpvgrw, real *berr, integer *n_err_bnds__, real *err_bnds_norm__, 
	real *err_bnds_comp__, integer *nparams, real *params, real *work, 
	integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgbtf2_(integer *m, integer *n, integer *kl, integer *ku, 
	 real *ab, integer *ldab, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_sgbtrf_(integer *m, integer *n, integer *kl, integer *ku, 
	 real *ab, integer *ldab, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_sgbtrs_(char *trans, integer *n, integer *kl, integer *
	ku, integer *nrhs, real *ab, integer *ldab, integer *ipiv, real *b, 
	integer *ldb, integer *info);

/* Subroutine */ int _starpu_sgebak_(char *job, char *side, integer *n, integer *ilo, 
	integer *ihi, real *scale, integer *m, real *v, integer *ldv, integer 
	*info);

/* Subroutine */ int _starpu_sgebal_(char *job, integer *n, real *a, integer *lda, 
	integer *ilo, integer *ihi, real *scale, integer *info);

/* Subroutine */ int _starpu_sgebd2_(integer *m, integer *n, real *a, integer *lda, 
	real *d__, real *e, real *tauq, real *taup, real *work, integer *info);

/* Subroutine */ int _starpu_sgebrd_(integer *m, integer *n, real *a, integer *lda, 
	real *d__, real *e, real *tauq, real *taup, real *work, integer *
	lwork, integer *info);

/* Subroutine */ int _starpu_sgecon_(char *norm, integer *n, real *a, integer *lda, 
	real *anorm, real *rcond, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgeequ_(integer *m, integer *n, real *a, integer *lda, 
	real *r__, real *c__, real *rowcnd, real *colcnd, real *amax, integer 
	*info);

/* Subroutine */ int _starpu_sgeequb_(integer *m, integer *n, real *a, integer *lda, 
	real *r__, real *c__, real *rowcnd, real *colcnd, real *amax, integer 
	*info);

/* Subroutine */ int _starpu_sgees_(char *jobvs, char *sort, L_fp select, integer *n, 
	real *a, integer *lda, integer *sdim, real *wr, real *wi, real *vs, 
	integer *ldvs, real *work, integer *lwork, logical *bwork, integer *
	info);

/* Subroutine */ int _starpu_sgeesx_(char *jobvs, char *sort, L_fp select, char *
	sense, integer *n, real *a, integer *lda, integer *sdim, real *wr, 
	real *wi, real *vs, integer *ldvs, real *rconde, real *rcondv, real *
	work, integer *lwork, integer *iwork, integer *liwork, logical *bwork, 
	 integer *info);

/* Subroutine */ int _starpu_sgeev_(char *jobvl, char *jobvr, integer *n, real *a, 
	integer *lda, real *wr, real *wi, real *vl, integer *ldvl, real *vr, 
	integer *ldvr, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgeevx_(char *balanc, char *jobvl, char *jobvr, char *
	sense, integer *n, real *a, integer *lda, real *wr, real *wi, real *
	vl, integer *ldvl, real *vr, integer *ldvr, integer *ilo, integer *
	ihi, real *scale, real *abnrm, real *rconde, real *rcondv, real *work, 
	 integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgegs_(char *jobvsl, char *jobvsr, integer *n, real *a, 
	integer *lda, real *b, integer *ldb, real *alphar, real *alphai, real 
	*beta, real *vsl, integer *ldvsl, real *vsr, integer *ldvsr, real *
	work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgegv_(char *jobvl, char *jobvr, integer *n, real *a, 
	integer *lda, real *b, integer *ldb, real *alphar, real *alphai, real 
	*beta, real *vl, integer *ldvl, real *vr, integer *ldvr, real *work, 
	integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgehd2_(integer *n, integer *ilo, integer *ihi, real *a, 
	integer *lda, real *tau, real *work, integer *info);

/* Subroutine */ int _starpu_sgehrd_(integer *n, integer *ilo, integer *ihi, real *a, 
	integer *lda, real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgejsv_(char *joba, char *jobu, char *jobv, char *jobr, 
	char *jobt, char *jobp, integer *m, integer *n, real *a, integer *lda, 
	 real *sva, real *u, integer *ldu, real *v, integer *ldv, real *work, 
	integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgelq2_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *info);

/* Subroutine */ int _starpu_sgelqf_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgels_(char *trans, integer *m, integer *n, integer *
	nrhs, real *a, integer *lda, real *b, integer *ldb, real *work, 
	integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgelsd_(integer *m, integer *n, integer *nrhs, real *a, 
	integer *lda, real *b, integer *ldb, real *s, real *rcond, integer *
	rank, real *work, integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgelss_(integer *m, integer *n, integer *nrhs, real *a, 
	integer *lda, real *b, integer *ldb, real *s, real *rcond, integer *
	rank, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgelsx_(integer *m, integer *n, integer *nrhs, real *a, 
	integer *lda, real *b, integer *ldb, integer *jpvt, real *rcond, 
	integer *rank, real *work, integer *info);

/* Subroutine */ int _starpu_sgelsy_(integer *m, integer *n, integer *nrhs, real *a, 
	integer *lda, real *b, integer *ldb, integer *jpvt, real *rcond, 
	integer *rank, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgeql2_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *info);

/* Subroutine */ int _starpu_sgeqlf_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgeqp3_(integer *m, integer *n, real *a, integer *lda, 
	integer *jpvt, real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgeqpf_(integer *m, integer *n, real *a, integer *lda, 
	integer *jpvt, real *tau, real *work, integer *info);

/* Subroutine */ int _starpu_sgeqr2_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *info);

/* Subroutine */ int _starpu_sgeqrf_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgerfs_(char *trans, integer *n, integer *nrhs, real *a, 
	integer *lda, real *af, integer *ldaf, integer *ipiv, real *b, 
	integer *ldb, real *x, integer *ldx, real *ferr, real *berr, real *
	work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgerfsx_(char *trans, char *equed, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, integer *ipiv, 
	real *r__, real *c__, real *b, integer *ldb, real *x, integer *ldx, 
	real *rcond, real *berr, integer *n_err_bnds__, real *err_bnds_norm__, 
	 real *err_bnds_comp__, integer *nparams, real *params, real *work, 
	integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgerq2_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *info);

/* Subroutine */ int _starpu_sgerqf_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgesc2_(integer *n, real *a, integer *lda, real *rhs, 
	integer *ipiv, integer *jpiv, real *scale);

/* Subroutine */ int _starpu_sgesdd_(char *jobz, integer *m, integer *n, real *a, 
	integer *lda, real *s, real *u, integer *ldu, real *vt, integer *ldvt, 
	 real *work, integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgesv_(integer *n, integer *nrhs, real *a, integer *lda, 
	integer *ipiv, real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_sgesvd_(char *jobu, char *jobvt, integer *m, integer *n, 
	real *a, integer *lda, real *s, real *u, integer *ldu, real *vt, 
	integer *ldvt, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgesvj_(char *joba, char *jobu, char *jobv, integer *m, 
	integer *n, real *a, integer *lda, real *sva, integer *mv, real *v, 
	integer *ldv, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgesvx_(char *fact, char *trans, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, integer *ipiv, 
	char *equed, real *r__, real *c__, real *b, integer *ldb, real *x, 
	integer *ldx, real *rcond, real *ferr, real *berr, real *work, 
	integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgesvxx_(char *fact, char *trans, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, integer *ipiv, 
	char *equed, real *r__, real *c__, real *b, integer *ldb, real *x, 
	integer *ldx, real *rcond, real *rpvgrw, real *berr, integer *
	n_err_bnds__, real *err_bnds_norm__, real *err_bnds_comp__, integer *
	nparams, real *params, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgetc2_(integer *n, real *a, integer *lda, integer *ipiv, 
	 integer *jpiv, integer *info);

/* Subroutine */ int _starpu_sgetf2_(integer *m, integer *n, real *a, integer *lda, 
	integer *ipiv, integer *info);

/* Subroutine */ int _starpu_sgetrf_(integer *m, integer *n, real *a, integer *lda, 
	integer *ipiv, integer *info);

/* Subroutine */ int _starpu_sgetri_(integer *n, real *a, integer *lda, integer *ipiv, 
	 real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgetrs_(char *trans, integer *n, integer *nrhs, real *a, 
	integer *lda, integer *ipiv, real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_sggbak_(char *job, char *side, integer *n, integer *ilo, 
	integer *ihi, real *lscale, real *rscale, integer *m, real *v, 
	integer *ldv, integer *info);

/* Subroutine */ int _starpu_sggbal_(char *job, integer *n, real *a, integer *lda, 
	real *b, integer *ldb, integer *ilo, integer *ihi, real *lscale, real 
	*rscale, real *work, integer *info);

/* Subroutine */ int _starpu_sgges_(char *jobvsl, char *jobvsr, char *sort, L_fp 
	selctg, integer *n, real *a, integer *lda, real *b, integer *ldb, 
	integer *sdim, real *alphar, real *alphai, real *beta, real *vsl, 
	integer *ldvsl, real *vsr, integer *ldvsr, real *work, integer *lwork, 
	 logical *bwork, integer *info);

/* Subroutine */ int _starpu_sggesx_(char *jobvsl, char *jobvsr, char *sort, L_fp 
	selctg, char *sense, integer *n, real *a, integer *lda, real *b, 
	integer *ldb, integer *sdim, real *alphar, real *alphai, real *beta, 
	real *vsl, integer *ldvsl, real *vsr, integer *ldvsr, real *rconde, 
	real *rcondv, real *work, integer *lwork, integer *iwork, integer *
	liwork, logical *bwork, integer *info);

/* Subroutine */ int _starpu_sggev_(char *jobvl, char *jobvr, integer *n, real *a, 
	integer *lda, real *b, integer *ldb, real *alphar, real *alphai, real 
	*beta, real *vl, integer *ldvl, real *vr, integer *ldvr, real *work, 
	integer *lwork, integer *info);

/* Subroutine */ int _starpu_sggevx_(char *balanc, char *jobvl, char *jobvr, char *
	sense, integer *n, real *a, integer *lda, real *b, integer *ldb, real 
	*alphar, real *alphai, real *beta, real *vl, integer *ldvl, real *vr, 
	integer *ldvr, integer *ilo, integer *ihi, real *lscale, real *rscale, 
	 real *abnrm, real *bbnrm, real *rconde, real *rcondv, real *work, 
	integer *lwork, integer *iwork, logical *bwork, integer *info);

/* Subroutine */ int _starpu_sggglm_(integer *n, integer *m, integer *p, real *a, 
	integer *lda, real *b, integer *ldb, real *d__, real *x, real *y, 
	real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgghrd_(char *compq, char *compz, integer *n, integer *
	ilo, integer *ihi, real *a, integer *lda, real *b, integer *ldb, real 
	*q, integer *ldq, real *z__, integer *ldz, integer *info);

/* Subroutine */ int _starpu_sgglse_(integer *m, integer *n, integer *p, real *a, 
	integer *lda, real *b, integer *ldb, real *c__, real *d__, real *x, 
	real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sggqrf_(integer *n, integer *m, integer *p, real *a, 
	integer *lda, real *taua, real *b, integer *ldb, real *taub, real *
	work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sggrqf_(integer *m, integer *p, integer *n, real *a, 
	integer *lda, real *taua, real *b, integer *ldb, real *taub, real *
	work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sggsvd_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *n, integer *p, integer *k, integer *l, real *a, integer *lda, 
	 real *b, integer *ldb, real *alpha, real *beta, real *u, integer *
	ldu, real *v, integer *ldv, real *q, integer *ldq, real *work, 
	integer *iwork, integer *info);

/* Subroutine */ int _starpu_sggsvp_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *p, integer *n, real *a, integer *lda, real *b, integer *ldb, 
	real *tola, real *tolb, integer *k, integer *l, real *u, integer *ldu, 
	 real *v, integer *ldv, real *q, integer *ldq, integer *iwork, real *
	tau, real *work, integer *info);

/* Subroutine */ int _starpu_sgsvj0_(char *jobv, integer *m, integer *n, real *a, 
	integer *lda, real *d__, real *sva, integer *mv, real *v, integer *
	ldv, real *eps, real *sfmin, real *tol, integer *nsweep, real *work, 
	integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgsvj1_(char *jobv, integer *m, integer *n, integer *n1, 
	real *a, integer *lda, real *d__, real *sva, integer *mv, real *v, 
	integer *ldv, real *eps, real *sfmin, real *tol, integer *nsweep, 
	real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sgtcon_(char *norm, integer *n, real *dl, real *d__, 
	real *du, real *du2, integer *ipiv, real *anorm, real *rcond, real *
	work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgtrfs_(char *trans, integer *n, integer *nrhs, real *dl, 
	 real *d__, real *du, real *dlf, real *df, real *duf, real *du2, 
	integer *ipiv, real *b, integer *ldb, real *x, integer *ldx, real *
	ferr, real *berr, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sgtsv_(integer *n, integer *nrhs, real *dl, real *d__, 
	real *du, real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_sgtsvx_(char *fact, char *trans, integer *n, integer *
	nrhs, real *dl, real *d__, real *du, real *dlf, real *df, real *duf, 
	real *du2, integer *ipiv, real *b, integer *ldb, real *x, integer *
	ldx, real *rcond, real *ferr, real *berr, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_sgttrf_(integer *n, real *dl, real *d__, real *du, real *
	du2, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_sgttrs_(char *trans, integer *n, integer *nrhs, real *dl, 
	 real *d__, real *du, real *du2, integer *ipiv, real *b, integer *ldb, 
	 integer *info);

/* Subroutine */ int _starpu_sgtts2_(integer *itrans, integer *n, integer *nrhs, real 
	*dl, real *d__, real *du, real *du2, integer *ipiv, real *b, integer *
	ldb);

/* Subroutine */ int _starpu_shgeqz_(char *job, char *compq, char *compz, integer *n, 
	integer *ilo, integer *ihi, real *h__, integer *ldh, real *t, integer 
	*ldt, real *alphar, real *alphai, real *beta, real *q, integer *ldq, 
	real *z__, integer *ldz, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_shsein_(char *side, char *eigsrc, char *initv, logical *
	select, integer *n, real *h__, integer *ldh, real *wr, real *wi, real 
	*vl, integer *ldvl, real *vr, integer *ldvr, integer *mm, integer *m, 
	real *work, integer *ifaill, integer *ifailr, integer *info);

/* Subroutine */ int _starpu_shseqr_(char *job, char *compz, integer *n, integer *ilo, 
	 integer *ihi, real *h__, integer *ldh, real *wr, real *wi, real *z__, 
	 integer *ldz, real *work, integer *lwork, integer *info);

logical _starpu_sisnan_(real *sin__);

/* Subroutine */ int _starpu_sla_gbamv__(integer *trans, integer *m, integer *n, 
	integer *kl, integer *ku, real *alpha, real *ab, integer *ldab, real *
	x, integer *incx, real *beta, real *y, integer *incy);

doublereal _starpu_sla_gbrcond__(char *trans, integer *n, integer *kl, integer *ku, 
	real *ab, integer *ldab, real *afb, integer *ldafb, integer *ipiv, 
	integer *cmode, real *c__, integer *info, real *work, integer *iwork, 
	ftnlen trans_len);

/* Subroutine */ int _starpu_sla_gbrfsx_extended__(integer *prec_type__, integer *
	trans_type__, integer *n, integer *kl, integer *ku, integer *nrhs, 
	real *ab, integer *ldab, real *afb, integer *ldafb, integer *ipiv, 
	logical *colequ, real *c__, real *b, integer *ldb, real *y, integer *
	ldy, real *berr_out__, integer *n_norms__, real *errs_n__, real *
	errs_c__, real *res, real *ayb, real *dy, real *y_tail__, real *rcond,
	 integer *ithresh, real *rthresh, real *dz_ub__, logical *
	ignore_cwise__, integer *info);

doublereal _starpu_sla_gbrpvgrw__(integer *n, integer *kl, integer *ku, integer *
	ncols, real *ab, integer *ldab, real *afb, integer *ldafb);

/* Subroutine */ int _starpu_sla_geamv__(integer *trans, integer *m, integer *n, real 
	*alpha, real *a, integer *lda, real *x, integer *incx, real *beta, 
	real *y, integer *incy);

doublereal _starpu_sla_gercond__(char *trans, integer *n, real *a, integer *lda, real 
	*af, integer *ldaf, integer *ipiv, integer *cmode, real *c__, integer 
	*info, real *work, integer *iwork, ftnlen trans_len);

/* Subroutine */ int _starpu_sla_gerfsx_extended__(integer *prec_type__, integer *
	trans_type__, integer *n, integer *nrhs, real *a, integer *lda, real *
	af, integer *ldaf, integer *ipiv, logical *colequ, real *c__, real *b,
	 integer *ldb, real *y, integer *ldy, real *berr_out__, integer *
	n_norms__, real *errs_n__, real *errs_c__, real *res, real *ayb, real 
	*dy, real *y_tail__, real *rcond, integer *ithresh, real *rthresh, 
	real *dz_ub__, logical *ignore_cwise__, integer *info);

/* Subroutine */ int _starpu_sla_lin_berr__(integer *n, integer *nz, integer *nrhs, 
	real *res, real *ayb, real *berr);

doublereal _starpu_sla_porcond__(char *uplo, integer *n, real *a, integer *lda, real *
	af, integer *ldaf, integer *cmode, real *c__, integer *info, real *
	work, integer *iwork, ftnlen uplo_len);

/* Subroutine */ int _starpu_sla_porfsx_extended__(integer *prec_type__, char *uplo, 
	integer *n, integer *nrhs, real *a, integer *lda, real *af, integer *
	ldaf, logical *colequ, real *c__, real *b, integer *ldb, real *y, 
	integer *ldy, real *berr_out__, integer *n_norms__, real *errs_n__, 
	real *errs_c__, real *res, real *ayb, real *dy, real *y_tail__, real *
	rcond, integer *ithresh, real *rthresh, real *dz_ub__, logical *
	ignore_cwise__, integer *info, ftnlen uplo_len);

doublereal _starpu_sla_porpvgrw__(char *uplo, integer *ncols, real *a, integer *lda, 
	real *af, integer *ldaf, real *work, ftnlen uplo_len);

doublereal _starpu_sla_rpvgrw__(integer *n, integer *ncols, real *a, integer *lda, 
	real *af, integer *ldaf);

/* Subroutine */ int _starpu_sla_syamv__(integer *uplo, integer *n, real *alpha, real 
	*a, integer *lda, real *x, integer *incx, real *beta, real *y, 
	integer *incy);

doublereal _starpu_sla_syrcond__(char *uplo, integer *n, real *a, integer *lda, real *
	af, integer *ldaf, integer *ipiv, integer *cmode, real *c__, integer *
	info, real *work, integer *iwork, ftnlen uplo_len);

/* Subroutine */ int _starpu_sla_syrfsx_extended__(integer *prec_type__, char *uplo, 
	integer *n, integer *nrhs, real *a, integer *lda, real *af, integer *
	ldaf, integer *ipiv, logical *colequ, real *c__, real *b, integer *
	ldb, real *y, integer *ldy, real *berr_out__, integer *n_norms__, 
	real *errs_n__, real *errs_c__, real *res, real *ayb, real *dy, real *
	y_tail__, real *rcond, integer *ithresh, real *rthresh, real *dz_ub__,
	 logical *ignore_cwise__, integer *info, ftnlen uplo_len);

doublereal _starpu_sla_syrpvgrw__(char *uplo, integer *n, integer *info, real *a, 
	integer *lda, real *af, integer *ldaf, integer *ipiv, real *work, 
	ftnlen uplo_len);

/* Subroutine */ int _starpu_sla_wwaddw__(integer *n, real *x, real *y, real *w);

/* Subroutine */ int _starpu_slabad_(real *small, real *large);

/* Subroutine */ int _starpu_slabrd_(integer *m, integer *n, integer *nb, real *a, 
	integer *lda, real *d__, real *e, real *tauq, real *taup, real *x, 
	integer *ldx, real *y, integer *ldy);

/* Subroutine */ int _starpu_slacn2_(integer *n, real *v, real *x, integer *isgn, 
	real *est, integer *kase, integer *isave);

/* Subroutine */ int _starpu_slacon_(integer *n, real *v, real *x, integer *isgn, 
	real *est, integer *kase);

/* Subroutine */ int _starpu_slacpy_(char *uplo, integer *m, integer *n, real *a, 
	integer *lda, real *b, integer *ldb);

/* Subroutine */ int _starpu_sladiv_(real *a, real *b, real *c__, real *d__, real *p, 
	real *q);

/* Subroutine */ int _starpu_slae2_(real *a, real *b, real *c__, real *rt1, real *rt2);

/* Subroutine */ int _starpu_slaebz_(integer *ijob, integer *nitmax, integer *n, 
	integer *mmax, integer *minp, integer *nbmin, real *abstol, real *
	reltol, real *pivmin, real *d__, real *e, real *e2, integer *nval, 
	real *ab, real *c__, integer *mout, integer *nab, real *work, integer 
	*iwork, integer *info);

/* Subroutine */ int _starpu_slaed0_(integer *icompq, integer *qsiz, integer *n, real 
	*d__, real *e, real *q, integer *ldq, real *qstore, integer *ldqs, 
	real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_slaed1_(integer *n, real *d__, real *q, integer *ldq, 
	integer *indxq, real *rho, integer *cutpnt, real *work, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_slaed2_(integer *k, integer *n, integer *n1, real *d__, 
	real *q, integer *ldq, integer *indxq, real *rho, real *z__, real *
	dlamda, real *w, real *q2, integer *indx, integer *indxc, integer *
	indxp, integer *coltyp, integer *info);

/* Subroutine */ int _starpu_slaed3_(integer *k, integer *n, integer *n1, real *d__, 
	real *q, integer *ldq, real *rho, real *dlamda, real *q2, integer *
	indx, integer *ctot, real *w, real *s, integer *info);

/* Subroutine */ int _starpu_slaed4_(integer *n, integer *i__, real *d__, real *z__, 
	real *delta, real *rho, real *dlam, integer *info);

/* Subroutine */ int _starpu_slaed5_(integer *i__, real *d__, real *z__, real *delta, 
	real *rho, real *dlam);

/* Subroutine */ int _starpu_slaed6_(integer *kniter, logical *orgati, real *rho, 
	real *d__, real *z__, real *finit, real *tau, integer *info);

/* Subroutine */ int _starpu_slaed7_(integer *icompq, integer *n, integer *qsiz, 
	integer *tlvls, integer *curlvl, integer *curpbm, real *d__, real *q, 
	integer *ldq, integer *indxq, real *rho, integer *cutpnt, real *
	qstore, integer *qptr, integer *prmptr, integer *perm, integer *
	givptr, integer *givcol, real *givnum, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_slaed8_(integer *icompq, integer *k, integer *n, integer 
	*qsiz, real *d__, real *q, integer *ldq, integer *indxq, real *rho, 
	integer *cutpnt, real *z__, real *dlamda, real *q2, integer *ldq2, 
	real *w, integer *perm, integer *givptr, integer *givcol, real *
	givnum, integer *indxp, integer *indx, integer *info);

/* Subroutine */ int _starpu_slaed9_(integer *k, integer *kstart, integer *kstop, 
	integer *n, real *d__, real *q, integer *ldq, real *rho, real *dlamda, 
	 real *w, real *s, integer *lds, integer *info);

/* Subroutine */ int _starpu_slaeda_(integer *n, integer *tlvls, integer *curlvl, 
	integer *curpbm, integer *prmptr, integer *perm, integer *givptr, 
	integer *givcol, real *givnum, real *q, integer *qptr, real *z__, 
	real *ztemp, integer *info);

/* Subroutine */ int _starpu_slaein_(logical *rightv, logical *noinit, integer *n, 
	real *h__, integer *ldh, real *wr, real *wi, real *vr, real *vi, real 
	*b, integer *ldb, real *work, real *eps3, real *smlnum, real *bignum, 
	integer *info);

/* Subroutine */ int _starpu_slaev2_(real *a, real *b, real *c__, real *rt1, real *
	rt2, real *cs1, real *sn1);

/* Subroutine */ int _starpu_slaexc_(logical *wantq, integer *n, real *t, integer *
	ldt, real *q, integer *ldq, integer *j1, integer *n1, integer *n2, 
	real *work, integer *info);

/* Subroutine */ int _starpu_slag2_(real *a, integer *lda, real *b, integer *ldb, 
	real *safmin, real *scale1, real *scale2, real *wr1, real *wr2, real *
	wi);

/* Subroutine */ int _starpu_slag2d_(integer *m, integer *n, real *sa, integer *ldsa, 
	doublereal *a, integer *lda, integer *info);

/* Subroutine */ int _starpu_slags2_(logical *upper, real *a1, real *a2, real *a3, 
	real *b1, real *b2, real *b3, real *csu, real *snu, real *csv, real *
	snv, real *csq, real *snq);

/* Subroutine */ int _starpu_slagtf_(integer *n, real *a, real *lambda, real *b, real 
	*c__, real *tol, real *d__, integer *in, integer *info);

/* Subroutine */ int _starpu_slagtm_(char *trans, integer *n, integer *nrhs, real *
	alpha, real *dl, real *d__, real *du, real *x, integer *ldx, real *
	beta, real *b, integer *ldb);

/* Subroutine */ int _starpu_slagts_(integer *job, integer *n, real *a, real *b, real 
	*c__, real *d__, integer *in, real *y, real *tol, integer *info);

/* Subroutine */ int _starpu_slagv2_(real *a, integer *lda, real *b, integer *ldb, 
	real *alphar, real *alphai, real *beta, real *csl, real *snl, real *
	csr, real *snr);

/* Subroutine */ int _starpu_slahqr_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, real *h__, integer *ldh, real *wr, real *
	wi, integer *iloz, integer *ihiz, real *z__, integer *ldz, integer *
	info);

/* Subroutine */ int _starpu_slahr2_(integer *n, integer *k, integer *nb, real *a, 
	integer *lda, real *tau, real *t, integer *ldt, real *y, integer *ldy);

/* Subroutine */ int _starpu_slahrd_(integer *n, integer *k, integer *nb, real *a, 
	integer *lda, real *tau, real *t, integer *ldt, real *y, integer *ldy);

/* Subroutine */ int _starpu_slaic1_(integer *job, integer *j, real *x, real *sest, 
	real *w, real *gamma, real *sestpr, real *s, real *c__);

logical _starpu_slaisnan_(real *sin1, real *sin2);

/* Subroutine */ int _starpu_slaln2_(logical *ltrans, integer *na, integer *nw, real *
	smin, real *ca, real *a, integer *lda, real *d1, real *d2, real *b, 
	integer *ldb, real *wr, real *wi, real *x, integer *ldx, real *scale, 
	real *xnorm, integer *info);

/* Subroutine */ int _starpu_slals0_(integer *icompq, integer *nl, integer *nr, 
	integer *sqre, integer *nrhs, real *b, integer *ldb, real *bx, 
	integer *ldbx, integer *perm, integer *givptr, integer *givcol, 
	integer *ldgcol, real *givnum, integer *ldgnum, real *poles, real *
	difl, real *difr, real *z__, integer *k, real *c__, real *s, real *
	work, integer *info);

/* Subroutine */ int _starpu_slalsa_(integer *icompq, integer *smlsiz, integer *n, 
	integer *nrhs, real *b, integer *ldb, real *bx, integer *ldbx, real *
	u, integer *ldu, real *vt, integer *k, real *difl, real *difr, real *
	z__, real *poles, integer *givptr, integer *givcol, integer *ldgcol, 
	integer *perm, real *givnum, real *c__, real *s, real *work, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_slalsd_(char *uplo, integer *smlsiz, integer *n, integer 
	*nrhs, real *d__, real *e, real *b, integer *ldb, real *rcond, 
	integer *rank, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_slamrg_(integer *n1, integer *n2, real *a, integer *
	strd1, integer *strd2, integer *index);

integer _starpu_slaneg_(integer *n, real *d__, real *lld, real *sigma, real *pivmin, 
	integer *r__);

doublereal _starpu_slangb_(char *norm, integer *n, integer *kl, integer *ku, real *ab, 
	 integer *ldab, real *work);

doublereal _starpu_slange_(char *norm, integer *m, integer *n, real *a, integer *lda, 
	real *work);

doublereal _starpu_slangt_(char *norm, integer *n, real *dl, real *d__, real *du);

doublereal _starpu_slanhs_(char *norm, integer *n, real *a, integer *lda, real *work);

doublereal _starpu_slansb_(char *norm, char *uplo, integer *n, integer *k, real *ab, 
	integer *ldab, real *work);

doublereal _starpu_slansf_(char *norm, char *transr, char *uplo, integer *n, real *a, 
	real *work);

doublereal _starpu_slansp_(char *norm, char *uplo, integer *n, real *ap, real *work);

doublereal _starpu_slanst_(char *norm, integer *n, real *d__, real *e);

doublereal _starpu_slansy_(char *norm, char *uplo, integer *n, real *a, integer *lda, 
	real *work);

doublereal _starpu_slantb_(char *norm, char *uplo, char *diag, integer *n, integer *k, 
	 real *ab, integer *ldab, real *work);

doublereal _starpu_slantp_(char *norm, char *uplo, char *diag, integer *n, real *ap, 
	real *work);

doublereal _starpu_slantr_(char *norm, char *uplo, char *diag, integer *m, integer *n, 
	 real *a, integer *lda, real *work);

/* Subroutine */ int _starpu_slanv2_(real *a, real *b, real *c__, real *d__, real *
	rt1r, real *rt1i, real *rt2r, real *rt2i, real *cs, real *sn);

/* Subroutine */ int _starpu_slapll_(integer *n, real *x, integer *incx, real *y, 
	integer *incy, real *ssmin);

/* Subroutine */ int _starpu_slapmt_(logical *forwrd, integer *m, integer *n, real *x, 
	 integer *ldx, integer *k);

doublereal _starpu_slapy2_(real *x, real *y);

doublereal _starpu_slapy3_(real *x, real *y, real *z__);

/* Subroutine */ int _starpu_slaqgb_(integer *m, integer *n, integer *kl, integer *ku, 
	 real *ab, integer *ldab, real *r__, real *c__, real *rowcnd, real *
	colcnd, real *amax, char *equed);

/* Subroutine */ int _starpu_slaqge_(integer *m, integer *n, real *a, integer *lda, 
	real *r__, real *c__, real *rowcnd, real *colcnd, real *amax, char *
	equed);

/* Subroutine */ int _starpu_slaqp2_(integer *m, integer *n, integer *offset, real *a, 
	 integer *lda, integer *jpvt, real *tau, real *vn1, real *vn2, real *
	work);

/* Subroutine */ int _starpu_slaqps_(integer *m, integer *n, integer *offset, integer 
	*nb, integer *kb, real *a, integer *lda, integer *jpvt, real *tau, 
	real *vn1, real *vn2, real *auxv, real *f, integer *ldf);

/* Subroutine */ int _starpu_slaqr0_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, real *h__, integer *ldh, real *wr, real *
	wi, integer *iloz, integer *ihiz, real *z__, integer *ldz, real *work, 
	 integer *lwork, integer *info);

/* Subroutine */ int _starpu_slaqr1_(integer *n, real *h__, integer *ldh, real *sr1, 
	real *si1, real *sr2, real *si2, real *v);

/* Subroutine */ int _starpu_slaqr2_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, real *h__, integer *ldh, 
	integer *iloz, integer *ihiz, real *z__, integer *ldz, integer *ns, 
	integer *nd, real *sr, real *si, real *v, integer *ldv, integer *nh, 
	real *t, integer *ldt, integer *nv, real *wv, integer *ldwv, real *
	work, integer *lwork);

/* Subroutine */ int _starpu_slaqr3_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, real *h__, integer *ldh, 
	integer *iloz, integer *ihiz, real *z__, integer *ldz, integer *ns, 
	integer *nd, real *sr, real *si, real *v, integer *ldv, integer *nh, 
	real *t, integer *ldt, integer *nv, real *wv, integer *ldwv, real *
	work, integer *lwork);

/* Subroutine */ int _starpu_slaqr4_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, real *h__, integer *ldh, real *wr, real *
	wi, integer *iloz, integer *ihiz, real *z__, integer *ldz, real *work, 
	 integer *lwork, integer *info);

/* Subroutine */ int _starpu_slaqr5_(logical *wantt, logical *wantz, integer *kacc22, 
	integer *n, integer *ktop, integer *kbot, integer *nshfts, real *sr, 
	real *si, real *h__, integer *ldh, integer *iloz, integer *ihiz, real 
	*z__, integer *ldz, real *v, integer *ldv, real *u, integer *ldu, 
	integer *nv, real *wv, integer *ldwv, integer *nh, real *wh, integer *
	ldwh);

/* Subroutine */ int _starpu_slaqsb_(char *uplo, integer *n, integer *kd, real *ab, 
	integer *ldab, real *s, real *scond, real *amax, char *equed);

/* Subroutine */ int _starpu_slaqsp_(char *uplo, integer *n, real *ap, real *s, real *
	scond, real *amax, char *equed);

/* Subroutine */ int _starpu_slaqsy_(char *uplo, integer *n, real *a, integer *lda, 
	real *s, real *scond, real *amax, char *equed);

/* Subroutine */ int _starpu_slaqtr_(logical *ltran, logical *lreal, integer *n, real 
	*t, integer *ldt, real *b, real *w, real *scale, real *x, real *work, 
	integer *info);

/* Subroutine */ int _starpu_slar1v_(integer *n, integer *b1, integer *bn, real *
	lambda, real *d__, real *l, real *ld, real *lld, real *pivmin, real *
	gaptol, real *z__, logical *wantnc, integer *negcnt, real *ztz, real *
	mingma, integer *r__, integer *isuppz, real *nrminv, real *resid, 
	real *rqcorr, real *work);

/* Subroutine */ int _starpu_slar2v_(integer *n, real *x, real *y, real *z__, integer 
	*incx, real *c__, real *s, integer *incc);

/* Subroutine */ int _starpu_slarf_(char *side, integer *m, integer *n, real *v, 
	integer *incv, real *tau, real *c__, integer *ldc, real *work);

/* Subroutine */ int _starpu_slarfb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, real *v, integer *ldv, 
	real *t, integer *ldt, real *c__, integer *ldc, real *work, integer *
	ldwork);

/* Subroutine */ int _starpu_slarfg_(integer *n, real *alpha, real *x, integer *incx, 
	real *tau);

/* Subroutine */ int _starpu_slarfp_(integer *n, real *alpha, real *x, integer *incx, 
	real *tau);

/* Subroutine */ int _starpu_slarft_(char *direct, char *storev, integer *n, integer *
	k, real *v, integer *ldv, real *tau, real *t, integer *ldt);

/* Subroutine */ int _starpu_slarfx_(char *side, integer *m, integer *n, real *v, 
	real *tau, real *c__, integer *ldc, real *work);

/* Subroutine */ int _starpu_slargv_(integer *n, real *x, integer *incx, real *y, 
	integer *incy, real *c__, integer *incc);

/* Subroutine */ int _starpu_slarnv_(integer *idist, integer *iseed, integer *n, real 
	*x);

/* Subroutine */ int _starpu_slarra_(integer *n, real *d__, real *e, real *e2, real *
	spltol, real *tnrm, integer *nsplit, integer *isplit, integer *info);

/* Subroutine */ int _starpu_slarrb_(integer *n, real *d__, real *lld, integer *
	ifirst, integer *ilast, real *rtol1, real *rtol2, integer *offset, 
	real *w, real *wgap, real *werr, real *work, integer *iwork, real *
	pivmin, real *spdiam, integer *twist, integer *info);

/* Subroutine */ int _starpu_slarrc_(char *jobt, integer *n, real *vl, real *vu, real 
	*d__, real *e, real *pivmin, integer *eigcnt, integer *lcnt, integer *
	rcnt, integer *info);

/* Subroutine */ int _starpu_slarrd_(char *range, char *order, integer *n, real *vl, 
	real *vu, integer *il, integer *iu, real *gers, real *reltol, real *
	d__, real *e, real *e2, real *pivmin, integer *nsplit, integer *
	isplit, integer *m, real *w, real *werr, real *wl, real *wu, integer *
	iblock, integer *indexw, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_slarre_(char *range, integer *n, real *vl, real *vu, 
	integer *il, integer *iu, real *d__, real *e, real *e2, real *rtol1, 
	real *rtol2, real *spltol, integer *nsplit, integer *isplit, integer *
	m, real *w, real *werr, real *wgap, integer *iblock, integer *indexw, 
	real *gers, real *pivmin, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_slarrf_(integer *n, real *d__, real *l, real *ld, 
	integer *clstrt, integer *clend, real *w, real *wgap, real *werr, 
	real *spdiam, real *clgapl, real *clgapr, real *pivmin, real *sigma, 
	real *dplus, real *lplus, real *work, integer *info);

/* Subroutine */ int _starpu_slarrj_(integer *n, real *d__, real *e2, integer *ifirst, 
	 integer *ilast, real *rtol, integer *offset, real *w, real *werr, 
	real *work, integer *iwork, real *pivmin, real *spdiam, integer *info);

/* Subroutine */ int _starpu_slarrk_(integer *n, integer *iw, real *gl, real *gu, 
	real *d__, real *e2, real *pivmin, real *reltol, real *w, real *werr, 
	integer *info);

/* Subroutine */ int _starpu_slarrr_(integer *n, real *d__, real *e, integer *info);

/* Subroutine */ int _starpu_slarrv_(integer *n, real *vl, real *vu, real *d__, real *
	l, real *pivmin, integer *isplit, integer *m, integer *dol, integer *
	dou, real *minrgp, real *rtol1, real *rtol2, real *w, real *werr, 
	real *wgap, integer *iblock, integer *indexw, real *gers, real *z__, 
	integer *ldz, integer *isuppz, real *work, integer *iwork, integer *
	info);

/* Subroutine */ int _starpu_slarscl2_(integer *m, integer *n, real *d__, real *x, 
	integer *ldx);

/* Subroutine */ int _starpu_slartg_(real *f, real *g, real *cs, real *sn, real *r__);

/* Subroutine */ int _starpu_slartv_(integer *n, real *x, integer *incx, real *y, 
	integer *incy, real *c__, real *s, integer *incc);

/* Subroutine */ int _starpu_slaruv_(integer *iseed, integer *n, real *x);

/* Subroutine */ int _starpu_slarz_(char *side, integer *m, integer *n, integer *l, 
	real *v, integer *incv, real *tau, real *c__, integer *ldc, real *
	work);

/* Subroutine */ int _starpu_slarzb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, integer *l, real *v, 
	integer *ldv, real *t, integer *ldt, real *c__, integer *ldc, real *
	work, integer *ldwork);

/* Subroutine */ int _starpu_slarzt_(char *direct, char *storev, integer *n, integer *
	k, real *v, integer *ldv, real *tau, real *t, integer *ldt);

/* Subroutine */ int _starpu_slas2_(real *f, real *g, real *h__, real *ssmin, real *
	ssmax);

/* Subroutine */ int _starpu_slascl_(char *type__, integer *kl, integer *ku, real *
	cfrom, real *cto, integer *m, integer *n, real *a, integer *lda, 
	integer *info);

/* Subroutine */ int _starpu_slascl2_(integer *m, integer *n, real *d__, real *x, 
	integer *ldx);

/* Subroutine */ int _starpu_slasd0_(integer *n, integer *sqre, real *d__, real *e, 
	real *u, integer *ldu, real *vt, integer *ldvt, integer *smlsiz, 
	integer *iwork, real *work, integer *info);

/* Subroutine */ int _starpu_slasd1_(integer *nl, integer *nr, integer *sqre, real *
	d__, real *alpha, real *beta, real *u, integer *ldu, real *vt, 
	integer *ldvt, integer *idxq, integer *iwork, real *work, integer *
	info);

/* Subroutine */ int _starpu_slasd2_(integer *nl, integer *nr, integer *sqre, integer 
	*k, real *d__, real *z__, real *alpha, real *beta, real *u, integer *
	ldu, real *vt, integer *ldvt, real *dsigma, real *u2, integer *ldu2, 
	real *vt2, integer *ldvt2, integer *idxp, integer *idx, integer *idxc, 
	 integer *idxq, integer *coltyp, integer *info);

/* Subroutine */ int _starpu_slasd3_(integer *nl, integer *nr, integer *sqre, integer 
	*k, real *d__, real *q, integer *ldq, real *dsigma, real *u, integer *
	ldu, real *u2, integer *ldu2, real *vt, integer *ldvt, real *vt2, 
	integer *ldvt2, integer *idxc, integer *ctot, real *z__, integer *
	info);

/* Subroutine */ int _starpu_slasd4_(integer *n, integer *i__, real *d__, real *z__, 
	real *delta, real *rho, real *sigma, real *work, integer *info);

/* Subroutine */ int _starpu_slasd5_(integer *i__, real *d__, real *z__, real *delta, 
	real *rho, real *dsigma, real *work);

/* Subroutine */ int _starpu_slasd6_(integer *icompq, integer *nl, integer *nr, 
	integer *sqre, real *d__, real *vf, real *vl, real *alpha, real *beta, 
	 integer *idxq, integer *perm, integer *givptr, integer *givcol, 
	integer *ldgcol, real *givnum, integer *ldgnum, real *poles, real *
	difl, real *difr, real *z__, integer *k, real *c__, real *s, real *
	work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_slasd7_(integer *icompq, integer *nl, integer *nr, 
	integer *sqre, integer *k, real *d__, real *z__, real *zw, real *vf, 
	real *vfw, real *vl, real *vlw, real *alpha, real *beta, real *dsigma, 
	 integer *idx, integer *idxp, integer *idxq, integer *perm, integer *
	givptr, integer *givcol, integer *ldgcol, real *givnum, integer *
	ldgnum, real *c__, real *s, integer *info);

/* Subroutine */ int _starpu_slasd8_(integer *icompq, integer *k, real *d__, real *
	z__, real *vf, real *vl, real *difl, real *difr, integer *lddifr, 
	real *dsigma, real *work, integer *info);

/* Subroutine */ int _starpu_slasda_(integer *icompq, integer *smlsiz, integer *n, 
	integer *sqre, real *d__, real *e, real *u, integer *ldu, real *vt, 
	integer *k, real *difl, real *difr, real *z__, real *poles, integer *
	givptr, integer *givcol, integer *ldgcol, integer *perm, real *givnum, 
	 real *c__, real *s, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_slasdq_(char *uplo, integer *sqre, integer *n, integer *
	ncvt, integer *nru, integer *ncc, real *d__, real *e, real *vt, 
	integer *ldvt, real *u, integer *ldu, real *c__, integer *ldc, real *
	work, integer *info);

/* Subroutine */ int _starpu_slasdt_(integer *n, integer *lvl, integer *nd, integer *
	inode, integer *ndiml, integer *ndimr, integer *msub);

/* Subroutine */ int _starpu_slaset_(char *uplo, integer *m, integer *n, real *alpha, 
	real *beta, real *a, integer *lda);

/* Subroutine */ int _starpu_slasq1_(integer *n, real *d__, real *e, real *work, 
	integer *info);

/* Subroutine */ int _starpu_slasq2_(integer *n, real *z__, integer *info);

/* Subroutine */ int _starpu_slasq3_(integer *i0, integer *n0, real *z__, integer *pp, 
	 real *dmin__, real *sigma, real *desig, real *qmax, integer *nfail, 
	integer *iter, integer *ndiv, logical *ieee, integer *ttype, real *
	dmin1, real *dmin2, real *dn, real *dn1, real *dn2, real *g, real *
	tau);

/* Subroutine */ int _starpu_slasq4_(integer *i0, integer *n0, real *z__, integer *pp, 
	 integer *n0in, real *dmin__, real *dmin1, real *dmin2, real *dn, 
	real *dn1, real *dn2, real *tau, integer *ttype, real *g);

/* Subroutine */ int _starpu_slasq5_(integer *i0, integer *n0, real *z__, integer *pp, 
	 real *tau, real *dmin__, real *dmin1, real *dmin2, real *dn, real *
	dnm1, real *dnm2, logical *ieee);

/* Subroutine */ int _starpu_slasq6_(integer *i0, integer *n0, real *z__, integer *pp, 
	 real *dmin__, real *dmin1, real *dmin2, real *dn, real *dnm1, real *
	dnm2);

/* Subroutine */ int _starpu_slasr_(char *side, char *pivot, char *direct, integer *m, 
	 integer *n, real *c__, real *s, real *a, integer *lda);

/* Subroutine */ int _starpu_slasrt_(char *id, integer *n, real *d__, integer *info);

/* Subroutine */ int _starpu_slassq_(integer *n, real *x, integer *incx, real *scale, 
	real *sumsq);

/* Subroutine */ int _starpu_slasv2_(real *f, real *g, real *h__, real *ssmin, real *
	ssmax, real *snr, real *csr, real *snl, real *csl);

/* Subroutine */ int _starpu_slaswp_(integer *n, real *a, integer *lda, integer *k1, 
	integer *k2, integer *ipiv, integer *incx);

/* Subroutine */ int _starpu_slasy2_(logical *ltranl, logical *ltranr, integer *isgn, 
	integer *n1, integer *n2, real *tl, integer *ldtl, real *tr, integer *
	ldtr, real *b, integer *ldb, real *scale, real *x, integer *ldx, real 
	*xnorm, integer *info);

/* Subroutine */ int _starpu_slasyf_(char *uplo, integer *n, integer *nb, integer *kb, 
	 real *a, integer *lda, integer *ipiv, real *w, integer *ldw, integer 
	*info);

/* Subroutine */ int _starpu_slatbs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, integer *kd, real *ab, integer *ldab, real *x, 
	real *scale, real *cnorm, integer *info);

/* Subroutine */ int _starpu_slatdf_(integer *ijob, integer *n, real *z__, integer *
	ldz, real *rhs, real *rdsum, real *rdscal, integer *ipiv, integer *
	jpiv);

/* Subroutine */ int _starpu_slatps_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, real *ap, real *x, real *scale, real *cnorm, 
	integer *info);

/* Subroutine */ int _starpu_slatrd_(char *uplo, integer *n, integer *nb, real *a, 
	integer *lda, real *e, real *tau, real *w, integer *ldw);

/* Subroutine */ int _starpu_slatrs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, real *a, integer *lda, real *x, real *scale, real 
	*cnorm, integer *info);

/* Subroutine */ int _starpu_slatrz_(integer *m, integer *n, integer *l, real *a, 
	integer *lda, real *tau, real *work);

/* Subroutine */ int _starpu_slatzm_(char *side, integer *m, integer *n, real *v, 
	integer *incv, real *tau, real *c1, real *c2, integer *ldc, real *
	work);

/* Subroutine */ int _starpu_slauu2_(char *uplo, integer *n, real *a, integer *lda, 
	integer *info);

/* Subroutine */ int _starpu_slauum_(char *uplo, integer *n, real *a, integer *lda, 
	integer *info);

/* Subroutine */ int _starpu_sopgtr_(char *uplo, integer *n, real *ap, real *tau, 
	real *q, integer *ldq, real *work, integer *info);

/* Subroutine */ int _starpu_sopmtr_(char *side, char *uplo, char *trans, integer *m, 
	integer *n, real *ap, real *tau, real *c__, integer *ldc, real *work, 
	integer *info);

/* Subroutine */ int _starpu_sorg2l_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *info);

/* Subroutine */ int _starpu_sorg2r_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *info);

/* Subroutine */ int _starpu_sorgbr_(char *vect, integer *m, integer *n, integer *k, 
	real *a, integer *lda, real *tau, real *work, integer *lwork, integer 
	*info);

/* Subroutine */ int _starpu_sorghr_(integer *n, integer *ilo, integer *ihi, real *a, 
	integer *lda, real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sorgl2_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *info);

/* Subroutine */ int _starpu_sorglq_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sorgql_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sorgqr_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sorgr2_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *info);

/* Subroutine */ int _starpu_sorgrq_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sorgtr_(char *uplo, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sorm2l_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *info);

/* Subroutine */ int _starpu_sorm2r_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *info);

/* Subroutine */ int _starpu_sormbr_(char *vect, char *side, char *trans, integer *m, 
	integer *n, integer *k, real *a, integer *lda, real *tau, real *c__, 
	integer *ldc, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sormhr_(char *side, char *trans, integer *m, integer *n, 
	integer *ilo, integer *ihi, real *a, integer *lda, real *tau, real *
	c__, integer *ldc, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sorml2_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *info);

/* Subroutine */ int _starpu_sormlq_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sormql_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sormqr_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sormr2_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *info);

/* Subroutine */ int _starpu_sormr3_(char *side, char *trans, integer *m, integer *n, 
	integer *k, integer *l, real *a, integer *lda, real *tau, real *c__, 
	integer *ldc, real *work, integer *info);

/* Subroutine */ int _starpu_sormrq_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sormrz_(char *side, char *trans, integer *m, integer *n, 
	integer *k, integer *l, real *a, integer *lda, real *tau, real *c__, 
	integer *ldc, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_sormtr_(char *side, char *uplo, char *trans, integer *m, 
	integer *n, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_spbcon_(char *uplo, integer *n, integer *kd, real *ab, 
	integer *ldab, real *anorm, real *rcond, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_spbequ_(char *uplo, integer *n, integer *kd, real *ab, 
	integer *ldab, real *s, real *scond, real *amax, integer *info);

/* Subroutine */ int _starpu_spbrfs_(char *uplo, integer *n, integer *kd, integer *
	nrhs, real *ab, integer *ldab, real *afb, integer *ldafb, real *b, 
	integer *ldb, real *x, integer *ldx, real *ferr, real *berr, real *
	work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_spbstf_(char *uplo, integer *n, integer *kd, real *ab, 
	integer *ldab, integer *info);

/* Subroutine */ int _starpu_spbsv_(char *uplo, integer *n, integer *kd, integer *
	nrhs, real *ab, integer *ldab, real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_spbsvx_(char *fact, char *uplo, integer *n, integer *kd, 
	integer *nrhs, real *ab, integer *ldab, real *afb, integer *ldafb, 
	char *equed, real *s, real *b, integer *ldb, real *x, integer *ldx, 
	real *rcond, real *ferr, real *berr, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_spbtf2_(char *uplo, integer *n, integer *kd, real *ab, 
	integer *ldab, integer *info);

/* Subroutine */ int _starpu_spbtrf_(char *uplo, integer *n, integer *kd, real *ab, 
	integer *ldab, integer *info);

/* Subroutine */ int _starpu_spbtrs_(char *uplo, integer *n, integer *kd, integer *
	nrhs, real *ab, integer *ldab, real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_spftrf_(char *transr, char *uplo, integer *n, real *a, 
	integer *info);

/* Subroutine */ int _starpu_spftri_(char *transr, char *uplo, integer *n, real *a, 
	integer *info);

/* Subroutine */ int _starpu_spftrs_(char *transr, char *uplo, integer *n, integer *
	nrhs, real *a, real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_spocon_(char *uplo, integer *n, real *a, integer *lda, 
	real *anorm, real *rcond, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_spoequ_(integer *n, real *a, integer *lda, real *s, real 
	*scond, real *amax, integer *info);

/* Subroutine */ int _starpu_spoequb_(integer *n, real *a, integer *lda, real *s, 
	real *scond, real *amax, integer *info);

/* Subroutine */ int _starpu_sporfs_(char *uplo, integer *n, integer *nrhs, real *a, 
	integer *lda, real *af, integer *ldaf, real *b, integer *ldb, real *x, 
	 integer *ldx, real *ferr, real *berr, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_sporfsx_(char *uplo, char *equed, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, real *s, real *
	b, integer *ldb, real *x, integer *ldx, real *rcond, real *berr, 
	integer *n_err_bnds__, real *err_bnds_norm__, real *err_bnds_comp__, 
	integer *nparams, real *params, real *work, integer *iwork, integer *
	info);

/* Subroutine */ int _starpu_sposv_(char *uplo, integer *n, integer *nrhs, real *a, 
	integer *lda, real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_sposvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, char *equed, 
	real *s, real *b, integer *ldb, real *x, integer *ldx, real *rcond, 
	real *ferr, real *berr, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sposvxx_(char *fact, char *uplo, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, char *equed, 
	real *s, real *b, integer *ldb, real *x, integer *ldx, real *rcond, 
	real *rpvgrw, real *berr, integer *n_err_bnds__, real *
	err_bnds_norm__, real *err_bnds_comp__, integer *nparams, real *
	params, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_spotf2_(char *uplo, integer *n, real *a, integer *lda, 
	integer *info);

/* Subroutine */ int _starpu_spotrf_(char *uplo, integer *n, real *a, integer *lda, 
	integer *info);

/* Subroutine */ int _starpu_spotri_(char *uplo, integer *n, real *a, integer *lda, 
	integer *info);

/* Subroutine */ int _starpu_spotrs_(char *uplo, integer *n, integer *nrhs, real *a, 
	integer *lda, real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_sppcon_(char *uplo, integer *n, real *ap, real *anorm, 
	real *rcond, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sppequ_(char *uplo, integer *n, real *ap, real *s, real *
	scond, real *amax, integer *info);

/* Subroutine */ int _starpu_spprfs_(char *uplo, integer *n, integer *nrhs, real *ap, 
	real *afp, real *b, integer *ldb, real *x, integer *ldx, real *ferr, 
	real *berr, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sppsv_(char *uplo, integer *n, integer *nrhs, real *ap, 
	real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_sppsvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, real *ap, real *afp, char *equed, real *s, real *b, integer *
	ldb, real *x, integer *ldx, real *rcond, real *ferr, real *berr, real 
	*work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_spptrf_(char *uplo, integer *n, real *ap, integer *info);

/* Subroutine */ int _starpu_spptri_(char *uplo, integer *n, real *ap, integer *info);

/* Subroutine */ int _starpu_spptrs_(char *uplo, integer *n, integer *nrhs, real *ap, 
	real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_spstf2_(char *uplo, integer *n, real *a, integer *lda, 
	integer *piv, integer *rank, real *tol, real *work, integer *info);

/* Subroutine */ int _starpu_spstrf_(char *uplo, integer *n, real *a, integer *lda, 
	integer *piv, integer *rank, real *tol, real *work, integer *info);

/* Subroutine */ int _starpu_sptcon_(integer *n, real *d__, real *e, real *anorm, 
	real *rcond, real *work, integer *info);

/* Subroutine */ int _starpu_spteqr_(char *compz, integer *n, real *d__, real *e, 
	real *z__, integer *ldz, real *work, integer *info);

/* Subroutine */ int _starpu_sptrfs_(integer *n, integer *nrhs, real *d__, real *e, 
	real *df, real *ef, real *b, integer *ldb, real *x, integer *ldx, 
	real *ferr, real *berr, real *work, integer *info);

/* Subroutine */ int _starpu_sptsv_(integer *n, integer *nrhs, real *d__, real *e, 
	real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_sptsvx_(char *fact, integer *n, integer *nrhs, real *d__, 
	 real *e, real *df, real *ef, real *b, integer *ldb, real *x, integer 
	*ldx, real *rcond, real *ferr, real *berr, real *work, integer *info);

/* Subroutine */ int _starpu_spttrf_(integer *n, real *d__, real *e, integer *info);

/* Subroutine */ int _starpu_spttrs_(integer *n, integer *nrhs, real *d__, real *e, 
	real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_sptts2_(integer *n, integer *nrhs, real *d__, real *e, 
	real *b, integer *ldb);

/* Subroutine */ int _starpu_srscl_(integer *n, real *sa, real *sx, integer *incx);

/* Subroutine */ int _starpu_ssbev_(char *jobz, char *uplo, integer *n, integer *kd, 
	real *ab, integer *ldab, real *w, real *z__, integer *ldz, real *work, 
	 integer *info);

/* Subroutine */ int _starpu_ssbevd_(char *jobz, char *uplo, integer *n, integer *kd, 
	real *ab, integer *ldab, real *w, real *z__, integer *ldz, real *work, 
	 integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_ssbevx_(char *jobz, char *range, char *uplo, integer *n, 
	integer *kd, real *ab, integer *ldab, real *q, integer *ldq, real *vl, 
	 real *vu, integer *il, integer *iu, real *abstol, integer *m, real *
	w, real *z__, integer *ldz, real *work, integer *iwork, integer *
	ifail, integer *info);

/* Subroutine */ int _starpu_ssbgst_(char *vect, char *uplo, integer *n, integer *ka, 
	integer *kb, real *ab, integer *ldab, real *bb, integer *ldbb, real *
	x, integer *ldx, real *work, integer *info);

/* Subroutine */ int _starpu_ssbgv_(char *jobz, char *uplo, integer *n, integer *ka, 
	integer *kb, real *ab, integer *ldab, real *bb, integer *ldbb, real *
	w, real *z__, integer *ldz, real *work, integer *info);

/* Subroutine */ int _starpu_ssbgvd_(char *jobz, char *uplo, integer *n, integer *ka, 
	integer *kb, real *ab, integer *ldab, real *bb, integer *ldbb, real *
	w, real *z__, integer *ldz, real *work, integer *lwork, integer *
	iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_ssbgvx_(char *jobz, char *range, char *uplo, integer *n, 
	integer *ka, integer *kb, real *ab, integer *ldab, real *bb, integer *
	ldbb, real *q, integer *ldq, real *vl, real *vu, integer *il, integer 
	*iu, real *abstol, integer *m, real *w, real *z__, integer *ldz, real 
	*work, integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu_ssbtrd_(char *vect, char *uplo, integer *n, integer *kd, 
	real *ab, integer *ldab, real *d__, real *e, real *q, integer *ldq, 
	real *work, integer *info);

/* Subroutine */ int _starpu_ssfrk_(char *transr, char *uplo, char *trans, integer *n, 
	 integer *k, real *alpha, real *a, integer *lda, real *beta, real *
	c__);

/* Subroutine */ int _starpu_sspcon_(char *uplo, integer *n, real *ap, integer *ipiv, 
	real *anorm, real *rcond, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sspev_(char *jobz, char *uplo, integer *n, real *ap, 
	real *w, real *z__, integer *ldz, real *work, integer *info);

/* Subroutine */ int _starpu_sspevd_(char *jobz, char *uplo, integer *n, real *ap, 
	real *w, real *z__, integer *ldz, real *work, integer *lwork, integer 
	*iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_sspevx_(char *jobz, char *range, char *uplo, integer *n, 
	real *ap, real *vl, real *vu, integer *il, integer *iu, real *abstol, 
	integer *m, real *w, real *z__, integer *ldz, real *work, integer *
	iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu_sspgst_(integer *itype, char *uplo, integer *n, real *ap, 
	 real *bp, integer *info);

/* Subroutine */ int _starpu_sspgv_(integer *itype, char *jobz, char *uplo, integer *
	n, real *ap, real *bp, real *w, real *z__, integer *ldz, real *work, 
	integer *info);

/* Subroutine */ int _starpu_sspgvd_(integer *itype, char *jobz, char *uplo, integer *
	n, real *ap, real *bp, real *w, real *z__, integer *ldz, real *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_sspgvx_(integer *itype, char *jobz, char *range, char *
	uplo, integer *n, real *ap, real *bp, real *vl, real *vu, integer *il, 
	 integer *iu, real *abstol, integer *m, real *w, real *z__, integer *
	ldz, real *work, integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu_ssprfs_(char *uplo, integer *n, integer *nrhs, real *ap, 
	real *afp, integer *ipiv, real *b, integer *ldb, real *x, integer *
	ldx, real *ferr, real *berr, real *work, integer *iwork, integer *
	info);

/* Subroutine */ int _starpu_sspsv_(char *uplo, integer *n, integer *nrhs, real *ap, 
	integer *ipiv, real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_sspsvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, real *ap, real *afp, integer *ipiv, real *b, integer *ldb, real 
	*x, integer *ldx, real *rcond, real *ferr, real *berr, real *work, 
	integer *iwork, integer *info);

/* Subroutine */ int _starpu_ssptrd_(char *uplo, integer *n, real *ap, real *d__, 
	real *e, real *tau, integer *info);

/* Subroutine */ int _starpu_ssptrf_(char *uplo, integer *n, real *ap, integer *ipiv, 
	integer *info);

/* Subroutine */ int _starpu_ssptri_(char *uplo, integer *n, real *ap, integer *ipiv, 
	real *work, integer *info);

/* Subroutine */ int _starpu_ssptrs_(char *uplo, integer *n, integer *nrhs, real *ap, 
	integer *ipiv, real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_sstebz_(char *range, char *order, integer *n, real *vl, 
	real *vu, integer *il, integer *iu, real *abstol, real *d__, real *e, 
	integer *m, integer *nsplit, real *w, integer *iblock, integer *
	isplit, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_sstedc_(char *compz, integer *n, real *d__, real *e, 
	real *z__, integer *ldz, real *work, integer *lwork, integer *iwork, 
	integer *liwork, integer *info);

/* Subroutine */ int _starpu_sstegr_(char *jobz, char *range, integer *n, real *d__, 
	real *e, real *vl, real *vu, integer *il, integer *iu, real *abstol, 
	integer *m, real *w, real *z__, integer *ldz, integer *isuppz, real *
	work, integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_sstein_(integer *n, real *d__, real *e, integer *m, real 
	*w, integer *iblock, integer *isplit, real *z__, integer *ldz, real *
	work, integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu_sstemr_(char *jobz, char *range, integer *n, real *d__, 
	real *e, real *vl, real *vu, integer *il, integer *iu, integer *m, 
	real *w, real *z__, integer *ldz, integer *nzc, integer *isuppz, 
	logical *tryrac, real *work, integer *lwork, integer *iwork, integer *
	liwork, integer *info);

/* Subroutine */ int _starpu_ssteqr_(char *compz, integer *n, real *d__, real *e, 
	real *z__, integer *ldz, real *work, integer *info);

/* Subroutine */ int _starpu_ssterf_(integer *n, real *d__, real *e, integer *info);

/* Subroutine */ int _starpu_sstev_(char *jobz, integer *n, real *d__, real *e, real *
	z__, integer *ldz, real *work, integer *info);

/* Subroutine */ int _starpu_sstevd_(char *jobz, integer *n, real *d__, real *e, real 
	*z__, integer *ldz, real *work, integer *lwork, integer *iwork, 
	integer *liwork, integer *info);

/* Subroutine */ int _starpu_sstevr_(char *jobz, char *range, integer *n, real *d__, 
	real *e, real *vl, real *vu, integer *il, integer *iu, real *abstol, 
	integer *m, real *w, real *z__, integer *ldz, integer *isuppz, real *
	work, integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_sstevx_(char *jobz, char *range, integer *n, real *d__, 
	real *e, real *vl, real *vu, integer *il, integer *iu, real *abstol, 
	integer *m, real *w, real *z__, integer *ldz, real *work, integer *
	iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu_ssycon_(char *uplo, integer *n, real *a, integer *lda, 
	integer *ipiv, real *anorm, real *rcond, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_ssyequb_(char *uplo, integer *n, real *a, integer *lda, 
	real *s, real *scond, real *amax, real *work, integer *info);

/* Subroutine */ int _starpu_ssyev_(char *jobz, char *uplo, integer *n, real *a, 
	integer *lda, real *w, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_ssyevd_(char *jobz, char *uplo, integer *n, real *a, 
	integer *lda, real *w, real *work, integer *lwork, integer *iwork, 
	integer *liwork, integer *info);

/* Subroutine */ int _starpu_ssyevr_(char *jobz, char *range, char *uplo, integer *n, 
	real *a, integer *lda, real *vl, real *vu, integer *il, integer *iu, 
	real *abstol, integer *m, real *w, real *z__, integer *ldz, integer *
	isuppz, real *work, integer *lwork, integer *iwork, integer *liwork, 
	integer *info);

/* Subroutine */ int _starpu_ssyevx_(char *jobz, char *range, char *uplo, integer *n, 
	real *a, integer *lda, real *vl, real *vu, integer *il, integer *iu, 
	real *abstol, integer *m, real *w, real *z__, integer *ldz, real *
	work, integer *lwork, integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu_ssygs2_(integer *itype, char *uplo, integer *n, real *a, 
	integer *lda, real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_ssygst_(integer *itype, char *uplo, integer *n, real *a, 
	integer *lda, real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_ssygv_(integer *itype, char *jobz, char *uplo, integer *
	n, real *a, integer *lda, real *b, integer *ldb, real *w, real *work, 
	integer *lwork, integer *info);

/* Subroutine */ int _starpu_ssygvd_(integer *itype, char *jobz, char *uplo, integer *
	n, real *a, integer *lda, real *b, integer *ldb, real *w, real *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_ssygvx_(integer *itype, char *jobz, char *range, char *
	uplo, integer *n, real *a, integer *lda, real *b, integer *ldb, real *
	vl, real *vu, integer *il, integer *iu, real *abstol, integer *m, 
	real *w, real *z__, integer *ldz, real *work, integer *lwork, integer 
	*iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu_ssyrfs_(char *uplo, integer *n, integer *nrhs, real *a, 
	integer *lda, real *af, integer *ldaf, integer *ipiv, real *b, 
	integer *ldb, real *x, integer *ldx, real *ferr, real *berr, real *
	work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_ssyrfsx_(char *uplo, char *equed, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, integer *ipiv, 
	real *s, real *b, integer *ldb, real *x, integer *ldx, real *rcond, 
	real *berr, integer *n_err_bnds__, real *err_bnds_norm__, real *
	err_bnds_comp__, integer *nparams, real *params, real *work, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_ssysv_(char *uplo, integer *n, integer *nrhs, real *a, 
	integer *lda, integer *ipiv, real *b, integer *ldb, real *work, 
	integer *lwork, integer *info);

/* Subroutine */ int _starpu_ssysvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, integer *ipiv, 
	real *b, integer *ldb, real *x, integer *ldx, real *rcond, real *ferr, 
	 real *berr, real *work, integer *lwork, integer *iwork, integer *
	info);

/* Subroutine */ int _starpu_ssysvxx_(char *fact, char *uplo, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, integer *ipiv, 
	char *equed, real *s, real *b, integer *ldb, real *x, integer *ldx, 
	real *rcond, real *rpvgrw, real *berr, integer *n_err_bnds__, real *
	err_bnds_norm__, real *err_bnds_comp__, integer *nparams, real *
	params, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_ssytd2_(char *uplo, integer *n, real *a, integer *lda, 
	real *d__, real *e, real *tau, integer *info);

/* Subroutine */ int _starpu_ssytf2_(char *uplo, integer *n, real *a, integer *lda, 
	integer *ipiv, integer *info);

/* Subroutine */ int _starpu_ssytrd_(char *uplo, integer *n, real *a, integer *lda, 
	real *d__, real *e, real *tau, real *work, integer *lwork, integer *
	info);

/* Subroutine */ int _starpu_ssytrf_(char *uplo, integer *n, real *a, integer *lda, 
	integer *ipiv, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_ssytri_(char *uplo, integer *n, real *a, integer *lda, 
	integer *ipiv, real *work, integer *info);

/* Subroutine */ int _starpu_ssytrs_(char *uplo, integer *n, integer *nrhs, real *a, 
	integer *lda, integer *ipiv, real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_stbcon_(char *norm, char *uplo, char *diag, integer *n, 
	integer *kd, real *ab, integer *ldab, real *rcond, real *work, 
	integer *iwork, integer *info);

/* Subroutine */ int _starpu_stbrfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *kd, integer *nrhs, real *ab, integer *ldab, real *b, integer 
	*ldb, real *x, integer *ldx, real *ferr, real *berr, real *work, 
	integer *iwork, integer *info);

/* Subroutine */ int _starpu_stbtrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *kd, integer *nrhs, real *ab, integer *ldab, real *b, integer 
	*ldb, integer *info);

/* Subroutine */ int _starpu_stfsm_(char *transr, char *side, char *uplo, char *trans, 
	 char *diag, integer *m, integer *n, real *alpha, real *a, real *b, 
	integer *ldb);

/* Subroutine */ int _starpu_stftri_(char *transr, char *uplo, char *diag, integer *n, 
	 real *a, integer *info);

/* Subroutine */ int _starpu_stfttp_(char *transr, char *uplo, integer *n, real *arf, 
	real *ap, integer *info);

/* Subroutine */ int _starpu_stfttr_(char *transr, char *uplo, integer *n, real *arf, 
	real *a, integer *lda, integer *info);

/* Subroutine */ int _starpu_stgevc_(char *side, char *howmny, logical *select, 
	integer *n, real *s, integer *lds, real *p, integer *ldp, real *vl, 
	integer *ldvl, real *vr, integer *ldvr, integer *mm, integer *m, real 
	*work, integer *info);

/* Subroutine */ int _starpu_stgex2_(logical *wantq, logical *wantz, integer *n, real 
	*a, integer *lda, real *b, integer *ldb, real *q, integer *ldq, real *
	z__, integer *ldz, integer *j1, integer *n1, integer *n2, real *work, 
	integer *lwork, integer *info);

/* Subroutine */ int _starpu_stgexc_(logical *wantq, logical *wantz, integer *n, real 
	*a, integer *lda, real *b, integer *ldb, real *q, integer *ldq, real *
	z__, integer *ldz, integer *ifst, integer *ilst, real *work, integer *
	lwork, integer *info);

/* Subroutine */ int _starpu_stgsen_(integer *ijob, logical *wantq, logical *wantz, 
	logical *select, integer *n, real *a, integer *lda, real *b, integer *
	ldb, real *alphar, real *alphai, real *beta, real *q, integer *ldq, 
	real *z__, integer *ldz, integer *m, real *pl, real *pr, real *dif, 
	real *work, integer *lwork, integer *iwork, integer *liwork, integer *
	info);

/* Subroutine */ int _starpu_stgsja_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *p, integer *n, integer *k, integer *l, real *a, integer *lda, 
	 real *b, integer *ldb, real *tola, real *tolb, real *alpha, real *
	beta, real *u, integer *ldu, real *v, integer *ldv, real *q, integer *
	ldq, real *work, integer *ncycle, integer *info);

/* Subroutine */ int _starpu_stgsna_(char *job, char *howmny, logical *select, 
	integer *n, real *a, integer *lda, real *b, integer *ldb, real *vl, 
	integer *ldvl, real *vr, integer *ldvr, real *s, real *dif, integer *
	mm, integer *m, real *work, integer *lwork, integer *iwork, integer *
	info);

/* Subroutine */ int _starpu_stgsy2_(char *trans, integer *ijob, integer *m, integer *
	n, real *a, integer *lda, real *b, integer *ldb, real *c__, integer *
	ldc, real *d__, integer *ldd, real *e, integer *lde, real *f, integer 
	*ldf, real *scale, real *rdsum, real *rdscal, integer *iwork, integer 
	*pq, integer *info);

/* Subroutine */ int _starpu_stgsyl_(char *trans, integer *ijob, integer *m, integer *
	n, real *a, integer *lda, real *b, integer *ldb, real *c__, integer *
	ldc, real *d__, integer *ldd, real *e, integer *lde, real *f, integer 
	*ldf, real *scale, real *dif, real *work, integer *lwork, integer *
	iwork, integer *info);

/* Subroutine */ int _starpu_stpcon_(char *norm, char *uplo, char *diag, integer *n, 
	real *ap, real *rcond, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_stprfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, real *ap, real *b, integer *ldb, real *x, integer *ldx, 
	 real *ferr, real *berr, real *work, integer *iwork, integer *info);

/* Subroutine */ int _starpu_stptri_(char *uplo, char *diag, integer *n, real *ap, 
	integer *info);

/* Subroutine */ int _starpu_stptrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, real *ap, real *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_stpttf_(char *transr, char *uplo, integer *n, real *ap, 
	real *arf, integer *info);

/* Subroutine */ int _starpu_stpttr_(char *uplo, integer *n, real *ap, real *a, 
	integer *lda, integer *info);

/* Subroutine */ int _starpu_strcon_(char *norm, char *uplo, char *diag, integer *n, 
	real *a, integer *lda, real *rcond, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_strevc_(char *side, char *howmny, logical *select, 
	integer *n, real *t, integer *ldt, real *vl, integer *ldvl, real *vr, 
	integer *ldvr, integer *mm, integer *m, real *work, integer *info);

/* Subroutine */ int _starpu_strexc_(char *compq, integer *n, real *t, integer *ldt, 
	real *q, integer *ldq, integer *ifst, integer *ilst, real *work, 
	integer *info);

/* Subroutine */ int _starpu_strrfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, real *a, integer *lda, real *b, integer *ldb, real *x, 
	integer *ldx, real *ferr, real *berr, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int _starpu_strsen_(char *job, char *compq, logical *select, integer 
	*n, real *t, integer *ldt, real *q, integer *ldq, real *wr, real *wi, 
	integer *m, real *s, real *sep, real *work, integer *lwork, integer *
	iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_strsna_(char *job, char *howmny, logical *select, 
	integer *n, real *t, integer *ldt, real *vl, integer *ldvl, real *vr, 
	integer *ldvr, real *s, real *sep, integer *mm, integer *m, real *
	work, integer *ldwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_strsyl_(char *trana, char *tranb, integer *isgn, integer 
	*m, integer *n, real *a, integer *lda, real *b, integer *ldb, real *
	c__, integer *ldc, real *scale, integer *info);

/* Subroutine */ int _starpu_strti2_(char *uplo, char *diag, integer *n, real *a, 
	integer *lda, integer *info);

/* Subroutine */ int _starpu_strtri_(char *uplo, char *diag, integer *n, real *a, 
	integer *lda, integer *info);

/* Subroutine */ int _starpu_strtrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, real *a, integer *lda, real *b, integer *ldb, integer *
	info);

/* Subroutine */ int _starpu_strttf_(char *transr, char *uplo, integer *n, real *a, 
	integer *lda, real *arf, integer *info);

/* Subroutine */ int _starpu_strttp_(char *uplo, integer *n, real *a, integer *lda, 
	real *ap, integer *info);

/* Subroutine */ int _starpu_stzrqf_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, integer *info);

/* Subroutine */ int _starpu_stzrzf_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_xerbla_(char *srname, integer *info);

/* Subroutine */ int _starpu_xerbla_array__(char *srname_array__, integer *
	srname_len__, integer *info, ftnlen srname_array_len);

/* Subroutine */ int _starpu_zbdsqr_(char *uplo, integer *n, integer *ncvt, integer *
	nru, integer *ncc, doublereal *d__, doublereal *e, doublecomplex *vt, 
	integer *ldvt, doublecomplex *u, integer *ldu, doublecomplex *c__, 
	integer *ldc, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zcgesv_(integer *n, integer *nrhs, doublecomplex *a, 
	integer *lda, integer *ipiv, doublecomplex *b, integer *ldb, 
	doublecomplex *x, integer *ldx, doublecomplex *work, complex *swork, 
	doublereal *rwork, integer *iter, integer *info);

/* Subroutine */ int _starpu_zcposv_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublecomplex *x, integer *ldx, doublecomplex *work, complex *swork, 
	doublereal *rwork, integer *iter, integer *info);

/* Subroutine */ int _starpu_zdrscl_(integer *n, doublereal *sa, doublecomplex *sx, 
	integer *incx);

/* Subroutine */ int _starpu_zgbbrd_(char *vect, integer *m, integer *n, integer *ncc, 
	 integer *kl, integer *ku, doublecomplex *ab, integer *ldab, 
	doublereal *d__, doublereal *e, doublecomplex *q, integer *ldq, 
	doublecomplex *pt, integer *ldpt, doublecomplex *c__, integer *ldc, 
	doublecomplex *work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgbcon_(char *norm, integer *n, integer *kl, integer *ku, 
	 doublecomplex *ab, integer *ldab, integer *ipiv, doublereal *anorm, 
	doublereal *rcond, doublecomplex *work, doublereal *rwork, integer *
	info);

/* Subroutine */ int _starpu_zgbequ_(integer *m, integer *n, integer *kl, integer *ku, 
	 doublecomplex *ab, integer *ldab, doublereal *r__, doublereal *c__, 
	doublereal *rowcnd, doublereal *colcnd, doublereal *amax, integer *
	info);

/* Subroutine */ int _starpu_zgbequb_(integer *m, integer *n, integer *kl, integer *
	ku, doublecomplex *ab, integer *ldab, doublereal *r__, doublereal *
	c__, doublereal *rowcnd, doublereal *colcnd, doublereal *amax, 
	integer *info);

/* Subroutine */ int _starpu_zgbrfs_(char *trans, integer *n, integer *kl, integer *
	ku, integer *nrhs, doublecomplex *ab, integer *ldab, doublecomplex *
	afb, integer *ldafb, integer *ipiv, doublecomplex *b, integer *ldb, 
	doublecomplex *x, integer *ldx, doublereal *ferr, doublereal *berr, 
	doublecomplex *work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgbrfsx_(char *trans, char *equed, integer *n, integer *
	kl, integer *ku, integer *nrhs, doublecomplex *ab, integer *ldab, 
	doublecomplex *afb, integer *ldafb, integer *ipiv, doublereal *r__, 
	doublereal *c__, doublecomplex *b, integer *ldb, doublecomplex *x, 
	integer *ldx, doublereal *rcond, doublereal *berr, integer *
	n_err_bnds__, doublereal *err_bnds_norm__, doublereal *
	err_bnds_comp__, integer *nparams, doublereal *params, doublecomplex *
	work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgbsv_(integer *n, integer *kl, integer *ku, integer *
	nrhs, doublecomplex *ab, integer *ldab, integer *ipiv, doublecomplex *
	b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_zgbsvx_(char *fact, char *trans, integer *n, integer *kl, 
	 integer *ku, integer *nrhs, doublecomplex *ab, integer *ldab, 
	doublecomplex *afb, integer *ldafb, integer *ipiv, char *equed, 
	doublereal *r__, doublereal *c__, doublecomplex *b, integer *ldb, 
	doublecomplex *x, integer *ldx, doublereal *rcond, doublereal *ferr, 
	doublereal *berr, doublecomplex *work, doublereal *rwork, integer *
	info);

/* Subroutine */ int _starpu_zgbsvxx_(char *fact, char *trans, integer *n, integer *
	kl, integer *ku, integer *nrhs, doublecomplex *ab, integer *ldab, 
	doublecomplex *afb, integer *ldafb, integer *ipiv, char *equed, 
	doublereal *r__, doublereal *c__, doublecomplex *b, integer *ldb, 
	doublecomplex *x, integer *ldx, doublereal *rcond, doublereal *rpvgrw, 
	 doublereal *berr, integer *n_err_bnds__, doublereal *err_bnds_norm__, 
	 doublereal *err_bnds_comp__, integer *nparams, doublereal *params, 
	doublecomplex *work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgbtf2_(integer *m, integer *n, integer *kl, integer *ku, 
	 doublecomplex *ab, integer *ldab, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_zgbtrf_(integer *m, integer *n, integer *kl, integer *ku, 
	 doublecomplex *ab, integer *ldab, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_zgbtrs_(char *trans, integer *n, integer *kl, integer *
	ku, integer *nrhs, doublecomplex *ab, integer *ldab, integer *ipiv, 
	doublecomplex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_zgebak_(char *job, char *side, integer *n, integer *ilo, 
	integer *ihi, doublereal *scale, integer *m, doublecomplex *v, 
	integer *ldv, integer *info);

/* Subroutine */ int _starpu_zgebal_(char *job, integer *n, doublecomplex *a, integer 
	*lda, integer *ilo, integer *ihi, doublereal *scale, integer *info);

/* Subroutine */ int _starpu_zgebd2_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublereal *d__, doublereal *e, doublecomplex *tauq, 
	doublecomplex *taup, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zgebrd_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublereal *d__, doublereal *e, doublecomplex *tauq, 
	doublecomplex *taup, doublecomplex *work, integer *lwork, integer *
	info);

/* Subroutine */ int _starpu_zgecon_(char *norm, integer *n, doublecomplex *a, 
	integer *lda, doublereal *anorm, doublereal *rcond, doublecomplex *
	work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgeequ_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublereal *r__, doublereal *c__, doublereal *rowcnd, 
	doublereal *colcnd, doublereal *amax, integer *info);

/* Subroutine */ int _starpu_zgeequb_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublereal *r__, doublereal *c__, doublereal *rowcnd, 
	doublereal *colcnd, doublereal *amax, integer *info);

/* Subroutine */ int _starpu_zgees_(char *jobvs, char *sort, L_fp select, integer *n, 
	doublecomplex *a, integer *lda, integer *sdim, doublecomplex *w, 
	doublecomplex *vs, integer *ldvs, doublecomplex *work, integer *lwork, 
	 doublereal *rwork, logical *bwork, integer *info);

/* Subroutine */ int _starpu_zgeesx_(char *jobvs, char *sort, L_fp select, char *
	sense, integer *n, doublecomplex *a, integer *lda, integer *sdim, 
	doublecomplex *w, doublecomplex *vs, integer *ldvs, doublereal *
	rconde, doublereal *rcondv, doublecomplex *work, integer *lwork, 
	doublereal *rwork, logical *bwork, integer *info);

/* Subroutine */ int _starpu_zgeev_(char *jobvl, char *jobvr, integer *n, 
	doublecomplex *a, integer *lda, doublecomplex *w, doublecomplex *vl, 
	integer *ldvl, doublecomplex *vr, integer *ldvr, doublecomplex *work, 
	integer *lwork, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgeevx_(char *balanc, char *jobvl, char *jobvr, char *
	sense, integer *n, doublecomplex *a, integer *lda, doublecomplex *w, 
	doublecomplex *vl, integer *ldvl, doublecomplex *vr, integer *ldvr, 
	integer *ilo, integer *ihi, doublereal *scale, doublereal *abnrm, 
	doublereal *rconde, doublereal *rcondv, doublecomplex *work, integer *
	lwork, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgegs_(char *jobvsl, char *jobvsr, integer *n, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublecomplex *alpha, doublecomplex *beta, doublecomplex *vsl, 
	integer *ldvsl, doublecomplex *vsr, integer *ldvsr, doublecomplex *
	work, integer *lwork, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgegv_(char *jobvl, char *jobvr, integer *n, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublecomplex *alpha, doublecomplex *beta, doublecomplex *vl, integer 
	*ldvl, doublecomplex *vr, integer *ldvr, doublecomplex *work, integer 
	*lwork, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgehd2_(integer *n, integer *ilo, integer *ihi, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *info);

/* Subroutine */ int _starpu_zgehrd_(integer *n, integer *ilo, integer *ihi, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zgelq2_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublecomplex *tau, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zgelqf_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublecomplex *tau, doublecomplex *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int _starpu_zgels_(char *trans, integer *m, integer *n, integer *
	nrhs, doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublecomplex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zgelsd_(integer *m, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublereal *s, doublereal *rcond, integer *rank, doublecomplex *work, 
	integer *lwork, doublereal *rwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_zgelss_(integer *m, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublereal *s, doublereal *rcond, integer *rank, doublecomplex *work, 
	integer *lwork, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgelsx_(integer *m, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	integer *jpvt, doublereal *rcond, integer *rank, doublecomplex *work, 
	doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgelsy_(integer *m, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	integer *jpvt, doublereal *rcond, integer *rank, doublecomplex *work, 
	integer *lwork, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgeql2_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublecomplex *tau, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zgeqlf_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublecomplex *tau, doublecomplex *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int _starpu_zgeqp3_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, integer *jpvt, doublecomplex *tau, doublecomplex *work, 
	integer *lwork, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgeqpf_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, integer *jpvt, doublecomplex *tau, doublecomplex *work, 
	doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgeqr2_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublecomplex *tau, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zgeqrf_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublecomplex *tau, doublecomplex *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int _starpu_zgerfs_(char *trans, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, doublecomplex *af, integer *ldaf, 
	integer *ipiv, doublecomplex *b, integer *ldb, doublecomplex *x, 
	integer *ldx, doublereal *ferr, doublereal *berr, doublecomplex *work, 
	 doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgerfsx_(char *trans, char *equed, integer *n, integer *
	nrhs, doublecomplex *a, integer *lda, doublecomplex *af, integer *
	ldaf, integer *ipiv, doublereal *r__, doublereal *c__, doublecomplex *
	b, integer *ldb, doublecomplex *x, integer *ldx, doublereal *rcond, 
	doublereal *berr, integer *n_err_bnds__, doublereal *err_bnds_norm__, 
	doublereal *err_bnds_comp__, integer *nparams, doublereal *params, 
	doublecomplex *work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgerq2_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublecomplex *tau, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zgerqf_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublecomplex *tau, doublecomplex *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int _starpu_zgesc2_(integer *n, doublecomplex *a, integer *lda, 
	doublecomplex *rhs, integer *ipiv, integer *jpiv, doublereal *scale);

/* Subroutine */ int _starpu_zgesdd_(char *jobz, integer *m, integer *n, 
	doublecomplex *a, integer *lda, doublereal *s, doublecomplex *u, 
	integer *ldu, doublecomplex *vt, integer *ldvt, doublecomplex *work, 
	integer *lwork, doublereal *rwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_zgesv_(integer *n, integer *nrhs, doublecomplex *a, 
	integer *lda, integer *ipiv, doublecomplex *b, integer *ldb, integer *
	info);

/* Subroutine */ int _starpu_zgesvd_(char *jobu, char *jobvt, integer *m, integer *n, 
	doublecomplex *a, integer *lda, doublereal *s, doublecomplex *u, 
	integer *ldu, doublecomplex *vt, integer *ldvt, doublecomplex *work, 
	integer *lwork, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgesvx_(char *fact, char *trans, integer *n, integer *
	nrhs, doublecomplex *a, integer *lda, doublecomplex *af, integer *
	ldaf, integer *ipiv, char *equed, doublereal *r__, doublereal *c__, 
	doublecomplex *b, integer *ldb, doublecomplex *x, integer *ldx, 
	doublereal *rcond, doublereal *ferr, doublereal *berr, doublecomplex *
	work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgesvxx_(char *fact, char *trans, integer *n, integer *
	nrhs, doublecomplex *a, integer *lda, doublecomplex *af, integer *
	ldaf, integer *ipiv, char *equed, doublereal *r__, doublereal *c__, 
	doublecomplex *b, integer *ldb, doublecomplex *x, integer *ldx, 
	doublereal *rcond, doublereal *rpvgrw, doublereal *berr, integer *
	n_err_bnds__, doublereal *err_bnds_norm__, doublereal *
	err_bnds_comp__, integer *nparams, doublereal *params, doublecomplex *
	work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgetc2_(integer *n, doublecomplex *a, integer *lda, 
	integer *ipiv, integer *jpiv, integer *info);

/* Subroutine */ int _starpu_zgetf2_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_zgetrf_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_zgetri_(integer *n, doublecomplex *a, integer *lda, 
	integer *ipiv, doublecomplex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zgetrs_(char *trans, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, integer *ipiv, doublecomplex *b, 
	integer *ldb, integer *info);

/* Subroutine */ int _starpu_zggbak_(char *job, char *side, integer *n, integer *ilo, 
	integer *ihi, doublereal *lscale, doublereal *rscale, integer *m, 
	doublecomplex *v, integer *ldv, integer *info);

/* Subroutine */ int _starpu_zggbal_(char *job, integer *n, doublecomplex *a, integer 
	*lda, doublecomplex *b, integer *ldb, integer *ilo, integer *ihi, 
	doublereal *lscale, doublereal *rscale, doublereal *work, integer *
	info);

/* Subroutine */ int _starpu_zgges_(char *jobvsl, char *jobvsr, char *sort, L_fp 
	selctg, integer *n, doublecomplex *a, integer *lda, doublecomplex *b, 
	integer *ldb, integer *sdim, doublecomplex *alpha, doublecomplex *
	beta, doublecomplex *vsl, integer *ldvsl, doublecomplex *vsr, integer 
	*ldvsr, doublecomplex *work, integer *lwork, doublereal *rwork, 
	logical *bwork, integer *info);

/* Subroutine */ int _starpu_zggesx_(char *jobvsl, char *jobvsr, char *sort, L_fp 
	selctg, char *sense, integer *n, doublecomplex *a, integer *lda, 
	doublecomplex *b, integer *ldb, integer *sdim, doublecomplex *alpha, 
	doublecomplex *beta, doublecomplex *vsl, integer *ldvsl, 
	doublecomplex *vsr, integer *ldvsr, doublereal *rconde, doublereal *
	rcondv, doublecomplex *work, integer *lwork, doublereal *rwork, 
	integer *iwork, integer *liwork, logical *bwork, integer *info);

/* Subroutine */ int _starpu_zggev_(char *jobvl, char *jobvr, integer *n, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublecomplex *alpha, doublecomplex *beta, doublecomplex *vl, integer 
	*ldvl, doublecomplex *vr, integer *ldvr, doublecomplex *work, integer 
	*lwork, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zggevx_(char *balanc, char *jobvl, char *jobvr, char *
	sense, integer *n, doublecomplex *a, integer *lda, doublecomplex *b, 
	integer *ldb, doublecomplex *alpha, doublecomplex *beta, 
	doublecomplex *vl, integer *ldvl, doublecomplex *vr, integer *ldvr, 
	integer *ilo, integer *ihi, doublereal *lscale, doublereal *rscale, 
	doublereal *abnrm, doublereal *bbnrm, doublereal *rconde, doublereal *
	rcondv, doublecomplex *work, integer *lwork, doublereal *rwork, 
	integer *iwork, logical *bwork, integer *info);

/* Subroutine */ int _starpu_zggglm_(integer *n, integer *m, integer *p, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublecomplex *d__, doublecomplex *x, doublecomplex *y, doublecomplex 
	*work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zgghrd_(char *compq, char *compz, integer *n, integer *
	ilo, integer *ihi, doublecomplex *a, integer *lda, doublecomplex *b, 
	integer *ldb, doublecomplex *q, integer *ldq, doublecomplex *z__, 
	integer *ldz, integer *info);

/* Subroutine */ int _starpu_zgglse_(integer *m, integer *n, integer *p, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublecomplex *c__, doublecomplex *d__, doublecomplex *x, 
	doublecomplex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zggqrf_(integer *n, integer *m, integer *p, 
	doublecomplex *a, integer *lda, doublecomplex *taua, doublecomplex *b, 
	 integer *ldb, doublecomplex *taub, doublecomplex *work, integer *
	lwork, integer *info);

/* Subroutine */ int _starpu_zggrqf_(integer *m, integer *p, integer *n, 
	doublecomplex *a, integer *lda, doublecomplex *taua, doublecomplex *b, 
	 integer *ldb, doublecomplex *taub, doublecomplex *work, integer *
	lwork, integer *info);

/* Subroutine */ int _starpu_zggsvd_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *n, integer *p, integer *k, integer *l, doublecomplex *a, 
	integer *lda, doublecomplex *b, integer *ldb, doublereal *alpha, 
	doublereal *beta, doublecomplex *u, integer *ldu, doublecomplex *v, 
	integer *ldv, doublecomplex *q, integer *ldq, doublecomplex *work, 
	doublereal *rwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_zggsvp_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *p, integer *n, doublecomplex *a, integer *lda, doublecomplex 
	*b, integer *ldb, doublereal *tola, doublereal *tolb, integer *k, 
	integer *l, doublecomplex *u, integer *ldu, doublecomplex *v, integer 
	*ldv, doublecomplex *q, integer *ldq, integer *iwork, doublereal *
	rwork, doublecomplex *tau, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zgtcon_(char *norm, integer *n, doublecomplex *dl, 
	doublecomplex *d__, doublecomplex *du, doublecomplex *du2, integer *
	ipiv, doublereal *anorm, doublereal *rcond, doublecomplex *work, 
	integer *info);

/* Subroutine */ int _starpu_zgtrfs_(char *trans, integer *n, integer *nrhs, 
	doublecomplex *dl, doublecomplex *d__, doublecomplex *du, 
	doublecomplex *dlf, doublecomplex *df, doublecomplex *duf, 
	doublecomplex *du2, integer *ipiv, doublecomplex *b, integer *ldb, 
	doublecomplex *x, integer *ldx, doublereal *ferr, doublereal *berr, 
	doublecomplex *work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zgtsv_(integer *n, integer *nrhs, doublecomplex *dl, 
	doublecomplex *d__, doublecomplex *du, doublecomplex *b, integer *ldb, 
	 integer *info);

/* Subroutine */ int _starpu_zgtsvx_(char *fact, char *trans, integer *n, integer *
	nrhs, doublecomplex *dl, doublecomplex *d__, doublecomplex *du, 
	doublecomplex *dlf, doublecomplex *df, doublecomplex *duf, 
	doublecomplex *du2, integer *ipiv, doublecomplex *b, integer *ldb, 
	doublecomplex *x, integer *ldx, doublereal *rcond, doublereal *ferr, 
	doublereal *berr, doublecomplex *work, doublereal *rwork, integer *
	info);

/* Subroutine */ int _starpu_zgttrf_(integer *n, doublecomplex *dl, doublecomplex *
	d__, doublecomplex *du, doublecomplex *du2, integer *ipiv, integer *
	info);

/* Subroutine */ int _starpu_zgttrs_(char *trans, integer *n, integer *nrhs, 
	doublecomplex *dl, doublecomplex *d__, doublecomplex *du, 
	doublecomplex *du2, integer *ipiv, doublecomplex *b, integer *ldb, 
	integer *info);

/* Subroutine */ int _starpu_zgtts2_(integer *itrans, integer *n, integer *nrhs, 
	doublecomplex *dl, doublecomplex *d__, doublecomplex *du, 
	doublecomplex *du2, integer *ipiv, doublecomplex *b, integer *ldb);

/* Subroutine */ int _starpu_zhbev_(char *jobz, char *uplo, integer *n, integer *kd, 
	doublecomplex *ab, integer *ldab, doublereal *w, doublecomplex *z__, 
	integer *ldz, doublecomplex *work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zhbevd_(char *jobz, char *uplo, integer *n, integer *kd, 
	doublecomplex *ab, integer *ldab, doublereal *w, doublecomplex *z__, 
	integer *ldz, doublecomplex *work, integer *lwork, doublereal *rwork, 
	integer *lrwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_zhbevx_(char *jobz, char *range, char *uplo, integer *n, 
	integer *kd, doublecomplex *ab, integer *ldab, doublecomplex *q, 
	integer *ldq, doublereal *vl, doublereal *vu, integer *il, integer *
	iu, doublereal *abstol, integer *m, doublereal *w, doublecomplex *z__, 
	 integer *ldz, doublecomplex *work, doublereal *rwork, integer *iwork, 
	 integer *ifail, integer *info);

/* Subroutine */ int _starpu_zhbgst_(char *vect, char *uplo, integer *n, integer *ka, 
	integer *kb, doublecomplex *ab, integer *ldab, doublecomplex *bb, 
	integer *ldbb, doublecomplex *x, integer *ldx, doublecomplex *work, 
	doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zhbgv_(char *jobz, char *uplo, integer *n, integer *ka, 
	integer *kb, doublecomplex *ab, integer *ldab, doublecomplex *bb, 
	integer *ldbb, doublereal *w, doublecomplex *z__, integer *ldz, 
	doublecomplex *work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zhbgvd_(char *jobz, char *uplo, integer *n, integer *ka, 
	integer *kb, doublecomplex *ab, integer *ldab, doublecomplex *bb, 
	integer *ldbb, doublereal *w, doublecomplex *z__, integer *ldz, 
	doublecomplex *work, integer *lwork, doublereal *rwork, integer *
	lrwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_zhbgvx_(char *jobz, char *range, char *uplo, integer *n, 
	integer *ka, integer *kb, doublecomplex *ab, integer *ldab, 
	doublecomplex *bb, integer *ldbb, doublecomplex *q, integer *ldq, 
	doublereal *vl, doublereal *vu, integer *il, integer *iu, doublereal *
	abstol, integer *m, doublereal *w, doublecomplex *z__, integer *ldz, 
	doublecomplex *work, doublereal *rwork, integer *iwork, integer *
	ifail, integer *info);

/* Subroutine */ int _starpu_zhbtrd_(char *vect, char *uplo, integer *n, integer *kd, 
	doublecomplex *ab, integer *ldab, doublereal *d__, doublereal *e, 
	doublecomplex *q, integer *ldq, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zhecon_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, integer *ipiv, doublereal *anorm, doublereal *rcond, 
	doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zheequb_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, doublereal *s, doublereal *scond, doublereal *amax, 
	doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zheev_(char *jobz, char *uplo, integer *n, doublecomplex 
	*a, integer *lda, doublereal *w, doublecomplex *work, integer *lwork, 
	doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zheevd_(char *jobz, char *uplo, integer *n, 
	doublecomplex *a, integer *lda, doublereal *w, doublecomplex *work, 
	integer *lwork, doublereal *rwork, integer *lrwork, integer *iwork, 
	integer *liwork, integer *info);

/* Subroutine */ int _starpu_zheevr_(char *jobz, char *range, char *uplo, integer *n, 
	doublecomplex *a, integer *lda, doublereal *vl, doublereal *vu, 
	integer *il, integer *iu, doublereal *abstol, integer *m, doublereal *
	w, doublecomplex *z__, integer *ldz, integer *isuppz, doublecomplex *
	work, integer *lwork, doublereal *rwork, integer *lrwork, integer *
	iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_zheevx_(char *jobz, char *range, char *uplo, integer *n, 
	doublecomplex *a, integer *lda, doublereal *vl, doublereal *vu, 
	integer *il, integer *iu, doublereal *abstol, integer *m, doublereal *
	w, doublecomplex *z__, integer *ldz, doublecomplex *work, integer *
	lwork, doublereal *rwork, integer *iwork, integer *ifail, integer *
	info);

/* Subroutine */ int _starpu_zhegs2_(integer *itype, char *uplo, integer *n, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	integer *info);

/* Subroutine */ int _starpu_zhegst_(integer *itype, char *uplo, integer *n, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	integer *info);

/* Subroutine */ int _starpu_zhegv_(integer *itype, char *jobz, char *uplo, integer *
	n, doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublereal *w, doublecomplex *work, integer *lwork, doublereal *rwork, 
	 integer *info);

/* Subroutine */ int _starpu_zhegvd_(integer *itype, char *jobz, char *uplo, integer *
	n, doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublereal *w, doublecomplex *work, integer *lwork, doublereal *rwork, 
	 integer *lrwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_zhegvx_(integer *itype, char *jobz, char *range, char *
	uplo, integer *n, doublecomplex *a, integer *lda, doublecomplex *b, 
	integer *ldb, doublereal *vl, doublereal *vu, integer *il, integer *
	iu, doublereal *abstol, integer *m, doublereal *w, doublecomplex *z__, 
	 integer *ldz, doublecomplex *work, integer *lwork, doublereal *rwork, 
	 integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu_zherfs_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, doublecomplex *af, integer *ldaf, 
	integer *ipiv, doublecomplex *b, integer *ldb, doublecomplex *x, 
	integer *ldx, doublereal *ferr, doublereal *berr, doublecomplex *work, 
	 doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zherfsx_(char *uplo, char *equed, integer *n, integer *
	nrhs, doublecomplex *a, integer *lda, doublecomplex *af, integer *
	ldaf, integer *ipiv, doublereal *s, doublecomplex *b, integer *ldb, 
	doublecomplex *x, integer *ldx, doublereal *rcond, doublereal *berr, 
	integer *n_err_bnds__, doublereal *err_bnds_norm__, doublereal *
	err_bnds_comp__, integer *nparams, doublereal *params, doublecomplex *
	work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zhesv_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, integer *ipiv, doublecomplex *b, 
	integer *ldb, doublecomplex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zhesvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublecomplex *a, integer *lda, doublecomplex *af, integer *
	ldaf, integer *ipiv, doublecomplex *b, integer *ldb, doublecomplex *x, 
	 integer *ldx, doublereal *rcond, doublereal *ferr, doublereal *berr, 
	doublecomplex *work, integer *lwork, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zhesvxx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublecomplex *a, integer *lda, doublecomplex *af, integer *
	ldaf, integer *ipiv, char *equed, doublereal *s, doublecomplex *b, 
	integer *ldb, doublecomplex *x, integer *ldx, doublereal *rcond, 
	doublereal *rpvgrw, doublereal *berr, integer *n_err_bnds__, 
	doublereal *err_bnds_norm__, doublereal *err_bnds_comp__, integer *
	nparams, doublereal *params, doublecomplex *work, doublereal *rwork, 
	integer *info);

/* Subroutine */ int _starpu_zhetd2_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, doublereal *d__, doublereal *e, doublecomplex *tau, 
	integer *info);

/* Subroutine */ int _starpu_zhetf2_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_zhetrd_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, doublereal *d__, doublereal *e, doublecomplex *tau, 
	doublecomplex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zhetrf_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, integer *ipiv, doublecomplex *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_zhetri_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, integer *ipiv, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zhetrs_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, integer *ipiv, doublecomplex *b, 
	integer *ldb, integer *info);

/* Subroutine */ int _starpu_zhfrk_(char *transr, char *uplo, char *trans, integer *n, 
	 integer *k, doublereal *alpha, doublecomplex *a, integer *lda, 
	doublereal *beta, doublecomplex *c__);

/* Subroutine */ int _starpu_zhgeqz_(char *job, char *compq, char *compz, integer *n, 
	integer *ilo, integer *ihi, doublecomplex *h__, integer *ldh, 
	doublecomplex *t, integer *ldt, doublecomplex *alpha, doublecomplex *
	beta, doublecomplex *q, integer *ldq, doublecomplex *z__, integer *
	ldz, doublecomplex *work, integer *lwork, doublereal *rwork, integer *
	info);

/* Subroutine */ int _starpu_zhpcon_(char *uplo, integer *n, doublecomplex *ap, 
	integer *ipiv, doublereal *anorm, doublereal *rcond, doublecomplex *
	work, integer *info);

/* Subroutine */ int _starpu_zhpev_(char *jobz, char *uplo, integer *n, doublecomplex 
	*ap, doublereal *w, doublecomplex *z__, integer *ldz, doublecomplex *
	work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zhpevd_(char *jobz, char *uplo, integer *n, 
	doublecomplex *ap, doublereal *w, doublecomplex *z__, integer *ldz, 
	doublecomplex *work, integer *lwork, doublereal *rwork, integer *
	lrwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_zhpevx_(char *jobz, char *range, char *uplo, integer *n, 
	doublecomplex *ap, doublereal *vl, doublereal *vu, integer *il, 
	integer *iu, doublereal *abstol, integer *m, doublereal *w, 
	doublecomplex *z__, integer *ldz, doublecomplex *work, doublereal *
	rwork, integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int _starpu_zhpgst_(integer *itype, char *uplo, integer *n, 
	doublecomplex *ap, doublecomplex *bp, integer *info);

/* Subroutine */ int _starpu_zhpgv_(integer *itype, char *jobz, char *uplo, integer *
	n, doublecomplex *ap, doublecomplex *bp, doublereal *w, doublecomplex 
	*z__, integer *ldz, doublecomplex *work, doublereal *rwork, integer *
	info);

/* Subroutine */ int _starpu_zhpgvd_(integer *itype, char *jobz, char *uplo, integer *
	n, doublecomplex *ap, doublecomplex *bp, doublereal *w, doublecomplex 
	*z__, integer *ldz, doublecomplex *work, integer *lwork, doublereal *
	rwork, integer *lrwork, integer *iwork, integer *liwork, integer *
	info);

/* Subroutine */ int _starpu_zhpgvx_(integer *itype, char *jobz, char *range, char *
	uplo, integer *n, doublecomplex *ap, doublecomplex *bp, doublereal *
	vl, doublereal *vu, integer *il, integer *iu, doublereal *abstol, 
	integer *m, doublereal *w, doublecomplex *z__, integer *ldz, 
	doublecomplex *work, doublereal *rwork, integer *iwork, integer *
	ifail, integer *info);

/* Subroutine */ int _starpu_zhprfs_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *ap, doublecomplex *afp, integer *ipiv, doublecomplex *
	b, integer *ldb, doublecomplex *x, integer *ldx, doublereal *ferr, 
	doublereal *berr, doublecomplex *work, doublereal *rwork, integer *
	info);

/* Subroutine */ int _starpu_zhpsv_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *ap, integer *ipiv, doublecomplex *b, integer *ldb, 
	integer *info);

/* Subroutine */ int _starpu_zhpsvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublecomplex *ap, doublecomplex *afp, integer *ipiv, 
	doublecomplex *b, integer *ldb, doublecomplex *x, integer *ldx, 
	doublereal *rcond, doublereal *ferr, doublereal *berr, doublecomplex *
	work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zhptrd_(char *uplo, integer *n, doublecomplex *ap, 
	doublereal *d__, doublereal *e, doublecomplex *tau, integer *info);

/* Subroutine */ int _starpu_zhptrf_(char *uplo, integer *n, doublecomplex *ap, 
	integer *ipiv, integer *info);

/* Subroutine */ int _starpu_zhptri_(char *uplo, integer *n, doublecomplex *ap, 
	integer *ipiv, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zhptrs_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *ap, integer *ipiv, doublecomplex *b, integer *ldb, 
	integer *info);

/* Subroutine */ int _starpu_zhsein_(char *side, char *eigsrc, char *initv, logical *
	select, integer *n, doublecomplex *h__, integer *ldh, doublecomplex *
	w, doublecomplex *vl, integer *ldvl, doublecomplex *vr, integer *ldvr, 
	 integer *mm, integer *m, doublecomplex *work, doublereal *rwork, 
	integer *ifaill, integer *ifailr, integer *info);

/* Subroutine */ int _starpu_zhseqr_(char *job, char *compz, integer *n, integer *ilo, 
	 integer *ihi, doublecomplex *h__, integer *ldh, doublecomplex *w, 
	doublecomplex *z__, integer *ldz, doublecomplex *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int _starpu_zla_gbamv__(integer *trans, integer *m, integer *n, 
	integer *kl, integer *ku, doublereal *alpha, doublecomplex *ab, 
	integer *ldab, doublecomplex *x, integer *incx, doublereal *beta, 
	doublereal *y, integer *incy);

doublereal _starpu_zla_gbrcond_c__(char *trans, integer *n, integer *kl, integer *ku, 
	doublecomplex *ab, integer *ldab, doublecomplex *afb, integer *ldafb, 
	integer *ipiv, doublereal *c__, logical *capply, integer *info, 
	doublecomplex *work, doublereal *rwork, ftnlen trans_len);

doublereal _starpu_zla_gbrcond_x__(char *trans, integer *n, integer *kl, integer *ku, 
	doublecomplex *ab, integer *ldab, doublecomplex *afb, integer *ldafb, 
	integer *ipiv, doublecomplex *x, integer *info, doublecomplex *work, 
	doublereal *rwork, ftnlen trans_len);

/* Subroutine */ int _starpu_zla_gbrfsx_extended__(integer *prec_type__, integer *
	trans_type__, integer *n, integer *kl, integer *ku, integer *nrhs, 
	doublecomplex *ab, integer *ldab, doublecomplex *afb, integer *ldafb, 
	integer *ipiv, logical *colequ, doublereal *c__, doublecomplex *b, 
	integer *ldb, doublecomplex *y, integer *ldy, doublereal *berr_out__, 
	integer *n_norms__, doublereal *errs_n__, doublereal *errs_c__, 
	doublecomplex *res, doublereal *ayb, doublecomplex *dy, doublecomplex 
	*y_tail__, doublereal *rcond, integer *ithresh, doublereal *rthresh, 
	doublereal *dz_ub__, logical *ignore_cwise__, integer *info);

doublereal _starpu_zla_gbrpvgrw__(integer *n, integer *kl, integer *ku, integer *
	ncols, doublecomplex *ab, integer *ldab, doublecomplex *afb, integer *
	ldafb);

/* Subroutine */ int _starpu_zla_geamv__(integer *trans, integer *m, integer *n, 
	doublereal *alpha, doublecomplex *a, integer *lda, doublecomplex *x, 
	integer *incx, doublereal *beta, doublereal *y, integer *incy);

doublereal _starpu_zla_gercond_c__(char *trans, integer *n, doublecomplex *a, integer 
	*lda, doublecomplex *af, integer *ldaf, integer *ipiv, doublereal *
	c__, logical *capply, integer *info, doublecomplex *work, doublereal *
	rwork, ftnlen trans_len);

doublereal _starpu_zla_gercond_x__(char *trans, integer *n, doublecomplex *a, integer 
	*lda, doublecomplex *af, integer *ldaf, integer *ipiv, doublecomplex *
	x, integer *info, doublecomplex *work, doublereal *rwork, ftnlen 
	trans_len);

/* Subroutine */ int _starpu_zla_gerfsx_extended__(integer *prec_type__, integer *
	trans_type__, integer *n, integer *nrhs, doublecomplex *a, integer *
	lda, doublecomplex *af, integer *ldaf, integer *ipiv, logical *colequ,
	 doublereal *c__, doublecomplex *b, integer *ldb, doublecomplex *y, 
	integer *ldy, doublereal *berr_out__, integer *n_norms__, doublereal *
	errs_n__, doublereal *errs_c__, doublecomplex *res, doublereal *ayb, 
	doublecomplex *dy, doublecomplex *y_tail__, doublereal *rcond, 
	integer *ithresh, doublereal *rthresh, doublereal *dz_ub__, logical *
	ignore_cwise__, integer *info);

/* Subroutine */ int _starpu_zla_heamv__(integer *uplo, integer *n, doublereal *alpha,
	 doublecomplex *a, integer *lda, doublecomplex *x, integer *incx, 
	doublereal *beta, doublereal *y, integer *incy);

doublereal _starpu_zla_hercond_c__(char *uplo, integer *n, doublecomplex *a, integer *
	lda, doublecomplex *af, integer *ldaf, integer *ipiv, doublereal *c__,
	 logical *capply, integer *info, doublecomplex *work, doublereal *
	rwork, ftnlen uplo_len);

doublereal _starpu_zla_hercond_x__(char *uplo, integer *n, doublecomplex *a, integer *
	lda, doublecomplex *af, integer *ldaf, integer *ipiv, doublecomplex *
	x, integer *info, doublecomplex *work, doublereal *rwork, ftnlen 
	uplo_len);

/* Subroutine */ int _starpu_zla_herfsx_extended__(integer *prec_type__, char *uplo, 
	integer *n, integer *nrhs, doublecomplex *a, integer *lda, 
	doublecomplex *af, integer *ldaf, integer *ipiv, logical *colequ, 
	doublereal *c__, doublecomplex *b, integer *ldb, doublecomplex *y, 
	integer *ldy, doublereal *berr_out__, integer *n_norms__, doublereal *
	errs_n__, doublereal *errs_c__, doublecomplex *res, doublereal *ayb, 
	doublecomplex *dy, doublecomplex *y_tail__, doublereal *rcond, 
	integer *ithresh, doublereal *rthresh, doublereal *dz_ub__, logical *
	ignore_cwise__, integer *info, ftnlen uplo_len);

doublereal _starpu_zla_herpvgrw__(char *uplo, integer *n, integer *info, 
	doublecomplex *a, integer *lda, doublecomplex *af, integer *ldaf, 
	integer *ipiv, doublereal *work, ftnlen uplo_len);

/* Subroutine */ int _starpu_zla_lin_berr__(integer *n, integer *nz, integer *nrhs, 
	doublecomplex *res, doublereal *ayb, doublereal *berr);

doublereal _starpu_zla_porcond_c__(char *uplo, integer *n, doublecomplex *a, integer *
	lda, doublecomplex *af, integer *ldaf, doublereal *c__, logical *
	capply, integer *info, doublecomplex *work, doublereal *rwork, ftnlen 
	uplo_len);

doublereal _starpu_zla_porcond_x__(char *uplo, integer *n, doublecomplex *a, integer *
	lda, doublecomplex *af, integer *ldaf, doublecomplex *x, integer *
	info, doublecomplex *work, doublereal *rwork, ftnlen uplo_len);

/* Subroutine */ int _starpu_zla_porfsx_extended__(integer *prec_type__, char *uplo, 
	integer *n, integer *nrhs, doublecomplex *a, integer *lda, 
	doublecomplex *af, integer *ldaf, logical *colequ, doublereal *c__, 
	doublecomplex *b, integer *ldb, doublecomplex *y, integer *ldy, 
	doublereal *berr_out__, integer *n_norms__, doublereal *errs_n__, 
	doublereal *errs_c__, doublecomplex *res, doublereal *ayb, 
	doublecomplex *dy, doublecomplex *y_tail__, doublereal *rcond, 
	integer *ithresh, doublereal *rthresh, doublereal *dz_ub__, logical *
	ignore_cwise__, integer *info, ftnlen uplo_len);

doublereal _starpu_zla_porpvgrw__(char *uplo, integer *ncols, doublecomplex *a, 
	integer *lda, doublecomplex *af, integer *ldaf, doublereal *work, 
	ftnlen uplo_len);

doublereal _starpu_zla_rpvgrw__(integer *n, integer *ncols, doublecomplex *a, integer 
	*lda, doublecomplex *af, integer *ldaf);

/* Subroutine */ int _starpu_zla_syamv__(integer *uplo, integer *n, doublereal *alpha,
	 doublecomplex *a, integer *lda, doublecomplex *x, integer *incx, 
	doublereal *beta, doublereal *y, integer *incy);

doublereal _starpu_zla_syrcond_c__(char *uplo, integer *n, doublecomplex *a, integer *
	lda, doublecomplex *af, integer *ldaf, integer *ipiv, doublereal *c__,
	 logical *capply, integer *info, doublecomplex *work, doublereal *
	rwork, ftnlen uplo_len);

doublereal _starpu_zla_syrcond_x__(char *uplo, integer *n, doublecomplex *a, integer *
	lda, doublecomplex *af, integer *ldaf, integer *ipiv, doublecomplex *
	x, integer *info, doublecomplex *work, doublereal *rwork, ftnlen 
	uplo_len);

/* Subroutine */ int _starpu_zla_syrfsx_extended__(integer *prec_type__, char *uplo, 
	integer *n, integer *nrhs, doublecomplex *a, integer *lda, 
	doublecomplex *af, integer *ldaf, integer *ipiv, logical *colequ, 
	doublereal *c__, doublecomplex *b, integer *ldb, doublecomplex *y, 
	integer *ldy, doublereal *berr_out__, integer *n_norms__, doublereal *
	errs_n__, doublereal *errs_c__, doublecomplex *res, doublereal *ayb, 
	doublecomplex *dy, doublecomplex *y_tail__, doublereal *rcond, 
	integer *ithresh, doublereal *rthresh, doublereal *dz_ub__, logical *
	ignore_cwise__, integer *info, ftnlen uplo_len);

doublereal _starpu_zla_syrpvgrw__(char *uplo, integer *n, integer *info, 
	doublecomplex *a, integer *lda, doublecomplex *af, integer *ldaf, 
	integer *ipiv, doublereal *work, ftnlen uplo_len);

/* Subroutine */ int _starpu_zla_wwaddw__(integer *n, doublecomplex *x, doublecomplex 
	*y, doublecomplex *w);

/* Subroutine */ int _starpu_zlabrd_(integer *m, integer *n, integer *nb, 
	doublecomplex *a, integer *lda, doublereal *d__, doublereal *e, 
	doublecomplex *tauq, doublecomplex *taup, doublecomplex *x, integer *
	ldx, doublecomplex *y, integer *ldy);

/* Subroutine */ int _starpu_zlacgv_(integer *n, doublecomplex *x, integer *incx);

/* Subroutine */ int _starpu_zlacn2_(integer *n, doublecomplex *v, doublecomplex *x, 
	doublereal *est, integer *kase, integer *isave);

/* Subroutine */ int _starpu_zlacon_(integer *n, doublecomplex *v, doublecomplex *x, 
	doublereal *est, integer *kase);

/* Subroutine */ int _starpu_zlacp2_(char *uplo, integer *m, integer *n, doublereal *
	a, integer *lda, doublecomplex *b, integer *ldb);

/* Subroutine */ int _starpu_zlacpy_(char *uplo, integer *m, integer *n, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb);

/* Subroutine */ int _starpu_zlacrm_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublereal *b, integer *ldb, doublecomplex *c__, 
	integer *ldc, doublereal *rwork);

/* Subroutine */ int _starpu_zlacrt_(integer *n, doublecomplex *cx, integer *incx, 
	doublecomplex *cy, integer *incy, doublecomplex *c__, doublecomplex *
	s);

/* Double Complex */ VOID _starpu_zladiv_(doublecomplex * ret_val, doublecomplex *x, 
	doublecomplex *y);

/* Subroutine */ int _starpu_zlaed0_(integer *qsiz, integer *n, doublereal *d__, 
	doublereal *e, doublecomplex *q, integer *ldq, doublecomplex *qstore, 
	integer *ldqs, doublereal *rwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_zlaed7_(integer *n, integer *cutpnt, integer *qsiz, 
	integer *tlvls, integer *curlvl, integer *curpbm, doublereal *d__, 
	doublecomplex *q, integer *ldq, doublereal *rho, integer *indxq, 
	doublereal *qstore, integer *qptr, integer *prmptr, integer *perm, 
	integer *givptr, integer *givcol, doublereal *givnum, doublecomplex *
	work, doublereal *rwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_zlaed8_(integer *k, integer *n, integer *qsiz, 
	doublecomplex *q, integer *ldq, doublereal *d__, doublereal *rho, 
	integer *cutpnt, doublereal *z__, doublereal *dlamda, doublecomplex *
	q2, integer *ldq2, doublereal *w, integer *indxp, integer *indx, 
	integer *indxq, integer *perm, integer *givptr, integer *givcol, 
	doublereal *givnum, integer *info);

/* Subroutine */ int _starpu_zlaein_(logical *rightv, logical *noinit, integer *n, 
	doublecomplex *h__, integer *ldh, doublecomplex *w, doublecomplex *v, 
	doublecomplex *b, integer *ldb, doublereal *rwork, doublereal *eps3, 
	doublereal *smlnum, integer *info);

/* Subroutine */ int _starpu_zlaesy_(doublecomplex *a, doublecomplex *b, 
	doublecomplex *c__, doublecomplex *rt1, doublecomplex *rt2, 
	doublecomplex *evscal, doublecomplex *cs1, doublecomplex *sn1);

/* Subroutine */ int _starpu_zlaev2_(doublecomplex *a, doublecomplex *b, 
	doublecomplex *c__, doublereal *rt1, doublereal *rt2, doublereal *cs1, 
	 doublecomplex *sn1);

/* Subroutine */ int _starpu_zlag2c_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, complex *sa, integer *ldsa, integer *info);

/* Subroutine */ int _starpu_zlags2_(logical *upper, doublereal *a1, doublecomplex *
	a2, doublereal *a3, doublereal *b1, doublecomplex *b2, doublereal *b3, 
	 doublereal *csu, doublecomplex *snu, doublereal *csv, doublecomplex *
	snv, doublereal *csq, doublecomplex *snq);

/* Subroutine */ int _starpu_zlagtm_(char *trans, integer *n, integer *nrhs, 
	doublereal *alpha, doublecomplex *dl, doublecomplex *d__, 
	doublecomplex *du, doublecomplex *x, integer *ldx, doublereal *beta, 
	doublecomplex *b, integer *ldb);

/* Subroutine */ int _starpu_zlahef_(char *uplo, integer *n, integer *nb, integer *kb, 
	 doublecomplex *a, integer *lda, integer *ipiv, doublecomplex *w, 
	integer *ldw, integer *info);

/* Subroutine */ int _starpu_zlahqr_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, doublecomplex *h__, integer *ldh, 
	doublecomplex *w, integer *iloz, integer *ihiz, doublecomplex *z__, 
	integer *ldz, integer *info);

/* Subroutine */ int _starpu_zlahr2_(integer *n, integer *k, integer *nb, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *t, 
	integer *ldt, doublecomplex *y, integer *ldy);

/* Subroutine */ int _starpu_zlahrd_(integer *n, integer *k, integer *nb, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *t, 
	integer *ldt, doublecomplex *y, integer *ldy);

/* Subroutine */ int _starpu_zlaic1_(integer *job, integer *j, doublecomplex *x, 
	doublereal *sest, doublecomplex *w, doublecomplex *gamma, doublereal *
	sestpr, doublecomplex *s, doublecomplex *c__);

/* Subroutine */ int _starpu_zlals0_(integer *icompq, integer *nl, integer *nr, 
	integer *sqre, integer *nrhs, doublecomplex *b, integer *ldb, 
	doublecomplex *bx, integer *ldbx, integer *perm, integer *givptr, 
	integer *givcol, integer *ldgcol, doublereal *givnum, integer *ldgnum, 
	 doublereal *poles, doublereal *difl, doublereal *difr, doublereal *
	z__, integer *k, doublereal *c__, doublereal *s, doublereal *rwork, 
	integer *info);

/* Subroutine */ int _starpu_zlalsa_(integer *icompq, integer *smlsiz, integer *n, 
	integer *nrhs, doublecomplex *b, integer *ldb, doublecomplex *bx, 
	integer *ldbx, doublereal *u, integer *ldu, doublereal *vt, integer *
	k, doublereal *difl, doublereal *difr, doublereal *z__, doublereal *
	poles, integer *givptr, integer *givcol, integer *ldgcol, integer *
	perm, doublereal *givnum, doublereal *c__, doublereal *s, doublereal *
	rwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_zlalsd_(char *uplo, integer *smlsiz, integer *n, integer 
	*nrhs, doublereal *d__, doublereal *e, doublecomplex *b, integer *ldb, 
	 doublereal *rcond, integer *rank, doublecomplex *work, doublereal *
	rwork, integer *iwork, integer *info);

doublereal _starpu_zlangb_(char *norm, integer *n, integer *kl, integer *ku, 
	doublecomplex *ab, integer *ldab, doublereal *work);

doublereal _starpu_zlange_(char *norm, integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublereal *work);

doublereal _starpu_zlangt_(char *norm, integer *n, doublecomplex *dl, doublecomplex *
	d__, doublecomplex *du);

doublereal _starpu_zlanhb_(char *norm, char *uplo, integer *n, integer *k, 
	doublecomplex *ab, integer *ldab, doublereal *work);

doublereal _starpu_zlanhe_(char *norm, char *uplo, integer *n, doublecomplex *a, 
	integer *lda, doublereal *work);

doublereal _starpu_zlanhf_(char *norm, char *transr, char *uplo, integer *n, 
	doublecomplex *a, doublereal *work);

doublereal _starpu_zlanhp_(char *norm, char *uplo, integer *n, doublecomplex *ap, 
	doublereal *work);

doublereal _starpu_zlanhs_(char *norm, integer *n, doublecomplex *a, integer *lda, 
	doublereal *work);

doublereal _starpu_zlanht_(char *norm, integer *n, doublereal *d__, doublecomplex *e);

doublereal _starpu_zlansb_(char *norm, char *uplo, integer *n, integer *k, 
	doublecomplex *ab, integer *ldab, doublereal *work);

doublereal _starpu_zlansp_(char *norm, char *uplo, integer *n, doublecomplex *ap, 
	doublereal *work);

doublereal _starpu_zlansy_(char *norm, char *uplo, integer *n, doublecomplex *a, 
	integer *lda, doublereal *work);

doublereal _starpu_zlantb_(char *norm, char *uplo, char *diag, integer *n, integer *k, 
	 doublecomplex *ab, integer *ldab, doublereal *work);

doublereal _starpu_zlantp_(char *norm, char *uplo, char *diag, integer *n, 
	doublecomplex *ap, doublereal *work);

doublereal _starpu_zlantr_(char *norm, char *uplo, char *diag, integer *m, integer *n, 
	 doublecomplex *a, integer *lda, doublereal *work);

/* Subroutine */ int _starpu_zlapll_(integer *n, doublecomplex *x, integer *incx, 
	doublecomplex *y, integer *incy, doublereal *ssmin);

/* Subroutine */ int _starpu_zlapmt_(logical *forwrd, integer *m, integer *n, 
	doublecomplex *x, integer *ldx, integer *k);

/* Subroutine */ int _starpu_zlaqgb_(integer *m, integer *n, integer *kl, integer *ku, 
	 doublecomplex *ab, integer *ldab, doublereal *r__, doublereal *c__, 
	doublereal *rowcnd, doublereal *colcnd, doublereal *amax, char *equed);

/* Subroutine */ int _starpu_zlaqge_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublereal *r__, doublereal *c__, doublereal *rowcnd, 
	doublereal *colcnd, doublereal *amax, char *equed);

/* Subroutine */ int _starpu_zlaqhb_(char *uplo, integer *n, integer *kd, 
	doublecomplex *ab, integer *ldab, doublereal *s, doublereal *scond, 
	doublereal *amax, char *equed);

/* Subroutine */ int _starpu_zlaqhe_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, doublereal *s, doublereal *scond, doublereal *amax, 
	char *equed);

/* Subroutine */ int _starpu_zlaqhp_(char *uplo, integer *n, doublecomplex *ap, 
	doublereal *s, doublereal *scond, doublereal *amax, char *equed);

/* Subroutine */ int _starpu_zlaqp2_(integer *m, integer *n, integer *offset, 
	doublecomplex *a, integer *lda, integer *jpvt, doublecomplex *tau, 
	doublereal *vn1, doublereal *vn2, doublecomplex *work);

/* Subroutine */ int _starpu_zlaqps_(integer *m, integer *n, integer *offset, integer 
	*nb, integer *kb, doublecomplex *a, integer *lda, integer *jpvt, 
	doublecomplex *tau, doublereal *vn1, doublereal *vn2, doublecomplex *
	auxv, doublecomplex *f, integer *ldf);

/* Subroutine */ int _starpu_zlaqr0_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, doublecomplex *h__, integer *ldh, 
	doublecomplex *w, integer *iloz, integer *ihiz, doublecomplex *z__, 
	integer *ldz, doublecomplex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zlaqr1_(integer *n, doublecomplex *h__, integer *ldh, 
	doublecomplex *s1, doublecomplex *s2, doublecomplex *v);

/* Subroutine */ int _starpu_zlaqr2_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, doublecomplex *h__, 
	integer *ldh, integer *iloz, integer *ihiz, doublecomplex *z__, 
	integer *ldz, integer *ns, integer *nd, doublecomplex *sh, 
	doublecomplex *v, integer *ldv, integer *nh, doublecomplex *t, 
	integer *ldt, integer *nv, doublecomplex *wv, integer *ldwv, 
	doublecomplex *work, integer *lwork);

/* Subroutine */ int _starpu_zlaqr3_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, doublecomplex *h__, 
	integer *ldh, integer *iloz, integer *ihiz, doublecomplex *z__, 
	integer *ldz, integer *ns, integer *nd, doublecomplex *sh, 
	doublecomplex *v, integer *ldv, integer *nh, doublecomplex *t, 
	integer *ldt, integer *nv, doublecomplex *wv, integer *ldwv, 
	doublecomplex *work, integer *lwork);

/* Subroutine */ int _starpu_zlaqr4_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, doublecomplex *h__, integer *ldh, 
	doublecomplex *w, integer *iloz, integer *ihiz, doublecomplex *z__, 
	integer *ldz, doublecomplex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zlaqr5_(logical *wantt, logical *wantz, integer *kacc22, 
	integer *n, integer *ktop, integer *kbot, integer *nshfts, 
	doublecomplex *s, doublecomplex *h__, integer *ldh, integer *iloz, 
	integer *ihiz, doublecomplex *z__, integer *ldz, doublecomplex *v, 
	integer *ldv, doublecomplex *u, integer *ldu, integer *nv, 
	doublecomplex *wv, integer *ldwv, integer *nh, doublecomplex *wh, 
	integer *ldwh);

/* Subroutine */ int _starpu_zlaqsb_(char *uplo, integer *n, integer *kd, 
	doublecomplex *ab, integer *ldab, doublereal *s, doublereal *scond, 
	doublereal *amax, char *equed);

/* Subroutine */ int _starpu_zlaqsp_(char *uplo, integer *n, doublecomplex *ap, 
	doublereal *s, doublereal *scond, doublereal *amax, char *equed);

/* Subroutine */ int _starpu_zlaqsy_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, doublereal *s, doublereal *scond, doublereal *amax, 
	char *equed);

/* Subroutine */ int _starpu_zlar1v_(integer *n, integer *b1, integer *bn, doublereal 
	*lambda, doublereal *d__, doublereal *l, doublereal *ld, doublereal *
	lld, doublereal *pivmin, doublereal *gaptol, doublecomplex *z__, 
	logical *wantnc, integer *negcnt, doublereal *ztz, doublereal *mingma, 
	 integer *r__, integer *isuppz, doublereal *nrminv, doublereal *resid, 
	 doublereal *rqcorr, doublereal *work);

/* Subroutine */ int _starpu_zlar2v_(integer *n, doublecomplex *x, doublecomplex *y, 
	doublecomplex *z__, integer *incx, doublereal *c__, doublecomplex *s, 
	integer *incc);

/* Subroutine */ int _starpu_zlarcm_(integer *m, integer *n, doublereal *a, integer *
	lda, doublecomplex *b, integer *ldb, doublecomplex *c__, integer *ldc, 
	 doublereal *rwork);

/* Subroutine */ int _starpu_zlarf_(char *side, integer *m, integer *n, doublecomplex 
	*v, integer *incv, doublecomplex *tau, doublecomplex *c__, integer *
	ldc, doublecomplex *work);

/* Subroutine */ int _starpu_zlarfb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, doublecomplex *v, integer 
	*ldv, doublecomplex *t, integer *ldt, doublecomplex *c__, integer *
	ldc, doublecomplex *work, integer *ldwork);

/* Subroutine */ int _starpu_zlarfg_(integer *n, doublecomplex *alpha, doublecomplex *
	x, integer *incx, doublecomplex *tau);

/* Subroutine */ int _starpu_zlarfp_(integer *n, doublecomplex *alpha, doublecomplex *
	x, integer *incx, doublecomplex *tau);

/* Subroutine */ int _starpu_zlarft_(char *direct, char *storev, integer *n, integer *
	k, doublecomplex *v, integer *ldv, doublecomplex *tau, doublecomplex *
	t, integer *ldt);

/* Subroutine */ int _starpu_zlarfx_(char *side, integer *m, integer *n, 
	doublecomplex *v, doublecomplex *tau, doublecomplex *c__, integer *
	ldc, doublecomplex *work);

/* Subroutine */ int _starpu_zlargv_(integer *n, doublecomplex *x, integer *incx, 
	doublecomplex *y, integer *incy, doublereal *c__, integer *incc);

/* Subroutine */ int _starpu_zlarnv_(integer *idist, integer *iseed, integer *n, 
	doublecomplex *x);

/* Subroutine */ int _starpu_zlarrv_(integer *n, doublereal *vl, doublereal *vu, 
	doublereal *d__, doublereal *l, doublereal *pivmin, integer *isplit, 
	integer *m, integer *dol, integer *dou, doublereal *minrgp, 
	doublereal *rtol1, doublereal *rtol2, doublereal *w, doublereal *werr, 
	 doublereal *wgap, integer *iblock, integer *indexw, doublereal *gers, 
	 doublecomplex *z__, integer *ldz, integer *isuppz, doublereal *work, 
	integer *iwork, integer *info);

/* Subroutine */ int _starpu_zlarscl2_(integer *m, integer *n, doublereal *d__, 
	doublecomplex *x, integer *ldx);

/* Subroutine */ int _starpu_zlartg_(doublecomplex *f, doublecomplex *g, doublereal *
	cs, doublecomplex *sn, doublecomplex *r__);

/* Subroutine */ int _starpu_zlartv_(integer *n, doublecomplex *x, integer *incx, 
	doublecomplex *y, integer *incy, doublereal *c__, doublecomplex *s, 
	integer *incc);

/* Subroutine */ int _starpu_zlarz_(char *side, integer *m, integer *n, integer *l, 
	doublecomplex *v, integer *incv, doublecomplex *tau, doublecomplex *
	c__, integer *ldc, doublecomplex *work);

/* Subroutine */ int _starpu_zlarzb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, integer *l, doublecomplex 
	*v, integer *ldv, doublecomplex *t, integer *ldt, doublecomplex *c__, 
	integer *ldc, doublecomplex *work, integer *ldwork);

/* Subroutine */ int _starpu_zlarzt_(char *direct, char *storev, integer *n, integer *
	k, doublecomplex *v, integer *ldv, doublecomplex *tau, doublecomplex *
	t, integer *ldt);

/* Subroutine */ int _starpu_zlascl_(char *type__, integer *kl, integer *ku, 
	doublereal *cfrom, doublereal *cto, integer *m, integer *n, 
	doublecomplex *a, integer *lda, integer *info);

/* Subroutine */ int _starpu_zlascl2_(integer *m, integer *n, doublereal *d__, 
	doublecomplex *x, integer *ldx);

/* Subroutine */ int _starpu_zlaset_(char *uplo, integer *m, integer *n, 
	doublecomplex *alpha, doublecomplex *beta, doublecomplex *a, integer *
	lda);

/* Subroutine */ int _starpu_zlasr_(char *side, char *pivot, char *direct, integer *m, 
	 integer *n, doublereal *c__, doublereal *s, doublecomplex *a, 
	integer *lda);

/* Subroutine */ int _starpu_zlassq_(integer *n, doublecomplex *x, integer *incx, 
	doublereal *scale, doublereal *sumsq);

/* Subroutine */ int _starpu_zlaswp_(integer *n, doublecomplex *a, integer *lda, 
	integer *k1, integer *k2, integer *ipiv, integer *incx);

/* Subroutine */ int _starpu_zlasyf_(char *uplo, integer *n, integer *nb, integer *kb, 
	 doublecomplex *a, integer *lda, integer *ipiv, doublecomplex *w, 
	integer *ldw, integer *info);

/* Subroutine */ int _starpu_zlat2c_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, complex *sa, integer *ldsa, integer *info);

/* Subroutine */ int _starpu_zlatbs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, integer *kd, doublecomplex *ab, integer *ldab, 
	doublecomplex *x, doublereal *scale, doublereal *cnorm, integer *info);

/* Subroutine */ int _starpu_zlatdf_(integer *ijob, integer *n, doublecomplex *z__, 
	integer *ldz, doublecomplex *rhs, doublereal *rdsum, doublereal *
	rdscal, integer *ipiv, integer *jpiv);

/* Subroutine */ int _starpu_zlatps_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, doublecomplex *ap, doublecomplex *x, doublereal *
	scale, doublereal *cnorm, integer *info);

/* Subroutine */ int _starpu_zlatrd_(char *uplo, integer *n, integer *nb, 
	doublecomplex *a, integer *lda, doublereal *e, doublecomplex *tau, 
	doublecomplex *w, integer *ldw);

/* Subroutine */ int _starpu_zlatrs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, doublecomplex *a, integer *lda, doublecomplex *x, 
	doublereal *scale, doublereal *cnorm, integer *info);

/* Subroutine */ int _starpu_zlatrz_(integer *m, integer *n, integer *l, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work);

/* Subroutine */ int _starpu_zlatzm_(char *side, integer *m, integer *n, 
	doublecomplex *v, integer *incv, doublecomplex *tau, doublecomplex *
	c1, doublecomplex *c2, integer *ldc, doublecomplex *work);

/* Subroutine */ int _starpu_zlauu2_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, integer *info);

/* Subroutine */ int _starpu_zlauum_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, integer *info);

/* Subroutine */ int _starpu_zpbcon_(char *uplo, integer *n, integer *kd, 
	doublecomplex *ab, integer *ldab, doublereal *anorm, doublereal *
	rcond, doublecomplex *work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zpbequ_(char *uplo, integer *n, integer *kd, 
	doublecomplex *ab, integer *ldab, doublereal *s, doublereal *scond, 
	doublereal *amax, integer *info);

/* Subroutine */ int _starpu_zpbrfs_(char *uplo, integer *n, integer *kd, integer *
	nrhs, doublecomplex *ab, integer *ldab, doublecomplex *afb, integer *
	ldafb, doublecomplex *b, integer *ldb, doublecomplex *x, integer *ldx, 
	 doublereal *ferr, doublereal *berr, doublecomplex *work, doublereal *
	rwork, integer *info);

/* Subroutine */ int _starpu_zpbstf_(char *uplo, integer *n, integer *kd, 
	doublecomplex *ab, integer *ldab, integer *info);

/* Subroutine */ int _starpu_zpbsv_(char *uplo, integer *n, integer *kd, integer *
	nrhs, doublecomplex *ab, integer *ldab, doublecomplex *b, integer *
	ldb, integer *info);

/* Subroutine */ int _starpu_zpbsvx_(char *fact, char *uplo, integer *n, integer *kd, 
	integer *nrhs, doublecomplex *ab, integer *ldab, doublecomplex *afb, 
	integer *ldafb, char *equed, doublereal *s, doublecomplex *b, integer 
	*ldb, doublecomplex *x, integer *ldx, doublereal *rcond, doublereal *
	ferr, doublereal *berr, doublecomplex *work, doublereal *rwork, 
	integer *info);

/* Subroutine */ int _starpu_zpbtf2_(char *uplo, integer *n, integer *kd, 
	doublecomplex *ab, integer *ldab, integer *info);

/* Subroutine */ int _starpu_zpbtrf_(char *uplo, integer *n, integer *kd, 
	doublecomplex *ab, integer *ldab, integer *info);

/* Subroutine */ int _starpu_zpbtrs_(char *uplo, integer *n, integer *kd, integer *
	nrhs, doublecomplex *ab, integer *ldab, doublecomplex *b, integer *
	ldb, integer *info);

/* Subroutine */ int _starpu_zpftrf_(char *transr, char *uplo, integer *n, 
	doublecomplex *a, integer *info);

/* Subroutine */ int _starpu_zpftri_(char *transr, char *uplo, integer *n, 
	doublecomplex *a, integer *info);

/* Subroutine */ int _starpu_zpftrs_(char *transr, char *uplo, integer *n, integer *
	nrhs, doublecomplex *a, doublecomplex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_zpocon_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, doublereal *anorm, doublereal *rcond, doublecomplex *
	work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zpoequ_(integer *n, doublecomplex *a, integer *lda, 
	doublereal *s, doublereal *scond, doublereal *amax, integer *info);

/* Subroutine */ int _starpu_zpoequb_(integer *n, doublecomplex *a, integer *lda, 
	doublereal *s, doublereal *scond, doublereal *amax, integer *info);

/* Subroutine */ int _starpu_zporfs_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, doublecomplex *af, integer *ldaf, 
	doublecomplex *b, integer *ldb, doublecomplex *x, integer *ldx, 
	doublereal *ferr, doublereal *berr, doublecomplex *work, doublereal *
	rwork, integer *info);

/* Subroutine */ int _starpu_zporfsx_(char *uplo, char *equed, integer *n, integer *
	nrhs, doublecomplex *a, integer *lda, doublecomplex *af, integer *
	ldaf, doublereal *s, doublecomplex *b, integer *ldb, doublecomplex *x, 
	 integer *ldx, doublereal *rcond, doublereal *berr, integer *
	n_err_bnds__, doublereal *err_bnds_norm__, doublereal *
	err_bnds_comp__, integer *nparams, doublereal *params, doublecomplex *
	work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zposv_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	integer *info);

/* Subroutine */ int _starpu_zposvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublecomplex *a, integer *lda, doublecomplex *af, integer *
	ldaf, char *equed, doublereal *s, doublecomplex *b, integer *ldb, 
	doublecomplex *x, integer *ldx, doublereal *rcond, doublereal *ferr, 
	doublereal *berr, doublecomplex *work, doublereal *rwork, integer *
	info);

/* Subroutine */ int _starpu_zposvxx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublecomplex *a, integer *lda, doublecomplex *af, integer *
	ldaf, char *equed, doublereal *s, doublecomplex *b, integer *ldb, 
	doublecomplex *x, integer *ldx, doublereal *rcond, doublereal *rpvgrw, 
	 doublereal *berr, integer *n_err_bnds__, doublereal *err_bnds_norm__, 
	 doublereal *err_bnds_comp__, integer *nparams, doublereal *params, 
	doublecomplex *work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zpotf2_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, integer *info);

/* Subroutine */ int _starpu_zpotrf_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, integer *info);

/* Subroutine */ int _starpu_zpotri_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, integer *info);

/* Subroutine */ int _starpu_zpotrs_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	integer *info);

/* Subroutine */ int _starpu_zppcon_(char *uplo, integer *n, doublecomplex *ap, 
	doublereal *anorm, doublereal *rcond, doublecomplex *work, doublereal 
	*rwork, integer *info);

/* Subroutine */ int _starpu_zppequ_(char *uplo, integer *n, doublecomplex *ap, 
	doublereal *s, doublereal *scond, doublereal *amax, integer *info);

/* Subroutine */ int _starpu_zpprfs_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *ap, doublecomplex *afp, doublecomplex *b, integer *ldb, 
	 doublecomplex *x, integer *ldx, doublereal *ferr, doublereal *berr, 
	doublecomplex *work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zppsv_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *ap, doublecomplex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_zppsvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublecomplex *ap, doublecomplex *afp, char *equed, doublereal *
	s, doublecomplex *b, integer *ldb, doublecomplex *x, integer *ldx, 
	doublereal *rcond, doublereal *ferr, doublereal *berr, doublecomplex *
	work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zpptrf_(char *uplo, integer *n, doublecomplex *ap, 
	integer *info);

/* Subroutine */ int _starpu_zpptri_(char *uplo, integer *n, doublecomplex *ap, 
	integer *info);

/* Subroutine */ int _starpu_zpptrs_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *ap, doublecomplex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_zpstf2_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, integer *piv, integer *rank, doublereal *tol, 
	doublereal *work, integer *info);

/* Subroutine */ int _starpu_zpstrf_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, integer *piv, integer *rank, doublereal *tol, 
	doublereal *work, integer *info);

/* Subroutine */ int _starpu_zptcon_(integer *n, doublereal *d__, doublecomplex *e, 
	doublereal *anorm, doublereal *rcond, doublereal *rwork, integer *
	info);

/* Subroutine */ int _starpu_zpteqr_(char *compz, integer *n, doublereal *d__, 
	doublereal *e, doublecomplex *z__, integer *ldz, doublereal *work, 
	integer *info);

/* Subroutine */ int _starpu_zptrfs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *d__, doublecomplex *e, doublereal *df, doublecomplex *ef, 
	doublecomplex *b, integer *ldb, doublecomplex *x, integer *ldx, 
	doublereal *ferr, doublereal *berr, doublecomplex *work, doublereal *
	rwork, integer *info);

/* Subroutine */ int _starpu_zptsv_(integer *n, integer *nrhs, doublereal *d__, 
	doublecomplex *e, doublecomplex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_zptsvx_(char *fact, integer *n, integer *nrhs, 
	doublereal *d__, doublecomplex *e, doublereal *df, doublecomplex *ef, 
	doublecomplex *b, integer *ldb, doublecomplex *x, integer *ldx, 
	doublereal *rcond, doublereal *ferr, doublereal *berr, doublecomplex *
	work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zpttrf_(integer *n, doublereal *d__, doublecomplex *e, 
	integer *info);

/* Subroutine */ int _starpu_zpttrs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *d__, doublecomplex *e, doublecomplex *b, integer *ldb, 
	integer *info);

/* Subroutine */ int _starpu_zptts2_(integer *iuplo, integer *n, integer *nrhs, 
	doublereal *d__, doublecomplex *e, doublecomplex *b, integer *ldb);

/* Subroutine */ int _starpu_zrot_(integer *n, doublecomplex *cx, integer *incx, 
	doublecomplex *cy, integer *incy, doublereal *c__, doublecomplex *s);

/* Subroutine */ int _starpu_zspcon_(char *uplo, integer *n, doublecomplex *ap, 
	integer *ipiv, doublereal *anorm, doublereal *rcond, doublecomplex *
	work, integer *info);

/* Subroutine */ int _starpu_zspmv_(char *uplo, integer *n, doublecomplex *alpha, 
	doublecomplex *ap, doublecomplex *x, integer *incx, doublecomplex *
	beta, doublecomplex *y, integer *incy);

/* Subroutine */ int _starpu_zspr_(char *uplo, integer *n, doublecomplex *alpha, 
	doublecomplex *x, integer *incx, doublecomplex *ap);

/* Subroutine */ int _starpu_zsprfs_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *ap, doublecomplex *afp, integer *ipiv, doublecomplex *
	b, integer *ldb, doublecomplex *x, integer *ldx, doublereal *ferr, 
	doublereal *berr, doublecomplex *work, doublereal *rwork, integer *
	info);

/* Subroutine */ int _starpu_zspsv_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *ap, integer *ipiv, doublecomplex *b, integer *ldb, 
	integer *info);

/* Subroutine */ int _starpu_zspsvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublecomplex *ap, doublecomplex *afp, integer *ipiv, 
	doublecomplex *b, integer *ldb, doublecomplex *x, integer *ldx, 
	doublereal *rcond, doublereal *ferr, doublereal *berr, doublecomplex *
	work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zsptrf_(char *uplo, integer *n, doublecomplex *ap, 
	integer *ipiv, integer *info);

/* Subroutine */ int _starpu_zsptri_(char *uplo, integer *n, doublecomplex *ap, 
	integer *ipiv, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zsptrs_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *ap, integer *ipiv, doublecomplex *b, integer *ldb, 
	integer *info);

/* Subroutine */ int _starpu_zstedc_(char *compz, integer *n, doublereal *d__, 
	doublereal *e, doublecomplex *z__, integer *ldz, doublecomplex *work, 
	integer *lwork, doublereal *rwork, integer *lrwork, integer *iwork, 
	integer *liwork, integer *info);

/* Subroutine */ int _starpu_zstegr_(char *jobz, char *range, integer *n, doublereal *
	d__, doublereal *e, doublereal *vl, doublereal *vu, integer *il, 
	integer *iu, doublereal *abstol, integer *m, doublereal *w, 
	doublecomplex *z__, integer *ldz, integer *isuppz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_zstein_(integer *n, doublereal *d__, doublereal *e, 
	integer *m, doublereal *w, integer *iblock, integer *isplit, 
	doublecomplex *z__, integer *ldz, doublereal *work, integer *iwork, 
	integer *ifail, integer *info);

/* Subroutine */ int _starpu_zstemr_(char *jobz, char *range, integer *n, doublereal *
	d__, doublereal *e, doublereal *vl, doublereal *vu, integer *il, 
	integer *iu, integer *m, doublereal *w, doublecomplex *z__, integer *
	ldz, integer *nzc, integer *isuppz, logical *tryrac, doublereal *work, 
	 integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int _starpu_zsteqr_(char *compz, integer *n, doublereal *d__, 
	doublereal *e, doublecomplex *z__, integer *ldz, doublereal *work, 
	integer *info);

/* Subroutine */ int _starpu_zsycon_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, integer *ipiv, doublereal *anorm, doublereal *rcond, 
	doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zsyequb_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, doublereal *s, doublereal *scond, doublereal *amax, 
	doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zsymv_(char *uplo, integer *n, doublecomplex *alpha, 
	doublecomplex *a, integer *lda, doublecomplex *x, integer *incx, 
	doublecomplex *beta, doublecomplex *y, integer *incy);

/* Subroutine */ int _starpu_zsyr_(char *uplo, integer *n, doublecomplex *alpha, 
	doublecomplex *x, integer *incx, doublecomplex *a, integer *lda);

/* Subroutine */ int _starpu_zsyrfs_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, doublecomplex *af, integer *ldaf, 
	integer *ipiv, doublecomplex *b, integer *ldb, doublecomplex *x, 
	integer *ldx, doublereal *ferr, doublereal *berr, doublecomplex *work, 
	 doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zsyrfsx_(char *uplo, char *equed, integer *n, integer *
	nrhs, doublecomplex *a, integer *lda, doublecomplex *af, integer *
	ldaf, integer *ipiv, doublereal *s, doublecomplex *b, integer *ldb, 
	doublecomplex *x, integer *ldx, doublereal *rcond, doublereal *berr, 
	integer *n_err_bnds__, doublereal *err_bnds_norm__, doublereal *
	err_bnds_comp__, integer *nparams, doublereal *params, doublecomplex *
	work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zsysv_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, integer *ipiv, doublecomplex *b, 
	integer *ldb, doublecomplex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zsysvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublecomplex *a, integer *lda, doublecomplex *af, integer *
	ldaf, integer *ipiv, doublecomplex *b, integer *ldb, doublecomplex *x, 
	 integer *ldx, doublereal *rcond, doublereal *ferr, doublereal *berr, 
	doublecomplex *work, integer *lwork, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_zsysvxx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublecomplex *a, integer *lda, doublecomplex *af, integer *
	ldaf, integer *ipiv, char *equed, doublereal *s, doublecomplex *b, 
	integer *ldb, doublecomplex *x, integer *ldx, doublereal *rcond, 
	doublereal *rpvgrw, doublereal *berr, integer *n_err_bnds__, 
	doublereal *err_bnds_norm__, doublereal *err_bnds_comp__, integer *
	nparams, doublereal *params, doublecomplex *work, doublereal *rwork, 
	integer *info);

/* Subroutine */ int _starpu_zsytf2_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, integer *ipiv, integer *info);

/* Subroutine */ int _starpu_zsytrf_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, integer *ipiv, doublecomplex *work, integer *lwork, 
	integer *info);

/* Subroutine */ int _starpu_zsytri_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, integer *ipiv, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zsytrs_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *a, integer *lda, integer *ipiv, doublecomplex *b, 
	integer *ldb, integer *info);

/* Subroutine */ int _starpu_ztbcon_(char *norm, char *uplo, char *diag, integer *n, 
	integer *kd, doublecomplex *ab, integer *ldab, doublereal *rcond, 
	doublecomplex *work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_ztbrfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *kd, integer *nrhs, doublecomplex *ab, integer *ldab, 
	doublecomplex *b, integer *ldb, doublecomplex *x, integer *ldx, 
	doublereal *ferr, doublereal *berr, doublecomplex *work, doublereal *
	rwork, integer *info);

/* Subroutine */ int _starpu_ztbtrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *kd, integer *nrhs, doublecomplex *ab, integer *ldab, 
	doublecomplex *b, integer *ldb, integer *info);

/* Subroutine */ int _starpu_ztfsm_(char *transr, char *side, char *uplo, char *trans, 
	 char *diag, integer *m, integer *n, doublecomplex *alpha, 
	doublecomplex *a, doublecomplex *b, integer *ldb);

/* Subroutine */ int _starpu_ztftri_(char *transr, char *uplo, char *diag, integer *n, 
	 doublecomplex *a, integer *info);

/* Subroutine */ int _starpu_ztfttp_(char *transr, char *uplo, integer *n, 
	doublecomplex *arf, doublecomplex *ap, integer *info);

/* Subroutine */ int _starpu_ztfttr_(char *transr, char *uplo, integer *n, 
	doublecomplex *arf, doublecomplex *a, integer *lda, integer *info);

/* Subroutine */ int _starpu_ztgevc_(char *side, char *howmny, logical *select, 
	integer *n, doublecomplex *s, integer *lds, doublecomplex *p, integer 
	*ldp, doublecomplex *vl, integer *ldvl, doublecomplex *vr, integer *
	ldvr, integer *mm, integer *m, doublecomplex *work, doublereal *rwork, 
	 integer *info);

/* Subroutine */ int _starpu_ztgex2_(logical *wantq, logical *wantz, integer *n, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublecomplex *q, integer *ldq, doublecomplex *z__, integer *ldz, 
	integer *j1, integer *info);

/* Subroutine */ int _starpu_ztgexc_(logical *wantq, logical *wantz, integer *n, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublecomplex *q, integer *ldq, doublecomplex *z__, integer *ldz, 
	integer *ifst, integer *ilst, integer *info);

/* Subroutine */ int _starpu_ztgsen_(integer *ijob, logical *wantq, logical *wantz, 
	logical *select, integer *n, doublecomplex *a, integer *lda, 
	doublecomplex *b, integer *ldb, doublecomplex *alpha, doublecomplex *
	beta, doublecomplex *q, integer *ldq, doublecomplex *z__, integer *
	ldz, integer *m, doublereal *pl, doublereal *pr, doublereal *dif, 
	doublecomplex *work, integer *lwork, integer *iwork, integer *liwork, 
	integer *info);

/* Subroutine */ int _starpu_ztgsja_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *p, integer *n, integer *k, integer *l, doublecomplex *a, 
	integer *lda, doublecomplex *b, integer *ldb, doublereal *tola, 
	doublereal *tolb, doublereal *alpha, doublereal *beta, doublecomplex *
	u, integer *ldu, doublecomplex *v, integer *ldv, doublecomplex *q, 
	integer *ldq, doublecomplex *work, integer *ncycle, integer *info);

/* Subroutine */ int _starpu_ztgsna_(char *job, char *howmny, logical *select, 
	integer *n, doublecomplex *a, integer *lda, doublecomplex *b, integer 
	*ldb, doublecomplex *vl, integer *ldvl, doublecomplex *vr, integer *
	ldvr, doublereal *s, doublereal *dif, integer *mm, integer *m, 
	doublecomplex *work, integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_ztgsy2_(char *trans, integer *ijob, integer *m, integer *
	n, doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublecomplex *c__, integer *ldc, doublecomplex *d__, integer *ldd, 
	doublecomplex *e, integer *lde, doublecomplex *f, integer *ldf, 
	doublereal *scale, doublereal *rdsum, doublereal *rdscal, integer *
	info);

/* Subroutine */ int _starpu_ztgsyl_(char *trans, integer *ijob, integer *m, integer *
	n, doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublecomplex *c__, integer *ldc, doublecomplex *d__, integer *ldd, 
	doublecomplex *e, integer *lde, doublecomplex *f, integer *ldf, 
	doublereal *scale, doublereal *dif, doublecomplex *work, integer *
	lwork, integer *iwork, integer *info);

/* Subroutine */ int _starpu_ztpcon_(char *norm, char *uplo, char *diag, integer *n, 
	doublecomplex *ap, doublereal *rcond, doublecomplex *work, doublereal 
	*rwork, integer *info);

/* Subroutine */ int _starpu_ztprfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, doublecomplex *ap, doublecomplex *b, integer *ldb, 
	doublecomplex *x, integer *ldx, doublereal *ferr, doublereal *berr, 
	doublecomplex *work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_ztptri_(char *uplo, char *diag, integer *n, 
	doublecomplex *ap, integer *info);

/* Subroutine */ int _starpu_ztptrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, doublecomplex *ap, doublecomplex *b, integer *ldb, 
	integer *info);

/* Subroutine */ int _starpu_ztpttf_(char *transr, char *uplo, integer *n, 
	doublecomplex *ap, doublecomplex *arf, integer *info);

/* Subroutine */ int _starpu_ztpttr_(char *uplo, integer *n, doublecomplex *ap, 
	doublecomplex *a, integer *lda, integer *info);

/* Subroutine */ int _starpu_ztrcon_(char *norm, char *uplo, char *diag, integer *n, 
	doublecomplex *a, integer *lda, doublereal *rcond, doublecomplex *
	work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_ztrevc_(char *side, char *howmny, logical *select, 
	integer *n, doublecomplex *t, integer *ldt, doublecomplex *vl, 
	integer *ldvl, doublecomplex *vr, integer *ldvr, integer *mm, integer 
	*m, doublecomplex *work, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_ztrexc_(char *compq, integer *n, doublecomplex *t, 
	integer *ldt, doublecomplex *q, integer *ldq, integer *ifst, integer *
	ilst, integer *info);

/* Subroutine */ int _starpu_ztrrfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, doublecomplex *a, integer *lda, doublecomplex *b, 
	integer *ldb, doublecomplex *x, integer *ldx, doublereal *ferr, 
	doublereal *berr, doublecomplex *work, doublereal *rwork, integer *
	info);

/* Subroutine */ int _starpu_ztrsen_(char *job, char *compq, logical *select, integer 
	*n, doublecomplex *t, integer *ldt, doublecomplex *q, integer *ldq, 
	doublecomplex *w, integer *m, doublereal *s, doublereal *sep, 
	doublecomplex *work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_ztrsna_(char *job, char *howmny, logical *select, 
	integer *n, doublecomplex *t, integer *ldt, doublecomplex *vl, 
	integer *ldvl, doublecomplex *vr, integer *ldvr, doublereal *s, 
	doublereal *sep, integer *mm, integer *m, doublecomplex *work, 
	integer *ldwork, doublereal *rwork, integer *info);

/* Subroutine */ int _starpu_ztrsyl_(char *trana, char *tranb, integer *isgn, integer 
	*m, integer *n, doublecomplex *a, integer *lda, doublecomplex *b, 
	integer *ldb, doublecomplex *c__, integer *ldc, doublereal *scale, 
	integer *info);

/* Subroutine */ int _starpu_ztrti2_(char *uplo, char *diag, integer *n, 
	doublecomplex *a, integer *lda, integer *info);

/* Subroutine */ int _starpu_ztrtri_(char *uplo, char *diag, integer *n, 
	doublecomplex *a, integer *lda, integer *info);

/* Subroutine */ int _starpu_ztrtrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, doublecomplex *a, integer *lda, doublecomplex *b, 
	integer *ldb, integer *info);

/* Subroutine */ int _starpu_ztrttf_(char *transr, char *uplo, integer *n, 
	doublecomplex *a, integer *lda, doublecomplex *arf, integer *info);

/* Subroutine */ int _starpu_ztrttp_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, doublecomplex *ap, integer *info);

/* Subroutine */ int _starpu_ztzrqf_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublecomplex *tau, integer *info);

/* Subroutine */ int _starpu_ztzrzf_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublecomplex *tau, doublecomplex *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int _starpu_zung2l_(integer *m, integer *n, integer *k, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *info);

/* Subroutine */ int _starpu_zung2r_(integer *m, integer *n, integer *k, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *info);

/* Subroutine */ int _starpu_zungbr_(char *vect, integer *m, integer *n, integer *k, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zunghr_(integer *n, integer *ilo, integer *ihi, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zungl2_(integer *m, integer *n, integer *k, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *info);

/* Subroutine */ int _starpu_zunglq_(integer *m, integer *n, integer *k, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zungql_(integer *m, integer *n, integer *k, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zungqr_(integer *m, integer *n, integer *k, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zungr2_(integer *m, integer *n, integer *k, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *info);

/* Subroutine */ int _starpu_zungrq_(integer *m, integer *n, integer *k, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zungtr_(char *uplo, integer *n, doublecomplex *a, 
	integer *lda, doublecomplex *tau, doublecomplex *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int _starpu_zunm2l_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublecomplex *a, integer *lda, doublecomplex *tau, 
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zunm2r_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublecomplex *a, integer *lda, doublecomplex *tau, 
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zunmbr_(char *vect, char *side, char *trans, integer *m, 
	integer *n, integer *k, doublecomplex *a, integer *lda, doublecomplex 
	*tau, doublecomplex *c__, integer *ldc, doublecomplex *work, integer *
	lwork, integer *info);

/* Subroutine */ int _starpu_zunmhr_(char *side, char *trans, integer *m, integer *n, 
	integer *ilo, integer *ihi, doublecomplex *a, integer *lda, 
	doublecomplex *tau, doublecomplex *c__, integer *ldc, doublecomplex *
	work, integer *lwork, integer *info);

/* Subroutine */ int _starpu_zunml2_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublecomplex *a, integer *lda, doublecomplex *tau, 
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zunmlq_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublecomplex *a, integer *lda, doublecomplex *tau, 
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int _starpu_zunmql_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublecomplex *a, integer *lda, doublecomplex *tau, 
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int _starpu_zunmqr_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublecomplex *a, integer *lda, doublecomplex *tau, 
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int _starpu_zunmr2_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublecomplex *a, integer *lda, doublecomplex *tau, 
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_zunmr3_(char *side, char *trans, integer *m, integer *n, 
	integer *k, integer *l, doublecomplex *a, integer *lda, doublecomplex 
	*tau, doublecomplex *c__, integer *ldc, doublecomplex *work, integer *
	info);

/* Subroutine */ int _starpu_zunmrq_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublecomplex *a, integer *lda, doublecomplex *tau, 
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int _starpu_zunmrz_(char *side, char *trans, integer *m, integer *n, 
	integer *k, integer *l, doublecomplex *a, integer *lda, doublecomplex 
	*tau, doublecomplex *c__, integer *ldc, doublecomplex *work, integer *
	lwork, integer *info);

/* Subroutine */ int _starpu_zunmtr_(char *side, char *uplo, char *trans, integer *m, 
	integer *n, doublecomplex *a, integer *lda, doublecomplex *tau, 
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int _starpu_zupgtr_(char *uplo, integer *n, doublecomplex *ap, 
	doublecomplex *tau, doublecomplex *q, integer *ldq, doublecomplex *
	work, integer *info);

/* Subroutine */ int _starpu_zupmtr_(char *side, char *uplo, char *trans, integer *m, 
	integer *n, doublecomplex *ap, doublecomplex *tau, doublecomplex *c__, 
	 integer *ldc, doublecomplex *work, integer *info);

/* Subroutine */ int _starpu_dlamc1_(integer *beta, integer *t, logical *rnd, logical 
	*ieee1);

doublereal _starpu_dsecnd_();

/* Subroutine */ int _starpu_ilaver_(integer *vers_major__, integer *vers_minor__, 
	integer *vers_patch__);

logical _starpu_lsame_(char *ca, char *cb);

doublereal _starpu_second_();

doublereal _starpu_slamch_(char *cmach);

/* Subroutine */ int _starpu_slamc1_(integer *beta, integer *t, logical *rnd, logical 
	*ieee1);

/* Subroutine */ int _starpu_slamc2_(integer *beta, integer *t, logical *rnd, real *
		    eps, integer *emin, real *rmin, integer *emax, real *rmax);

doublereal _starpu_slamc3_(real *a, real *b);

/* Subroutine */ int _starpu_slamc4_(integer *emin, real *start, integer *base);

/* Subroutine */ int _starpu_slamc5_(integer *beta, integer *p, integer *emin,
		    logical *ieee, integer *emax, real *rmax);


doublereal _starpu_dlamch_(char *cmach);

/* Subroutine */ int _starpu_dlamc1_(integer *beta, integer *t, logical *rnd, logical
		    *ieee1);

/* Subroutine */ int _starpu_dlamc2_(integer *beta, integer *t, logical *rnd,
		    doublereal *eps, integer *emin, doublereal *rmin, integer *emax,
			    doublereal *rmax);

doublereal _starpu_dlamc3_(doublereal *a, doublereal *b);

/* Subroutine */ int _starpu_dlamc4_(integer *emin, doublereal *start, integer *base);

/* Subroutine */ int _starpu_dlamc5_(integer *beta, integer *p, integer *emin,
		    logical *ieee, integer *emax, doublereal *rmax);

integer _starpu_ilaenv_(integer *ispec, char *name__, char *opts, integer *n1, 
	integer *n2, integer *n3, integer *n4);

#ifdef __cplusplus
}
#endif


#endif /* __CLAPACK_H */
