/* CLAPACK 3.0 BLAS wrapper macros
 * Feb 5, 2000
 */

#ifndef __BLASWRAP_H
#define __BLASWRAP_H

#ifndef NO_BLAS_WRAP
 
/* BLAS1 routines */
#define _starpu_srotg_ f2c_srotg
#define _starpu_crotg_ f2c_crotg
#define _starpu_drotg_ f2c_drotg
#define _starpu_zrotg_ f2c_zrotg
#define _starpu_srotmg_ f2c_srotmg
#define _starpu_drotmg_ f2c_drotmg
#define _starpu_srot_ f2c_srot
#define _starpu_drot_ f2c_drot
#define _starpu_srotm_ f2c_srotm
#define _starpu_drotm_ f2c_drotm
#define _starpu_sswap_ f2c_sswap
#define _starpu_dswap_ f2c_dswap
#define _starpu_cswap_ f2c_cswap
#define _starpu_zswap_ f2c_zswap
#define _starpu_sscal_ f2c_sscal
#define _starpu_dscal_ f2c_dscal
#define _starpu_cscal_ f2c_cscal
#define _starpu_zscal_ f2c_zscal
#define _starpu_csscal_ f2c_csscal
#define _starpu_zdscal_ f2c_zdscal
#define _starpu_scopy_ f2c_scopy
#define _starpu_dcopy_ f2c_dcopy
#define _starpu_ccopy_ f2c_ccopy
#define _starpu_zcopy_ f2c_zcopy
#define _starpu_saxpy_ f2c_saxpy
#define _starpu_daxpy_ f2c_daxpy
#define _starpu_caxpy_ f2c_caxpy
#define _starpu_zaxpy_ f2c_zaxpy
#define _starpu_sdot_ f2c_sdot
#define _starpu_ddot_ f2c_ddot
#define _starpu_cdotu_ f2c_cdotu
#define _starpu_zdotu_ f2c_zdotu
#define _starpu_cdotc_ f2c_cdotc
#define _starpu_zdotc_ f2c_zdotc
#define _starpu_snrm2_ f2c_snrm2
#define _starpu_dnrm2_ f2c_dnrm2
#define _starpu_scnrm2_ f2c_scnrm2
#define _starpu_dznrm2_ f2c_dznrm2
#define _starpu_sasum_ f2c_sasum
#define _starpu_dasum_ f2c_dasum
#define _starpu_scasum_ f2c_scasum
#define _starpu_dzasum_ f2c_dzasum
#define _starpu_isamax_ f2c_isamax
#define _starpu_idamax_ f2c_idamax
#define _starpu_icamax_ f2c_icamax
#define _starpu_izamax_ f2c_izamax
 
/* BLAS2 routines */
#define _starpu_sgemv_ f2c_sgemv
#define _starpu_dgemv_ f2c_dgemv
#define _starpu_cgemv_ f2c_cgemv
#define _starpu_zgemv_ f2c_zgemv
#define _starpu_sgbmv_ f2c_sgbmv
#define _starpu_dgbmv_ f2c_dgbmv
#define _starpu_cgbmv_ f2c_cgbmv
#define _starpu_zgbmv_ f2c_zgbmv
#define _starpu_chemv_ f2c_chemv
#define _starpu_zhemv_ f2c_zhemv
#define _starpu_chbmv_ f2c_chbmv
#define _starpu_zhbmv_ f2c_zhbmv
#define _starpu_chpmv_ f2c_chpmv
#define _starpu_zhpmv_ f2c_zhpmv
#define _starpu_ssymv_ f2c_ssymv
#define _starpu_dsymv_ f2c_dsymv
#define _starpu_ssbmv_ f2c_ssbmv
#define _starpu_dsbmv_ f2c_dsbmv
#define _starpu_sspmv_ f2c_sspmv
#define _starpu_dspmv_ f2c_dspmv
#define _starpu_strmv_ f2c_strmv
#define _starpu_dtrmv_ f2c_dtrmv
#define _starpu_ctrmv_ f2c_ctrmv
#define _starpu_ztrmv_ f2c_ztrmv
#define _starpu_stbmv_ f2c_stbmv
#define _starpu_dtbmv_ f2c_dtbmv
#define _starpu_ctbmv_ f2c_ctbmv
#define _starpu_ztbmv_ f2c_ztbmv
#define _starpu_stpmv_ f2c_stpmv
#define _starpu_dtpmv_ f2c_dtpmv
#define _starpu_ctpmv_ f2c_ctpmv
#define _starpu_ztpmv_ f2c_ztpmv
#define _starpu_strsv_ f2c_strsv
#define _starpu_dtrsv_ f2c_dtrsv
#define _starpu_ctrsv_ f2c_ctrsv
#define _starpu_ztrsv_ f2c_ztrsv
#define _starpu_stbsv_ f2c_stbsv
#define _starpu_dtbsv_ f2c_dtbsv
#define _starpu_ctbsv_ f2c_ctbsv
#define _starpu_ztbsv_ f2c_ztbsv
#define _starpu_stpsv_ f2c_stpsv
#define _starpu_dtpsv_ f2c_dtpsv
#define _starpu_ctpsv_ f2c_ctpsv
#define _starpu_ztpsv_ f2c_ztpsv
#define _starpu_sger_ f2c_sger
#define _starpu_dger_ f2c_dger
#define _starpu_cgeru_ f2c_cgeru
#define _starpu_zgeru_ f2c_zgeru
#define _starpu_cgerc_ f2c_cgerc
#define _starpu_zgerc_ f2c_zgerc
#define _starpu_cher_ f2c_cher
#define _starpu_zher_ f2c_zher
#define _starpu_chpr_ f2c_chpr
#define _starpu_zhpr_ f2c_zhpr
#define _starpu_cher2_ f2c_cher2
#define _starpu_zher2_ f2c_zher2
#define _starpu_chpr2_ f2c_chpr2
#define _starpu_zhpr2_ f2c_zhpr2
#define _starpu_ssyr_ f2c_ssyr
#define _starpu_dsyr_ f2c_dsyr
#define _starpu_sspr_ f2c_sspr
#define _starpu_dspr_ f2c_dspr
#define _starpu_ssyr2_ f2c_ssyr2
#define _starpu_dsyr2_ f2c_dsyr2
#define _starpu_sspr2_ f2c_sspr2
#define _starpu_dspr2_ f2c_dspr2
 
/* BLAS3 routines */
#define _starpu_sgemm_ f2c_sgemm
#define _starpu_dgemm_ f2c_dgemm
#define _starpu_cgemm_ f2c_cgemm
#define _starpu_zgemm_ f2c_zgemm
#define _starpu_ssymm_ f2c_ssymm
#define _starpu_dsymm_ f2c_dsymm
#define _starpu_csymm_ f2c_csymm
#define _starpu_zsymm_ f2c_zsymm
#define _starpu_chemm_ f2c_chemm
#define _starpu_zhemm_ f2c_zhemm
#define _starpu_ssyrk_ f2c_ssyrk
#define _starpu_dsyrk_ f2c_dsyrk
#define _starpu_csyrk_ f2c_csyrk
#define _starpu_zsyrk_ f2c_zsyrk
#define _starpu_cherk_ f2c_cherk
#define _starpu_zherk_ f2c_zherk
#define _starpu_ssyr2k_ f2c_ssyr2k
#define _starpu_dsyr2k_ f2c_dsyr2k
#define _starpu_csyr2k_ f2c_csyr2k
#define _starpu_zsyr2k_ f2c_zsyr2k
#define _starpu_cher2k_ f2c_cher2k
#define _starpu_zher2k_ f2c_zher2k
#define _starpu_strmm_ f2c_strmm
#define _starpu_dtrmm_ f2c_dtrmm
#define _starpu_ctrmm_ f2c_ctrmm
#define _starpu_ztrmm_ f2c_ztrmm
#define _starpu_strsm_ f2c_strsm
#define _starpu_dtrsm_ f2c_dtrsm
#define _starpu_ctrsm_ f2c_ctrsm
#define _starpu_ztrsm_ f2c_ztrsm

#endif /* NO_BLAS_WRAP */

#endif /* __BLASWRAP_H */
