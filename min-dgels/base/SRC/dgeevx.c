/* dgeevx.f -- translated by f2c (version 20061008).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"
#include "blaswrap.h"

/* Table of constant values */

static integer c__1 = 1;
static integer c__0 = 0;
static integer c_n1 = -1;

/* Subroutine */ int _starpu_dgeevx_(char *balanc, char *jobvl, char *jobvr, char *
	sense, integer *n, doublereal *a, integer *lda, doublereal *wr, 
	doublereal *wi, doublereal *vl, integer *ldvl, doublereal *vr, 
	integer *ldvr, integer *ilo, integer *ihi, doublereal *scale, 
	doublereal *abnrm, doublereal *rconde, doublereal *rcondv, doublereal 
	*work, integer *lwork, integer *iwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, vl_dim1, vl_offset, vr_dim1, vr_offset, i__1, 
	    i__2, i__3;
    doublereal d__1, d__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__, k;
    doublereal r__, cs, sn;
    char job[1];
    doublereal scl, dum[1], eps;
    char side[1];
    doublereal anrm;
    integer ierr, itau;
    extern /* Subroutine */ int _starpu_drot_(integer *, doublereal *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *);
    integer iwrk, nout;
    extern doublereal _starpu_dnrm2_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int _starpu_dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    integer icond;
    extern logical _starpu_lsame_(char *, char *);
    extern doublereal _starpu_dlapy2_(doublereal *, doublereal *);
    extern /* Subroutine */ int _starpu_dlabad_(doublereal *, doublereal *), _starpu_dgebak_(
	    char *, char *, integer *, integer *, integer *, doublereal *, 
	    integer *, doublereal *, integer *, integer *), 
	    _starpu_dgebal_(char *, integer *, doublereal *, integer *, integer *, 
	    integer *, doublereal *, integer *);
    logical scalea;
    extern doublereal _starpu_dlamch_(char *);
    doublereal cscale;
    extern doublereal _starpu_dlange_(char *, integer *, integer *, doublereal *, 
	    integer *, doublereal *);
    extern /* Subroutine */ int _starpu_dgehrd_(integer *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *, integer *, 
	    integer *), _starpu_dlascl_(char *, integer *, integer *, doublereal *, 
	    doublereal *, integer *, integer *, doublereal *, integer *, 
	    integer *);
    extern integer _starpu_idamax_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int _starpu_dlacpy_(char *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, integer *), 
	    _starpu_dlartg_(doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *), _starpu_xerbla_(char *, integer *);
    logical select[1];
    extern integer _starpu_ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    doublereal bignum;
    extern /* Subroutine */ int _starpu_dorghr_(integer *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *, integer *, 
	    integer *), _starpu_dhseqr_(char *, char *, integer *, integer *, integer 
	    *, doublereal *, integer *, doublereal *, doublereal *, 
	    doublereal *, integer *, doublereal *, integer *, integer *), _starpu_dtrevc_(char *, char *, logical *, integer *, 
	    doublereal *, integer *, doublereal *, integer *, doublereal *, 
	    integer *, integer *, integer *, doublereal *, integer *), _starpu_dtrsna_(char *, char *, logical *, integer *, doublereal 
	    *, integer *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, doublereal *, integer *, integer *, doublereal *, 
	    integer *, integer *, integer *);
    integer minwrk, maxwrk;
    logical wantvl, wntsnb;
    integer hswork;
    logical wntsne;
    doublereal smlnum;
    logical lquery, wantvr, wntsnn, wntsnv;


/*  -- LAPACK driver routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DGEEVX computes for an N-by-N real nonsymmetric matrix A, the */
/*  eigenvalues and, optionally, the left and/or right eigenvectors. */

/*  Optionally also, it computes a balancing transformation to improve */
/*  the conditioning of the eigenvalues and eigenvectors (ILO, IHI, */
/*  SCALE, and ABNRM), reciprocal condition numbers for the eigenvalues */
/*  (RCONDE), and reciprocal condition numbers for the right */
/*  eigenvectors (RCONDV). */

/*  The right eigenvector v(j) of A satisfies */
/*                   A * v(j) = lambda(j) * v(j) */
/*  where lambda(j) is its eigenvalue. */
/*  The left eigenvector u(j) of A satisfies */
/*                u(j)**H * A = lambda(j) * u(j)**H */
/*  where u(j)**H denotes the conjugate transpose of u(j). */

/*  The computed eigenvectors are normalized to have Euclidean norm */
/*  equal to 1 and largest component real. */

/*  Balancing a matrix means permuting the rows and columns to make it */
/*  more nearly upper triangular, and applying a diagonal similarity */
/*  transformation D * A * D**(-1), where D is a diagonal matrix, to */
/*  make its rows and columns closer in norm and the condition numbers */
/*  of its eigenvalues and eigenvectors smaller.  The computed */
/*  reciprocal condition numbers correspond to the balanced matrix. */
/*  Permuting rows and columns will not change the condition numbers */
/*  (in exact arithmetic) but diagonal scaling will.  For further */
/*  explanation of balancing, see section 4.10.2 of the LAPACK */
/*  Users' Guide. */

/*  Arguments */
/*  ========= */

/*  BALANC  (input) CHARACTER*1 */
/*          Indicates how the input matrix should be diagonally scaled */
/*          and/or permuted to improve the conditioning of its */
/*          eigenvalues. */
/*          = 'N': Do not diagonally scale or permute; */
/*          = 'P': Perform permutations to make the matrix more nearly */
/*                 upper triangular. Do not diagonally scale; */
/*          = 'S': Diagonally scale the matrix, i.e. replace A by */
/*                 D*A*D**(-1), where D is a diagonal matrix chosen */
/*                 to make the rows and columns of A more equal in */
/*                 norm. Do not permute; */
/*          = 'B': Both diagonally scale and permute A. */

/*          Computed reciprocal condition numbers will be for the matrix */
/*          after balancing and/or permuting. Permuting does not change */
/*          condition numbers (in exact arithmetic), but balancing does. */

/*  JOBVL   (input) CHARACTER*1 */
/*          = 'N': left eigenvectors of A are not computed; */
/*          = 'V': left eigenvectors of A are computed. */
/*          If SENSE = 'E' or 'B', JOBVL must = 'V'. */

/*  JOBVR   (input) CHARACTER*1 */
/*          = 'N': right eigenvectors of A are not computed; */
/*          = 'V': right eigenvectors of A are computed. */
/*          If SENSE = 'E' or 'B', JOBVR must = 'V'. */

/*  SENSE   (input) CHARACTER*1 */
/*          Determines which reciprocal condition numbers are computed. */
/*          = 'N': None are computed; */
/*          = 'E': Computed for eigenvalues only; */
/*          = 'V': Computed for right eigenvectors only; */
/*          = 'B': Computed for eigenvalues and right eigenvectors. */

/*          If SENSE = 'E' or 'B', both left and right eigenvectors */
/*          must also be computed (JOBVL = 'V' and JOBVR = 'V'). */

/*  N       (input) INTEGER */
/*          The order of the matrix A. N >= 0. */

/*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N) */
/*          On entry, the N-by-N matrix A. */
/*          On exit, A has been overwritten.  If JOBVL = 'V' or */
/*          JOBVR = 'V', A contains the real Schur form of the balanced */
/*          version of the input matrix A. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  WR      (output) DOUBLE PRECISION array, dimension (N) */
/*  WI      (output) DOUBLE PRECISION array, dimension (N) */
/*          WR and WI contain the real and imaginary parts, */
/*          respectively, of the computed eigenvalues.  Complex */
/*          conjugate pairs of eigenvalues will appear consecutively */
/*          with the eigenvalue having the positive imaginary part */
/*          first. */

/*  VL      (output) DOUBLE PRECISION array, dimension (LDVL,N) */
/*          If JOBVL = 'V', the left eigenvectors u(j) are stored one */
/*          after another in the columns of VL, in the same order */
/*          as their eigenvalues. */
/*          If JOBVL = 'N', VL is not referenced. */
/*          If the j-th eigenvalue is real, then u(j) = VL(:,j), */
/*          the j-th column of VL. */
/*          If the j-th and (j+1)-st eigenvalues form a complex */
/*          conjugate pair, then u(j) = VL(:,j) + i*VL(:,j+1) and */
/*          u(j+1) = VL(:,j) - i*VL(:,j+1). */

/*  LDVL    (input) INTEGER */
/*          The leading dimension of the array VL.  LDVL >= 1; if */
/*          JOBVL = 'V', LDVL >= N. */

/*  VR      (output) DOUBLE PRECISION array, dimension (LDVR,N) */
/*          If JOBVR = 'V', the right eigenvectors v(j) are stored one */
/*          after another in the columns of VR, in the same order */
/*          as their eigenvalues. */
/*          If JOBVR = 'N', VR is not referenced. */
/*          If the j-th eigenvalue is real, then v(j) = VR(:,j), */
/*          the j-th column of VR. */
/*          If the j-th and (j+1)-st eigenvalues form a complex */
/*          conjugate pair, then v(j) = VR(:,j) + i*VR(:,j+1) and */
/*          v(j+1) = VR(:,j) - i*VR(:,j+1). */

/*  LDVR    (input) INTEGER */
/*          The leading dimension of the array VR.  LDVR >= 1, and if */
/*          JOBVR = 'V', LDVR >= N. */

/*  ILO     (output) INTEGER */
/*  IHI     (output) INTEGER */
/*          ILO and IHI are integer values determined when A was */
/*          balanced.  The balanced A(i,j) = 0 if I > J and */
/*          J = 1,...,ILO-1 or I = IHI+1,...,N. */

/*  SCALE   (output) DOUBLE PRECISION array, dimension (N) */
/*          Details of the permutations and scaling factors applied */
/*          when balancing A.  If P(j) is the index of the row and column */
/*          interchanged with row and column j, and D(j) is the scaling */
/*          factor applied to row and column j, then */
/*          SCALE(J) = P(J),    for J = 1,...,ILO-1 */
/*                   = D(J),    for J = ILO,...,IHI */
/*                   = P(J)     for J = IHI+1,...,N. */
/*          The order in which the interchanges are made is N to IHI+1, */
/*          then 1 to ILO-1. */

/*  ABNRM   (output) DOUBLE PRECISION */
/*          The one-norm of the balanced matrix (the maximum */
/*          of the sum of absolute values of elements of any column). */

/*  RCONDE  (output) DOUBLE PRECISION array, dimension (N) */
/*          RCONDE(j) is the reciprocal condition number of the j-th */
/*          eigenvalue. */

/*  RCONDV  (output) DOUBLE PRECISION array, dimension (N) */
/*          RCONDV(j) is the reciprocal condition number of the j-th */
/*          right eigenvector. */

/*  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK.   If SENSE = 'N' or 'E', */
/*          LWORK >= max(1,2*N), and if JOBVL = 'V' or JOBVR = 'V', */
/*          LWORK >= 3*N.  If SENSE = 'V' or 'B', LWORK >= N*(N+6). */
/*          For good performance, LWORK must generally be larger. */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal size of the WORK array, returns */
/*          this value as the first entry of the WORK array, and no error */
/*          message related to LWORK is issued by XERBLA. */

/*  IWORK   (workspace) INTEGER array, dimension (2*N-2) */
/*          If SENSE = 'N' or 'E', not referenced. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/*          > 0:  if INFO = i, the QR algorithm failed to compute all the */
/*                eigenvalues, and no eigenvectors or condition numbers */
/*                have been computed; elements 1:ILO-1 and i+1:N of WR */
/*                and WI contain eigenvalues which have converged. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Local Arrays .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --wr;
    --wi;
    vl_dim1 = *ldvl;
    vl_offset = 1 + vl_dim1;
    vl -= vl_offset;
    vr_dim1 = *ldvr;
    vr_offset = 1 + vr_dim1;
    vr -= vr_offset;
    --scale;
    --rconde;
    --rcondv;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;
    lquery = *lwork == -1;
    wantvl = _starpu_lsame_(jobvl, "V");
    wantvr = _starpu_lsame_(jobvr, "V");
    wntsnn = _starpu_lsame_(sense, "N");
    wntsne = _starpu_lsame_(sense, "E");
    wntsnv = _starpu_lsame_(sense, "V");
    wntsnb = _starpu_lsame_(sense, "B");
    if (! (_starpu_lsame_(balanc, "N") || _starpu_lsame_(balanc, "S") || _starpu_lsame_(balanc, "P") 
	    || _starpu_lsame_(balanc, "B"))) {
	*info = -1;
    } else if (! wantvl && ! _starpu_lsame_(jobvl, "N")) {
	*info = -2;
    } else if (! wantvr && ! _starpu_lsame_(jobvr, "N")) {
	*info = -3;
    } else if (! (wntsnn || wntsne || wntsnb || wntsnv) || (wntsne || wntsnb) 
	    && ! (wantvl && wantvr)) {
	*info = -4;
    } else if (*n < 0) {
	*info = -5;
    } else if (*lda < max(1,*n)) {
	*info = -7;
    } else if (*ldvl < 1 || wantvl && *ldvl < *n) {
	*info = -11;
    } else if (*ldvr < 1 || wantvr && *ldvr < *n) {
	*info = -13;
    }

/*     Compute workspace */
/*      (Note: Comments in the code beginning "Workspace:" describe the */
/*       minimal amount of workspace needed at that point in the code, */
/*       as well as the preferred amount for good performance. */
/*       NB refers to the optimal block size for the immediately */
/*       following subroutine, as returned by ILAENV. */
/*       HSWORK refers to the workspace preferred by DHSEQR, as */
/*       calculated below. HSWORK is computed assuming ILO=1 and IHI=N, */
/*       the worst case.) */

    if (*info == 0) {
	if (*n == 0) {
	    minwrk = 1;
	    maxwrk = 1;
	} else {
	    maxwrk = *n + *n * _starpu_ilaenv_(&c__1, "DGEHRD", " ", n, &c__1, n, &
		    c__0);

	    if (wantvl) {
		_starpu_dhseqr_("S", "V", n, &c__1, n, &a[a_offset], lda, &wr[1], &wi[
			1], &vl[vl_offset], ldvl, &work[1], &c_n1, info);
	    } else if (wantvr) {
		_starpu_dhseqr_("S", "V", n, &c__1, n, &a[a_offset], lda, &wr[1], &wi[
			1], &vr[vr_offset], ldvr, &work[1], &c_n1, info);
	    } else {
		if (wntsnn) {
		    _starpu_dhseqr_("E", "N", n, &c__1, n, &a[a_offset], lda, &wr[1], 
			    &wi[1], &vr[vr_offset], ldvr, &work[1], &c_n1, 
			    info);
		} else {
		    _starpu_dhseqr_("S", "N", n, &c__1, n, &a[a_offset], lda, &wr[1], 
			    &wi[1], &vr[vr_offset], ldvr, &work[1], &c_n1, 
			    info);
		}
	    }
	    hswork = (integer) work[1];

	    if (! wantvl && ! wantvr) {
		minwrk = *n << 1;
		if (! wntsnn) {
/* Computing MAX */
		    i__1 = minwrk, i__2 = *n * *n + *n * 6;
		    minwrk = max(i__1,i__2);
		}
		maxwrk = max(maxwrk,hswork);
		if (! wntsnn) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *n * *n + *n * 6;
		    maxwrk = max(i__1,i__2);
		}
	    } else {
		minwrk = *n * 3;
		if (! wntsnn && ! wntsne) {
/* Computing MAX */
		    i__1 = minwrk, i__2 = *n * *n + *n * 6;
		    minwrk = max(i__1,i__2);
		}
		maxwrk = max(maxwrk,hswork);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *n + (*n - 1) * _starpu_ilaenv_(&c__1, "DORGHR", 
			 " ", n, &c__1, n, &c_n1);
		maxwrk = max(i__1,i__2);
		if (! wntsnn && ! wntsne) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *n * *n + *n * 6;
		    maxwrk = max(i__1,i__2);
		}
/* Computing MAX */
		i__1 = maxwrk, i__2 = *n * 3;
		maxwrk = max(i__1,i__2);
	    }
	    maxwrk = max(maxwrk,minwrk);
	}
	work[1] = (doublereal) maxwrk;

	if (*lwork < minwrk && ! lquery) {
	    *info = -21;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	_starpu_xerbla_("DGEEVX", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Get machine constants */

    eps = _starpu_dlamch_("P");
    smlnum = _starpu_dlamch_("S");
    bignum = 1. / smlnum;
    _starpu_dlabad_(&smlnum, &bignum);
    smlnum = sqrt(smlnum) / eps;
    bignum = 1. / smlnum;

/*     Scale A if max element outside range [SMLNUM,BIGNUM] */

    icond = 0;
    anrm = _starpu_dlange_("M", n, n, &a[a_offset], lda, dum);
    scalea = FALSE_;
    if (anrm > 0. && anrm < smlnum) {
	scalea = TRUE_;
	cscale = smlnum;
    } else if (anrm > bignum) {
	scalea = TRUE_;
	cscale = bignum;
    }
    if (scalea) {
	_starpu_dlascl_("G", &c__0, &c__0, &anrm, &cscale, n, n, &a[a_offset], lda, &
		ierr);
    }

/*     Balance the matrix and compute ABNRM */

    _starpu_dgebal_(balanc, n, &a[a_offset], lda, ilo, ihi, &scale[1], &ierr);
    *abnrm = _starpu_dlange_("1", n, n, &a[a_offset], lda, dum);
    if (scalea) {
	dum[0] = *abnrm;
	_starpu_dlascl_("G", &c__0, &c__0, &cscale, &anrm, &c__1, &c__1, dum, &c__1, &
		ierr);
	*abnrm = dum[0];
    }

/*     Reduce to upper Hessenberg form */
/*     (Workspace: need 2*N, prefer N+N*NB) */

    itau = 1;
    iwrk = itau + *n;
    i__1 = *lwork - iwrk + 1;
    _starpu_dgehrd_(n, ilo, ihi, &a[a_offset], lda, &work[itau], &work[iwrk], &i__1, &
	    ierr);

    if (wantvl) {

/*        Want left eigenvectors */
/*        Copy Householder vectors to VL */

	*(unsigned char *)side = 'L';
	_starpu_dlacpy_("L", n, n, &a[a_offset], lda, &vl[vl_offset], ldvl)
		;

/*        Generate orthogonal matrix in VL */
/*        (Workspace: need 2*N-1, prefer N+(N-1)*NB) */

	i__1 = *lwork - iwrk + 1;
	_starpu_dorghr_(n, ilo, ihi, &vl[vl_offset], ldvl, &work[itau], &work[iwrk], &
		i__1, &ierr);

/*        Perform QR iteration, accumulating Schur vectors in VL */
/*        (Workspace: need 1, prefer HSWORK (see comments) ) */

	iwrk = itau;
	i__1 = *lwork - iwrk + 1;
	_starpu_dhseqr_("S", "V", n, ilo, ihi, &a[a_offset], lda, &wr[1], &wi[1], &vl[
		vl_offset], ldvl, &work[iwrk], &i__1, info);

	if (wantvr) {

/*           Want left and right eigenvectors */
/*           Copy Schur vectors to VR */

	    *(unsigned char *)side = 'B';
	    _starpu_dlacpy_("F", n, n, &vl[vl_offset], ldvl, &vr[vr_offset], ldvr);
	}

    } else if (wantvr) {

/*        Want right eigenvectors */
/*        Copy Householder vectors to VR */

	*(unsigned char *)side = 'R';
	_starpu_dlacpy_("L", n, n, &a[a_offset], lda, &vr[vr_offset], ldvr)
		;

/*        Generate orthogonal matrix in VR */
/*        (Workspace: need 2*N-1, prefer N+(N-1)*NB) */

	i__1 = *lwork - iwrk + 1;
	_starpu_dorghr_(n, ilo, ihi, &vr[vr_offset], ldvr, &work[itau], &work[iwrk], &
		i__1, &ierr);

/*        Perform QR iteration, accumulating Schur vectors in VR */
/*        (Workspace: need 1, prefer HSWORK (see comments) ) */

	iwrk = itau;
	i__1 = *lwork - iwrk + 1;
	_starpu_dhseqr_("S", "V", n, ilo, ihi, &a[a_offset], lda, &wr[1], &wi[1], &vr[
		vr_offset], ldvr, &work[iwrk], &i__1, info);

    } else {

/*        Compute eigenvalues only */
/*        If condition numbers desired, compute Schur form */

	if (wntsnn) {
	    *(unsigned char *)job = 'E';
	} else {
	    *(unsigned char *)job = 'S';
	}

/*        (Workspace: need 1, prefer HSWORK (see comments) ) */

	iwrk = itau;
	i__1 = *lwork - iwrk + 1;
	_starpu_dhseqr_(job, "N", n, ilo, ihi, &a[a_offset], lda, &wr[1], &wi[1], &vr[
		vr_offset], ldvr, &work[iwrk], &i__1, info);
    }

/*     If INFO > 0 from DHSEQR, then quit */

    if (*info > 0) {
	goto L50;
    }

    if (wantvl || wantvr) {

/*        Compute left and/or right eigenvectors */
/*        (Workspace: need 3*N) */

	_starpu_dtrevc_(side, "B", select, n, &a[a_offset], lda, &vl[vl_offset], ldvl, 
		 &vr[vr_offset], ldvr, n, &nout, &work[iwrk], &ierr);
    }

/*     Compute condition numbers if desired */
/*     (Workspace: need N*N+6*N unless SENSE = 'E') */

    if (! wntsnn) {
	_starpu_dtrsna_(sense, "A", select, n, &a[a_offset], lda, &vl[vl_offset], 
		ldvl, &vr[vr_offset], ldvr, &rconde[1], &rcondv[1], n, &nout, 
		&work[iwrk], n, &iwork[1], &icond);
    }

    if (wantvl) {

/*        Undo balancing of left eigenvectors */

	_starpu_dgebak_(balanc, "L", n, ilo, ihi, &scale[1], n, &vl[vl_offset], ldvl, 
		&ierr);

/*        Normalize left eigenvectors and make largest component real */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (wi[i__] == 0.) {
		scl = 1. / _starpu_dnrm2_(n, &vl[i__ * vl_dim1 + 1], &c__1);
		_starpu_dscal_(n, &scl, &vl[i__ * vl_dim1 + 1], &c__1);
	    } else if (wi[i__] > 0.) {
		d__1 = _starpu_dnrm2_(n, &vl[i__ * vl_dim1 + 1], &c__1);
		d__2 = _starpu_dnrm2_(n, &vl[(i__ + 1) * vl_dim1 + 1], &c__1);
		scl = 1. / _starpu_dlapy2_(&d__1, &d__2);
		_starpu_dscal_(n, &scl, &vl[i__ * vl_dim1 + 1], &c__1);
		_starpu_dscal_(n, &scl, &vl[(i__ + 1) * vl_dim1 + 1], &c__1);
		i__2 = *n;
		for (k = 1; k <= i__2; ++k) {
/* Computing 2nd power */
		    d__1 = vl[k + i__ * vl_dim1];
/* Computing 2nd power */
		    d__2 = vl[k + (i__ + 1) * vl_dim1];
		    work[k] = d__1 * d__1 + d__2 * d__2;
/* L10: */
		}
		k = _starpu_idamax_(n, &work[1], &c__1);
		_starpu_dlartg_(&vl[k + i__ * vl_dim1], &vl[k + (i__ + 1) * vl_dim1], 
			&cs, &sn, &r__);
		_starpu_drot_(n, &vl[i__ * vl_dim1 + 1], &c__1, &vl[(i__ + 1) * 
			vl_dim1 + 1], &c__1, &cs, &sn);
		vl[k + (i__ + 1) * vl_dim1] = 0.;
	    }
/* L20: */
	}
    }

    if (wantvr) {

/*        Undo balancing of right eigenvectors */

	_starpu_dgebak_(balanc, "R", n, ilo, ihi, &scale[1], n, &vr[vr_offset], ldvr, 
		&ierr);

/*        Normalize right eigenvectors and make largest component real */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (wi[i__] == 0.) {
		scl = 1. / _starpu_dnrm2_(n, &vr[i__ * vr_dim1 + 1], &c__1);
		_starpu_dscal_(n, &scl, &vr[i__ * vr_dim1 + 1], &c__1);
	    } else if (wi[i__] > 0.) {
		d__1 = _starpu_dnrm2_(n, &vr[i__ * vr_dim1 + 1], &c__1);
		d__2 = _starpu_dnrm2_(n, &vr[(i__ + 1) * vr_dim1 + 1], &c__1);
		scl = 1. / _starpu_dlapy2_(&d__1, &d__2);
		_starpu_dscal_(n, &scl, &vr[i__ * vr_dim1 + 1], &c__1);
		_starpu_dscal_(n, &scl, &vr[(i__ + 1) * vr_dim1 + 1], &c__1);
		i__2 = *n;
		for (k = 1; k <= i__2; ++k) {
/* Computing 2nd power */
		    d__1 = vr[k + i__ * vr_dim1];
/* Computing 2nd power */
		    d__2 = vr[k + (i__ + 1) * vr_dim1];
		    work[k] = d__1 * d__1 + d__2 * d__2;
/* L30: */
		}
		k = _starpu_idamax_(n, &work[1], &c__1);
		_starpu_dlartg_(&vr[k + i__ * vr_dim1], &vr[k + (i__ + 1) * vr_dim1], 
			&cs, &sn, &r__);
		_starpu_drot_(n, &vr[i__ * vr_dim1 + 1], &c__1, &vr[(i__ + 1) * 
			vr_dim1 + 1], &c__1, &cs, &sn);
		vr[k + (i__ + 1) * vr_dim1] = 0.;
	    }
/* L40: */
	}
    }

/*     Undo scaling if necessary */

L50:
    if (scalea) {
	i__1 = *n - *info;
/* Computing MAX */
	i__3 = *n - *info;
	i__2 = max(i__3,1);
	_starpu_dlascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &wr[*info + 
		1], &i__2, &ierr);
	i__1 = *n - *info;
/* Computing MAX */
	i__3 = *n - *info;
	i__2 = max(i__3,1);
	_starpu_dlascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &wi[*info + 
		1], &i__2, &ierr);
	if (*info == 0) {
	    if ((wntsnv || wntsnb) && icond == 0) {
		_starpu_dlascl_("G", &c__0, &c__0, &cscale, &anrm, n, &c__1, &rcondv[
			1], n, &ierr);
	    }
	} else {
	    i__1 = *ilo - 1;
	    _starpu_dlascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &wr[1], 
		    n, &ierr);
	    i__1 = *ilo - 1;
	    _starpu_dlascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &wi[1], 
		    n, &ierr);
	}
    }

    work[1] = (doublereal) maxwrk;
    return 0;

/*     End of DGEEVX */

} /* _starpu_dgeevx_ */