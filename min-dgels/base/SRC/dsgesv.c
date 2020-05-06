/* dsgesv.f -- translated by f2c (version 20061008).
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

static doublereal c_b10 = -1.;
static doublereal c_b11 = 1.;
static integer c__1 = 1;

/* Subroutine */ int _starpu__starpu_dsgesv_(integer *n, integer *nrhs, doublereal *a, 
	integer *lda, integer *ipiv, doublereal *b, integer *ldb, doublereal *
	x, integer *ldx, doublereal *work, real *swork, integer *iter, 
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, work_dim1, work_offset, 
	    x_dim1, x_offset, i__1;
    doublereal d__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__;
    doublereal cte, eps, anrm;
    integer ptsa;
    doublereal rnrm, xnrm;
    integer ptsx;
    extern /* Subroutine */ int _starpu_dgemm_(char *, char *, integer *, integer *, 
	    integer *, doublereal *, doublereal *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, integer *);
    integer iiter;
    extern /* Subroutine */ int _starpu_daxpy_(integer *, doublereal *, doublereal *, 
	    integer *, doublereal *, integer *), _starpu_dlag2s_(integer *, integer *, 
	     doublereal *, integer *, real *, integer *, integer *), _starpu_slag2d_(
	    integer *, integer *, real *, integer *, doublereal *, integer *, 
	    integer *);
    extern doublereal _starpu_dlamch_(char *), _starpu_dlange_(char *, integer *, 
	    integer *, doublereal *, integer *, doublereal *);
    extern integer _starpu_idamax_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int _starpu_dlacpy_(char *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, integer *), 
	    _starpu_xerbla_(char *, integer *), _starpu_dgetrf_(integer *, integer *, 
	    doublereal *, integer *, integer *, integer *), _starpu_dgetrs_(char *, 
	    integer *, integer *, doublereal *, integer *, integer *, 
	    doublereal *, integer *, integer *), _starpu_sgetrf_(integer *, 
	    integer *, real *, integer *, integer *, integer *), _starpu_sgetrs_(char 
	    *, integer *, integer *, real *, integer *, integer *, real *, 
	    integer *, integer *);


/*  -- LAPACK PROTOTYPE driver routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     February 2007 */

/*     .. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DSGESV computes the solution to a real system of linear equations */
/*     A * X = B, */
/*  where A is an N-by-N matrix and X and B are N-by-NRHS matrices. */

/*  DSGESV first attempts to factorize the matrix in SINGLE PRECISION */
/*  and use this factorization within an iterative refinement procedure */
/*  to produce a solution with DOUBLE PRECISION normwise backward error */
/*  quality (see below). If the approach fails the method switches to a */
/*  DOUBLE PRECISION factorization and solve. */

/*  The iterative refinement is not going to be a winning strategy if */
/*  the ratio SINGLE PRECISION performance over DOUBLE PRECISION */
/*  performance is too small. A reasonable strategy should take the */
/*  number of right-hand sides and the size of the matrix into account. */
/*  This might be done with a call to ILAENV in the future. Up to now, we */
/*  always try iterative refinement. */

/*  The iterative refinement process is stopped if */
/*      ITER > ITERMAX */
/*  or for all the RHS we have: */
/*      RNRM < SQRT(N)*XNRM*ANRM*EPS*BWDMAX */
/*  where */
/*      o ITER is the number of the current iteration in the iterative */
/*        refinement process */
/*      o RNRM is the infinity-norm of the residual */
/*      o XNRM is the infinity-norm of the solution */
/*      o ANRM is the infinity-operator-norm of the matrix A */
/*      o EPS is the machine epsilon returned by DLAMCH('Epsilon') */
/*  The value ITERMAX and BWDMAX are fixed to 30 and 1.0D+00 */
/*  respectively. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The number of linear equations, i.e., the order of the */
/*          matrix A.  N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrix B.  NRHS >= 0. */

/*  A       (input or input/ouptut) DOUBLE PRECISION array, */
/*          dimension (LDA,N) */
/*          On entry, the N-by-N coefficient matrix A. */
/*          On exit, if iterative refinement has been successfully used */
/*          (INFO.EQ.0 and ITER.GE.0, see description below), then A is */
/*          unchanged, if double precision factorization has been used */
/*          (INFO.EQ.0 and ITER.LT.0, see description below), then the */
/*          array A contains the factors L and U from the factorization */
/*          A = P*L*U; the unit diagonal elements of L are not stored. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  IPIV    (output) INTEGER array, dimension (N) */
/*          The pivot indices that define the permutation matrix P; */
/*          row i of the matrix was interchanged with row IPIV(i). */
/*          Corresponds either to the single precision factorization */
/*          (if INFO.EQ.0 and ITER.GE.0) or the double precision */
/*          factorization (if INFO.EQ.0 and ITER.LT.0). */

/*  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS) */
/*          The N-by-NRHS right hand side matrix B. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  X       (output) DOUBLE PRECISION array, dimension (LDX,NRHS) */
/*          If INFO = 0, the N-by-NRHS solution matrix X. */

/*  LDX     (input) INTEGER */
/*          The leading dimension of the array X.  LDX >= max(1,N). */

/*  WORK    (workspace) DOUBLE PRECISION array, dimension (N*NRHS) */
/*          This array is used to hold the residual vectors. */

/*  SWORK   (workspace) REAL array, dimension (N*(N+NRHS)) */
/*          This array is used to use the single precision matrix and the */
/*          right-hand sides or solutions in single precision. */

/*  ITER    (output) INTEGER */
/*          < 0: iterative refinement has failed, double precision */
/*               factorization has been performed */
/*               -1 : the routine fell back to full precision for */
/*                    implementation- or machine-specific reasons */
/*               -2 : narrowing the precision induced an overflow, */
/*                    the routine fell back to full precision */
/*               -3 : failure of SGETRF */
/*               -31: stop the iterative refinement after the 30th */
/*                    iterations */
/*          > 0: iterative refinement has been sucessfully used. */
/*               Returns the number of iterations */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  if INFO = i, U(i,i) computed in DOUBLE PRECISION is */
/*                exactly zero.  The factorization has been completed, */
/*                but the factor U is exactly singular, so the solution */
/*                could not be computed. */

/*  ========= */

/*     .. Parameters .. */




/*     .. Local Scalars .. */

/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    work_dim1 = *n;
    work_offset = 1 + work_dim1;
    work -= work_offset;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    --swork;

    /* Function Body */
    *info = 0;
    *iter = 0;

/*     Test the input parameters. */

    if (*n < 0) {
	*info = -1;
    } else if (*nrhs < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    } else if (*ldb < max(1,*n)) {
	*info = -7;
    } else if (*ldx < max(1,*n)) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	_starpu_xerbla_("DSGESV", &i__1);
	return 0;
    }

/*     Quick return if (N.EQ.0). */

    if (*n == 0) {
	return 0;
    }

/*     Skip single precision iterative refinement if a priori slower */
/*     than double precision factorization. */

    if (FALSE_) {
	*iter = -1;
	goto L40;
    }

/*     Compute some constants. */

    anrm = _starpu_dlange_("I", n, n, &a[a_offset], lda, &work[work_offset]);
    eps = _starpu_dlamch_("Epsilon");
    cte = anrm * eps * sqrt((doublereal) (*n)) * 1.;

/*     Set the indices PTSA, PTSX for referencing SA and SX in SWORK. */

    ptsa = 1;
    ptsx = ptsa + *n * *n;

/*     Convert B from double precision to single precision and store the */
/*     result in SX. */

    _starpu_dlag2s_(n, nrhs, &b[b_offset], ldb, &swork[ptsx], n, info);

    if (*info != 0) {
	*iter = -2;
	goto L40;
    }

/*     Convert A from double precision to single precision and store the */
/*     result in SA. */

    _starpu_dlag2s_(n, n, &a[a_offset], lda, &swork[ptsa], n, info);

    if (*info != 0) {
	*iter = -2;
	goto L40;
    }

/*     Compute the LU factorization of SA. */

    _starpu_sgetrf_(n, n, &swork[ptsa], n, &ipiv[1], info);

    if (*info != 0) {
	*iter = -3;
	goto L40;
    }

/*     Solve the system SA*SX = SB. */

    _starpu_sgetrs_("No transpose", n, nrhs, &swork[ptsa], n, &ipiv[1], &swork[ptsx], 
	    n, info);

/*     Convert SX back to double precision */

    _starpu_slag2d_(n, nrhs, &swork[ptsx], n, &x[x_offset], ldx, info);

/*     Compute R = B - AX (R is WORK). */

    _starpu_dlacpy_("All", n, nrhs, &b[b_offset], ldb, &work[work_offset], n);

    _starpu_dgemm_("No Transpose", "No Transpose", n, nrhs, n, &c_b10, &a[a_offset], 
	    lda, &x[x_offset], ldx, &c_b11, &work[work_offset], n);

/*     Check whether the NRHS normwise backward errors satisfy the */
/*     stopping criterion. If yes, set ITER=0 and return. */

    i__1 = *nrhs;
    for (i__ = 1; i__ <= i__1; ++i__) {
	xnrm = (d__1 = x[_starpu_idamax_(n, &x[i__ * x_dim1 + 1], &c__1) + i__ * 
		x_dim1], abs(d__1));
	rnrm = (d__1 = work[_starpu_idamax_(n, &work[i__ * work_dim1 + 1], &c__1) + 
		i__ * work_dim1], abs(d__1));
	if (rnrm > xnrm * cte) {
	    goto L10;
	}
    }

/*     If we are here, the NRHS normwise backward errors satisfy the */
/*     stopping criterion. We are good to exit. */

    *iter = 0;
    return 0;

L10:

    for (iiter = 1; iiter <= 30; ++iiter) {

/*        Convert R (in WORK) from double precision to single precision */
/*        and store the result in SX. */

	_starpu_dlag2s_(n, nrhs, &work[work_offset], n, &swork[ptsx], n, info);

	if (*info != 0) {
	    *iter = -2;
	    goto L40;
	}

/*        Solve the system SA*SX = SR. */

	_starpu_sgetrs_("No transpose", n, nrhs, &swork[ptsa], n, &ipiv[1], &swork[
		ptsx], n, info);

/*        Convert SX back to double precision and update the current */
/*        iterate. */

	_starpu_slag2d_(n, nrhs, &swork[ptsx], n, &work[work_offset], n, info);

	i__1 = *nrhs;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    _starpu_daxpy_(n, &c_b11, &work[i__ * work_dim1 + 1], &c__1, &x[i__ * 
		    x_dim1 + 1], &c__1);
	}

/*        Compute R = B - AX (R is WORK). */

	_starpu_dlacpy_("All", n, nrhs, &b[b_offset], ldb, &work[work_offset], n);

	_starpu_dgemm_("No Transpose", "No Transpose", n, nrhs, n, &c_b10, &a[
		a_offset], lda, &x[x_offset], ldx, &c_b11, &work[work_offset], 
		 n);

/*        Check whether the NRHS normwise backward errors satisfy the */
/*        stopping criterion. If yes, set ITER=IITER>0 and return. */

	i__1 = *nrhs;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    xnrm = (d__1 = x[_starpu_idamax_(n, &x[i__ * x_dim1 + 1], &c__1) + i__ * 
		    x_dim1], abs(d__1));
	    rnrm = (d__1 = work[_starpu_idamax_(n, &work[i__ * work_dim1 + 1], &c__1) 
		    + i__ * work_dim1], abs(d__1));
	    if (rnrm > xnrm * cte) {
		goto L20;
	    }
	}

/*        If we are here, the NRHS normwise backward errors satisfy the */
/*        stopping criterion, we are good to exit. */

	*iter = iiter;

	return 0;

L20:

/* L30: */
	;
    }

/*     If we are at this place of the code, this is because we have */
/*     performed ITER=ITERMAX iterations and never satisified the */
/*     stopping criterion, set up the ITER flag accordingly and follow up */
/*     on double precision routine. */

    *iter = -31;

L40:

/*     Single-precision iterative refinement failed to converge to a */
/*     satisfactory solution, so we resort to double precision. */

    _starpu_dgetrf_(n, n, &a[a_offset], lda, &ipiv[1], info);

    if (*info != 0) {
	return 0;
    }

    _starpu_dlacpy_("All", n, nrhs, &b[b_offset], ldb, &x[x_offset], ldx);
    _starpu_dgetrs_("No transpose", n, nrhs, &a[a_offset], lda, &ipiv[1], &x[x_offset]
, ldx, info);

    return 0;

/*     End of DSGESV. */

} /* _starpu__starpu_dsgesv_ */