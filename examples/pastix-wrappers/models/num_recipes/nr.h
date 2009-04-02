/* CAUTION: This is the ANSI C (only) version of the Numerical Recipes
   utility file nr.h.  Do not confuse this file with the same-named
   file nr.h that is supplied in the 'misc' subdirectory.
   *That* file is the one from the book, and contains both ANSI and
   traditional K&R versions, along with #ifdef macros to select the
   correct version.  *This* file contains only ANSI C.               */

#ifndef _NR_H_
#define _NR_H_

#define REAL double


#ifndef _FCOMPLEX_DECLARE_T_
typedef struct FCOMPLEX {REAL r,i;} fcomplex;
#define _FCOMPLEX_DECLARE_T_
#endif /* _FCOMPLEX_DECLARE_T_ */

#ifndef _ARITHCODE_DECLARE_T_
typedef struct {
	unsigned long *ilob,*iupb,*ncumfq,jdif,nc,minint,nch,ncum,nrad;
} arithcode;
#define _ARITHCODE_DECLARE_T_
#endif /* _ARITHCODE_DECLARE_T_ */

#ifndef _HUFFCODE_DECLARE_T_
typedef struct {
	unsigned long *icod,*ncod,*left,*right,nch,nodemax;
} huffcode;
#define _HUFFCODE_DECLARE_T_
#endif /* _HUFFCODE_DECLARE_T_ */

#include <stdio.h>

void addint(double **uf, double **uc, double **res, int nf);
void airy(REAL x, REAL *ai, REAL *bi, REAL *aip, REAL *bip);
void amebsa(REAL **p, REAL y[], int ndim, REAL pb[],	REAL *yb,
	REAL ftol, REAL (*funk)(REAL []), int *iter, REAL temptr);
void amoeba(REAL **p, REAL y[], int ndim, REAL ftol,
	REAL (*funk)(REAL []), int *iter);
REAL amotry(REAL **p, REAL y[], REAL psum[], int ndim,
	REAL (*funk)(REAL []), int ihi, REAL fac);
REAL amotsa(REAL **p, REAL y[], REAL psum[], int ndim, REAL pb[],
	REAL *yb, REAL (*funk)(REAL []), int ihi, REAL *yhi, REAL fac);
void anneal(REAL x[], REAL y[], int iorder[], int ncity);
double anorm2(double **a, int n);
void arcmak(unsigned long nfreq[], unsigned long nchh, unsigned long nradd,
	arithcode *acode);
void arcode(unsigned long *ich, unsigned char **codep, unsigned long *lcode,
	unsigned long *lcd, int isign, arithcode *acode);
void arcsum(unsigned long iin[], unsigned long iout[], unsigned long ja,
	int nwk, unsigned long nrad, unsigned long nc);
void asolve(unsigned long n, double b[], double x[], int itrnsp);
void atimes(unsigned long n, double x[], double r[], int itrnsp);
void avevar(REAL data[], unsigned long n, REAL *ave, REAL *var);
void balanc(REAL **a, int n);
void banbks(REAL **a, unsigned long n, int m1, int m2, REAL **al,
	unsigned long indx[], REAL b[]);
void bandec(REAL **a, unsigned long n, int m1, int m2, REAL **al,
	unsigned long indx[], REAL *d);
void banmul(REAL **a, unsigned long n, int m1, int m2, REAL x[], REAL b[]);
void bcucof(REAL y[], REAL y1[], REAL y2[], REAL y12[], REAL d1,
	REAL d2, REAL **c);
void bcuint(REAL y[], REAL y1[], REAL y2[], REAL y12[],
	REAL x1l, REAL x1u, REAL x2l, REAL x2u, REAL x1,
	REAL x2, REAL *ansy, REAL *ansy1, REAL *ansy2);
void beschb(double x, double *gam1, double *gam2, double *gampl,
	double *gammi);
REAL bessi(int n, REAL x);
REAL bessi0(REAL x);
REAL bessi1(REAL x);
void bessik(REAL x, REAL xnu, REAL *ri, REAL *rk, REAL *rip,
	REAL *rkp);
REAL bessj(int n, REAL x);
REAL bessj0(REAL x);
REAL bessj1(REAL x);
void bessjy(REAL x, REAL xnu, REAL *rj, REAL *ry, REAL *rjp,
	REAL *ryp);
REAL bessk(int n, REAL x);
REAL bessk0(REAL x);
REAL bessk1(REAL x);
REAL bessy(int n, REAL x);
REAL bessy0(REAL x);
REAL bessy1(REAL x);
REAL beta(REAL z, REAL w);
REAL betacf(REAL a, REAL b, REAL x);
REAL betai(REAL a, REAL b, REAL x);
REAL bico(int n, int k);
void bksub(int ne, int nb, int jf, int k1, int k2, REAL ***c);
REAL bnldev(REAL pp, int n, long *idum);
REAL brent(REAL ax, REAL bx, REAL cx,
	REAL (*f)(REAL), REAL tol, REAL *xmin);
void broydn(REAL x[], int n, int *check,
	void (*vecfunc)(int, REAL [], REAL []));
void bsstep(REAL y[], REAL dydx[], int nv, REAL *xx, REAL htry,
	REAL eps, REAL yscal[], REAL *hdid, REAL *hnext,
	void (*derivs)(REAL, REAL [], REAL []));
void caldat(long julian, int *mm, int *id, int *iyyy);
void chder(REAL a, REAL b, REAL c[], REAL cder[], int n);
REAL chebev(REAL a, REAL b, REAL c[], int m, REAL x);
void chebft(REAL a, REAL b, REAL c[], int n, REAL (*func)(REAL));
void chebpc(REAL c[], REAL d[], int n);
void chint(REAL a, REAL b, REAL c[], REAL cint[], int n);
REAL chixy(REAL bang);
void choldc(REAL **a, int n, REAL p[]);
void cholsl(REAL **a, int n, REAL p[], REAL b[], REAL x[]);
void chsone(REAL bins[], REAL ebins[], int nbins, int knstrn,
	REAL *df, REAL *chsq, REAL *prob);
void chstwo(REAL bins1[], REAL bins2[], int nbins, int knstrn,
	REAL *df, REAL *chsq, REAL *prob);
void cisi(REAL x, REAL *ci, REAL *si);
void cntab1(int **nn, int ni, int nj, REAL *chisq,
	REAL *df, REAL *prob, REAL *cramrv, REAL *ccc);
void cntab2(int **nn, int ni, int nj, REAL *h, REAL *hx, REAL *hy,
	REAL *hygx, REAL *hxgy, REAL *uygx, REAL *uxgy, REAL *uxy);
void convlv(REAL data[], unsigned long n, REAL respns[], unsigned long m,
	int isign, REAL ans[]);
void copy(double **aout, double **ain, int n);
void correl(REAL data1[], REAL data2[], unsigned long n, REAL ans[]);
void cosft(REAL y[], int n, int isign);
void cosft1(REAL y[], int n);
void cosft2(REAL y[], int n, int isign);
void covsrt(REAL **covar, int ma, int ia[], int mfit);
void crank(unsigned long n, REAL w[], REAL *s);
void cyclic(REAL a[], REAL b[], REAL c[], REAL alpha, REAL beta,
	REAL r[], REAL x[], unsigned long n);
void daub4(REAL a[], unsigned long n, int isign);
REAL dawson(REAL x);
REAL dbrent(REAL ax, REAL bx, REAL cx,
	REAL (*f)(REAL), REAL (*df)(REAL), REAL tol, REAL *xmin);
void ddpoly(REAL c[], int nc, REAL x, REAL pd[], int nd);
int decchk(char string[], int n, char *ch);
void derivs(REAL x, REAL y[], REAL dydx[]);
REAL df1dim(REAL x);
void dfour1(double data[], unsigned long nn, int isign);
void dfpmin(REAL p[], int n, REAL gtol, int *iter, REAL *fret,
	REAL (*func)(REAL []), void (*dfunc)(REAL [], REAL []));
REAL dfridr(REAL (*func)(REAL), REAL x, REAL h, REAL *err);
void dftcor(REAL w, REAL delta, REAL a, REAL b, REAL endpts[],
	REAL *corre, REAL *corim, REAL *corfac);
void dftint(REAL (*func)(REAL), REAL a, REAL b, REAL w,
	REAL *cosint, REAL *sinint);
void difeq(int k, int k1, int k2, int jsf, int is1, int isf,
	int indexv[], int ne, REAL **s, REAL **y);
void dlinmin(REAL p[], REAL xi[], int n, REAL *fret,
	REAL (*func)(REAL []), void (*dfunc)(REAL [], REAL[]));
double dpythag(double a, double b);
void drealft(double data[], unsigned long n, int isign);
void dsprsax(double sa[], unsigned long ija[], double x[], double b[],
	unsigned long n);
void dsprstx(double sa[], unsigned long ija[], double x[], double b[],
	unsigned long n);
void dsvbksb(double **u, double w[], double **v, int m, int n, double b[],
	double x[]);
void dsvdcmp(double **a, int m, int n, double w[], double **v);
void eclass(int nf[], int n, int lista[], int listb[], int m);
void eclazz(int nf[], int n, int (*equiv)(int, int));
REAL ei(REAL x);
void eigsrt(REAL d[], REAL **v, int n);
REAL elle(REAL phi, REAL ak);
REAL ellf(REAL phi, REAL ak);
REAL ellpi(REAL phi, REAL en, REAL ak);
void elmhes(REAL **a, int n);
REAL erfcc(REAL x);
//REAL erff(REAL x);
REAL erffc(REAL x);
void eulsum(REAL *sum, REAL term, int jterm, REAL wksp[]);
REAL evlmem(REAL fdt, REAL d[], int m, REAL xms);
REAL expdev(long *idum);
REAL expint(int n, REAL x);
REAL f1(REAL x);
REAL f1dim(REAL x);
REAL f2(REAL y);
REAL f3(REAL z);
REAL factln(int n);
REAL factrl(int n);
void fasper(REAL x[], REAL y[], unsigned long n, REAL ofac, REAL hifac,
	REAL wk1[], REAL wk2[], unsigned long nwk, unsigned long *nout,
	unsigned long *jmax, REAL *prob);
void fdjac(int n, REAL x[], REAL fvec[], REAL **df,
	void (*vecfunc)(int, REAL [], REAL []));
void fgauss(REAL x, REAL a[], REAL *y, REAL dyda[], int na);
void fill0(double **u, int n);
void fit(REAL x[], REAL y[], int ndata, REAL sig[], int mwt,
	REAL *a, REAL *b, REAL *siga, REAL *sigb, REAL *chi2, REAL *q);
void fitexy(REAL x[], REAL y[], int ndat, REAL sigx[], REAL sigy[],
	REAL *a, REAL *b, REAL *siga, REAL *sigb, REAL *chi2, REAL *q);
void fixrts(REAL d[], int m);
void fleg(REAL x, REAL pl[], int nl);
void flmoon(int n, int nph, long *jd, REAL *frac);
REAL fmin(REAL x[]);
void four1(REAL data[], unsigned long nn, int isign);
void fourew(FILE *file[5], int *na, int *nb, int *nc, int *nd);
void fourfs(FILE *file[5], unsigned long nn[], int ndim, int isign);
void fourn(REAL data[], unsigned long nn[], int ndim, int isign);
void fpoly(REAL x, REAL p[], int np);
void fred2(int n, REAL a, REAL b, REAL t[], REAL f[], REAL w[],
	REAL (*g)(REAL), REAL (*ak)(REAL, REAL));
REAL fredin(REAL x, int n, REAL a, REAL b, REAL t[], REAL f[], REAL w[],
	REAL (*g)(REAL), REAL (*ak)(REAL, REAL));
void frenel(REAL x, REAL *s, REAL *c);
void frprmn(REAL p[], int n, REAL ftol, int *iter, REAL *fret,
	REAL (*func)(REAL []), void (*dfunc)(REAL [], REAL []));
void ftest(REAL data1[], unsigned long n1, REAL data2[], unsigned long n2,
	REAL *f, REAL *prob);
REAL gamdev(int ia, long *idum);
REAL gammln(REAL xx);
REAL gammp(REAL a, REAL x);
REAL gammq(REAL a, REAL x);
REAL gasdev(long *idum);
void gaucof(int n, REAL a[], REAL b[], REAL amu0, REAL x[], REAL w[]);
void gauher(REAL x[], REAL w[], int n);
void gaujac(REAL x[], REAL w[], int n, REAL alf, REAL bet);
void gaulag(REAL x[], REAL w[], int n, REAL alf);
void gauleg(REAL x1, REAL x2, REAL x[], REAL w[], int n);
void gaussj(REAL **a, int n, REAL **b, int m);
void gcf(REAL *gammcf, REAL a, REAL x, REAL *gln);
REAL golden(REAL ax, REAL bx, REAL cx, REAL (*f)(REAL), REAL tol,
	REAL *xmin);
void gser(REAL *gamser, REAL a, REAL x, REAL *gln);
void hpsel(unsigned long m, unsigned long n, REAL arr[], REAL heap[]);
void hpsort(unsigned long n, REAL ra[]);
void hqr(REAL **a, int n, REAL wr[], REAL wi[]);
void hufapp(unsigned long index[], unsigned long nprob[], unsigned long n,
	unsigned long i);
void hufdec(unsigned long *ich, unsigned char *code, unsigned long lcode,
	unsigned long *nb, huffcode *hcode);
void hufenc(unsigned long ich, unsigned char **codep, unsigned long *lcode,
	unsigned long *nb, huffcode *hcode);
void hufmak(unsigned long nfreq[], unsigned long nchin, unsigned long *ilong,
	unsigned long *nlong, huffcode *hcode);
void hunt(REAL xx[], unsigned long n, REAL x, unsigned long *jlo);
void hypdrv(REAL s, REAL yy[], REAL dyyds[]);
fcomplex hypgeo(fcomplex a, fcomplex b, fcomplex c, fcomplex z);
void hypser(fcomplex a, fcomplex b, fcomplex c, fcomplex z,
	fcomplex *series, fcomplex *deriv);
unsigned short icrc(unsigned short crc, unsigned char *bufptr,
	unsigned long len, short jinit, int jrev);
unsigned short icrc1(unsigned short crc, unsigned char onech);
unsigned long igray(unsigned long n, int is);
void iindexx(unsigned long n, long arr[], unsigned long indx[]);
void indexx(unsigned long n, REAL arr[], unsigned long indx[]);
void interp(double **uf, double **uc, int nf);
int irbit1(unsigned long *iseed);
int irbit2(unsigned long *iseed);
void jacobi(REAL **a, int n, REAL d[], REAL **v, int *nrot);
void jacobn(REAL x, REAL y[], REAL dfdx[], REAL **dfdy, int n);
long julday(int mm, int id, int iyyy);
void kendl1(REAL data1[], REAL data2[], unsigned long n, REAL *tau, REAL *z,
	REAL *prob);
void kendl2(REAL **tab, int i, int j, REAL *tau, REAL *z, REAL *prob);
void kermom(double w[], double y, int m);
void ks2d1s(REAL x1[], REAL y1[], unsigned long n1,
	void (*quadvl)(REAL, REAL, REAL *, REAL *, REAL *, REAL *),
	REAL *d1, REAL *prob);
void ks2d2s(REAL x1[], REAL y1[], unsigned long n1, REAL x2[], REAL y2[],
	unsigned long n2, REAL *d, REAL *prob);
void ksone(REAL data[], unsigned long n, REAL (*func)(REAL), REAL *d,
	REAL *prob);
void kstwo(REAL data1[], unsigned long n1, REAL data2[], unsigned long n2,
	REAL *d, REAL *prob);
void laguer(fcomplex a[], int m, fcomplex *x, int *its);
void lfit(REAL x[], REAL y[], REAL sig[], int ndat, REAL a[], int ia[],
	int ma, REAL **covar, REAL *chisq, void (*funcs)(REAL, REAL [], int));
void linbcg(unsigned long n, double b[], double x[], int itol, double tol,
	 int itmax, int *iter, double *err);
void linmin(REAL p[], REAL xi[], int n, REAL *fret,
	REAL (*func)(REAL []));
void lnsrch(int n, REAL xold[], REAL fold, REAL g[], REAL p[], REAL x[],
	 REAL *f, REAL stpmax, int *check, REAL (*func)(REAL []));
void load(REAL x1, REAL v[], REAL y[]);
void load1(REAL x1, REAL v1[], REAL y[]);
void load2(REAL x2, REAL v2[], REAL y[]);
void locate(REAL xx[], unsigned long n, REAL x, unsigned long *j);
void lop(double **out, double **u, int n);
void lubksb(REAL **a, int n, int *indx, REAL b[]);
void ludcmp(REAL **a, int n, int *indx, REAL *d);
void machar(int *ibeta, int *it, int *irnd, int *ngrd,
	int *machep, int *negep, int *iexp, int *minexp, int *maxexp,
	REAL *eps, REAL *epsneg, REAL *xmin, REAL *xmax);
void matadd(double **a, double **b, double **c, int n);
void matsub(double **a, double **b, double **c, int n);
void medfit(REAL x[], REAL y[], int ndata, REAL *a, REAL *b, REAL *abdev);
void memcof(REAL data[], int n, int m, REAL *xms, REAL d[]);
int metrop(REAL de, REAL t);
void mgfas(double **u, int n, int maxcyc);
void mglin(double **u, int n, int ncycle);
REAL midexp(REAL (*funk)(REAL), REAL aa, REAL bb, int n);
REAL midinf(REAL (*funk)(REAL), REAL aa, REAL bb, int n);
REAL midpnt(REAL (*func)(REAL), REAL a, REAL b, int n);
REAL midsql(REAL (*funk)(REAL), REAL aa, REAL bb, int n);
REAL midsqu(REAL (*funk)(REAL), REAL aa, REAL bb, int n);
void miser(REAL (*func)(REAL []), REAL regn[], int ndim, unsigned long npts,
	REAL dith, REAL *ave, REAL *var);
void mmid(REAL y[], REAL dydx[], int nvar, REAL xs, REAL htot,
	int nstep, REAL yout[], void (*derivs)(REAL, REAL[], REAL[]));
void mnbrak(REAL *ax, REAL *bx, REAL *cx, REAL *fa, REAL *fb,
	REAL *fc, REAL (*func)(REAL));
void mnewt(int ntrial, REAL x[], int n, REAL tolx, REAL tolf);
void moment(REAL data[], int n, REAL *ave, REAL *adev, REAL *sdev,
	REAL *var, REAL *skew, REAL *curt);
void mp2dfr(unsigned char a[], unsigned char s[], int n, int *m);
void mpadd(unsigned char w[], unsigned char u[], unsigned char v[], int n);
void mpdiv(unsigned char q[], unsigned char r[], unsigned char u[],
	unsigned char v[], int n, int m);
void mpinv(unsigned char u[], unsigned char v[], int n, int m);
void mplsh(unsigned char u[], int n);
void mpmov(unsigned char u[], unsigned char v[], int n);
void mpmul(unsigned char w[], unsigned char u[], unsigned char v[], int n,
	int m);
void mpneg(unsigned char u[], int n);
void mppi(int n);
void mprove(REAL **a, REAL **alud, int n, int indx[], REAL b[],
	REAL x[]);
void mpsad(unsigned char w[], unsigned char u[], int n, int iv);
void mpsdv(unsigned char w[], unsigned char u[], int n, int iv, int *ir);
void mpsmu(unsigned char w[], unsigned char u[], int n, int iv);
void mpsqrt(unsigned char w[], unsigned char u[], unsigned char v[], int n,
	int m);
void mpsub(int *is, unsigned char w[], unsigned char u[], unsigned char v[],
	int n);
void mrqcof(REAL x[], REAL y[], REAL sig[], int ndata, REAL a[],
	int ia[], int ma, REAL **alpha, REAL beta[], REAL *chisq,
	void (*funcs)(REAL, REAL [], REAL *, REAL [], int));
void mrqmin(REAL x[], REAL y[], REAL sig[], int ndata, REAL a[],
	int ia[], int ma, REAL **covar, REAL **alpha, REAL *chisq,
	void (*funcs)(REAL, REAL [], REAL *, REAL [], int), REAL *alamda);
void newt(REAL x[], int n, int *check,
	void (*vecfunc)(int, REAL [], REAL []));
void odeint(REAL ystart[], int nvar, REAL x1, REAL x2,
	REAL eps, REAL h1, REAL hmin, int *nok, int *nbad,
	void (*derivs)(REAL, REAL [], REAL []),
	void (*rkqs)(REAL [], REAL [], int, REAL *, REAL, REAL,
	REAL [], REAL *, REAL *, void (*)(REAL, REAL [], REAL [])));
void orthog(int n, REAL anu[], REAL alpha[], REAL beta[], REAL a[],
	REAL b[]);
void pade(double cof[], int n, REAL *resid);
void pccheb(REAL d[], REAL c[], int n);
void pcshft(REAL a, REAL b, REAL d[], int n);
void pearsn(REAL x[], REAL y[], unsigned long n, REAL *r, REAL *prob,
	REAL *z);
void period(REAL x[], REAL y[], int n, REAL ofac, REAL hifac,
	REAL px[], REAL py[], int np, int *nout, int *jmax, REAL *prob);
void piksr2(int n, REAL arr[], REAL brr[]);
void piksrt(int n, REAL arr[]);
void pinvs(int ie1, int ie2, int je1, int jsf, int jc1, int k,
	REAL ***c, REAL **s);
REAL plgndr(int l, int m, REAL x);
REAL poidev(REAL xm, long *idum);
void polcoe(REAL x[], REAL y[], int n, REAL cof[]);
void polcof(REAL xa[], REAL ya[], int n, REAL cof[]);
void poldiv(REAL u[], int n, REAL v[], int nv, REAL q[], REAL r[]);
void polin2(REAL x1a[], REAL x2a[], REAL **ya, int m, int n,
	REAL x1, REAL x2, REAL *y, REAL *dy);
void polint(REAL xa[], REAL ya[], int n, REAL x, REAL *y, REAL *dy);
void powell(REAL p[], REAL **xi, int n, REAL ftol, int *iter, REAL *fret,
	REAL (*func)(REAL []));
void predic(REAL data[], int ndata, REAL d[], int m, REAL future[], int nfut);
REAL probks(REAL alam);
void psdes(unsigned long *lword, unsigned long *irword);
void pwt(REAL a[], unsigned long n, int isign);
void pwtset(int n);
REAL pythag(REAL a, REAL b);
void pzextr(int iest, REAL xest, REAL yest[], REAL yz[], REAL dy[],
	int nv);
REAL qgaus(REAL (*func)(REAL), REAL a, REAL b);
void qrdcmp(REAL **a, int n, REAL *c, REAL *d, int *sing);
REAL qromb(REAL (*func)(REAL), REAL a, REAL b);
REAL qromo(REAL (*func)(REAL), REAL a, REAL b,
	REAL (*choose)(REAL (*)(REAL), REAL, REAL, int));
void qroot(REAL p[], int n, REAL *b, REAL *c, REAL eps);
void qrsolv(REAL **a, int n, REAL c[], REAL d[], REAL b[]);
void qrupdt(REAL **r, REAL **qt, int n, REAL u[], REAL v[]);
REAL qsimp(REAL (*func)(REAL), REAL a, REAL b);
REAL qtrap(REAL (*func)(REAL), REAL a, REAL b);
REAL quad3d(REAL (*func)(REAL, REAL, REAL), REAL x1, REAL x2);
void quadct(REAL x, REAL y, REAL xx[], REAL yy[], unsigned long nn,
	REAL *fa, REAL *fb, REAL *fc, REAL *fd);
void quadmx(REAL **a, int n);
void quadvl(REAL x, REAL y, REAL *fa, REAL *fb, REAL *fc, REAL *fd);
REAL ran0(long *idum);
REAL ran1(long *idum);
REAL ran2(long *idum);
REAL ran3(long *idum);
REAL ran4(long *idum);
void rank(unsigned long n, unsigned long indx[], unsigned long irank[]);
void ranpt(REAL pt[], REAL regn[], int n);
void ratint(REAL xa[], REAL ya[], int n, REAL x, REAL *y, REAL *dy);
void ratlsq(double (*fn)(double), double a, double b, int mm, int kk,
	double cof[], double *dev);
double ratval(double x, double cof[], int mm, int kk);
REAL rc(REAL x, REAL y);
REAL rd(REAL x, REAL y, REAL z);
void realft(REAL data[], unsigned long n, int isign);
void rebin(REAL rc, int nd, REAL r[], REAL xin[], REAL xi[]);
void red(int iz1, int iz2, int jz1, int jz2, int jm1, int jm2, int jmf,
	int ic1, int jc1, int jcf, int kc, REAL ***c, REAL **s);
void relax(double **u, double **rhs, int n);
void relax2(double **u, double **rhs, int n);
void resid(double **res, double **u, double **rhs, int n);
REAL revcst(REAL x[], REAL y[], int iorder[], int ncity, int n[]);
void reverse(int iorder[], int ncity, int n[]);
REAL rf(REAL x, REAL y, REAL z);
REAL rj(REAL x, REAL y, REAL z, REAL p);
void rk4(REAL y[], REAL dydx[], int n, REAL x, REAL h, REAL yout[],
	void (*derivs)(REAL, REAL [], REAL []));
void rkck(REAL y[], REAL dydx[], int n, REAL x, REAL h,
	REAL yout[], REAL yerr[], void (*derivs)(REAL, REAL [], REAL []));
void rkdumb(REAL vstart[], int nvar, REAL x1, REAL x2, int nstep,
	void (*derivs)(REAL, REAL [], REAL []));
void rkqs(REAL y[], REAL dydx[], int n, REAL *x,
	REAL htry, REAL eps, REAL yscal[], REAL *hdid, REAL *hnext,
	void (*derivs)(REAL, REAL [], REAL []));
void rlft3(REAL ***data, REAL **speq, unsigned long nn1,
	unsigned long nn2, unsigned long nn3, int isign);
REAL rofunc(REAL b);
void rotate(REAL **r, REAL **qt, int n, int i, REAL a, REAL b);
void rsolv(REAL **a, int n, REAL d[], REAL b[]);
void rstrct(double **uc, double **uf, int nc);
REAL rtbis(REAL (*func)(REAL), REAL x1, REAL x2, REAL xacc);
REAL rtflsp(REAL (*func)(REAL), REAL x1, REAL x2, REAL xacc);
REAL rtnewt(void (*funcd)(REAL, REAL *, REAL *), REAL x1, REAL x2,
	REAL xacc);
REAL rtsafe(void (*funcd)(REAL, REAL *, REAL *), REAL x1, REAL x2,
	REAL xacc);
REAL rtsec(REAL (*func)(REAL), REAL x1, REAL x2, REAL xacc);
void rzextr(int iest, REAL xest, REAL yest[], REAL yz[], REAL dy[], int nv);
void savgol(REAL c[], int np, int nl, int nr, int ld, int m);
void score(REAL xf, REAL y[], REAL f[]);
void scrsho(REAL (*fx)(REAL));
REAL select_(unsigned long k, unsigned long n, REAL arr[]);
REAL selip(unsigned long k, unsigned long n, REAL arr[]);
void shell(unsigned long n, REAL a[]);
void shoot(int n, REAL v[], REAL f[]);
void shootf(int n, REAL v[], REAL f[]);
void simp1(REAL **a, int mm, int ll[], int nll, int iabf, int *kp,
	REAL *bmax);
void simp2(REAL **a, int n, int l2[], int nl2, int *ip, int kp, REAL *q1);
void simp3(REAL **a, int i1, int k1, int ip, int kp);
void simplx(REAL **a, int m, int n, int m1, int m2, int m3, int *icase,
	int izrov[], int iposv[]);
void simpr(REAL y[], REAL dydx[], REAL dfdx[], REAL **dfdy,
	int n, REAL xs, REAL htot, int nstep, REAL yout[],
	void (*derivs)(REAL, REAL [], REAL []));
void sinft(REAL y[], int n);
void slvsm2(double **u, double **rhs);
void slvsml(double **u, double **rhs);
void sncndn(REAL uu, REAL emmc, REAL *sn, REAL *cn, REAL *dn);
double snrm(unsigned long n, double sx[], int itol);
void sobseq(int *n, REAL x[]);
void solvde(int itmax, REAL conv, REAL slowc, REAL scalv[],
	int indexv[], int ne, int nb, int m, REAL **y, REAL ***c, REAL **s);
void sor(double **a, double **b, double **c, double **d, double **e,
	double **f, double **u, int jmax, double rjac);
void sort(unsigned long n, REAL arr[]);
void sort2(unsigned long n, REAL arr[], REAL brr[]);
void sort3(unsigned long n, REAL ra[], REAL rb[], REAL rc[]);
void spctrm(FILE *fp, REAL p[], int m, int k, int ovrlap);
void spear(REAL data1[], REAL data2[], unsigned long n, REAL *d, REAL *zd,
	REAL *probd, REAL *rs, REAL *probrs);
void sphbes(int n, REAL x, REAL *sj, REAL *sy, REAL *sjp, REAL *syp);
void splie2(REAL x1a[], REAL x2a[], REAL **ya, int m, int n, REAL **y2a);
void splin2(REAL x1a[], REAL x2a[], REAL **ya, REAL **y2a, int m, int n,
	REAL x1, REAL x2, REAL *y);
void spline(REAL x[], REAL y[], int n, REAL yp1, REAL ypn, REAL y2[]);
void splint(REAL xa[], REAL ya[], REAL y2a[], int n, REAL x, REAL *y);
void spread(REAL y, REAL yy[], unsigned long n, REAL x, int m);
void sprsax(REAL sa[], unsigned long ija[], REAL x[], REAL b[],
	unsigned long n);
void sprsin(REAL **a, int n, REAL thresh, unsigned long nmax, REAL sa[],
	unsigned long ija[]);
void sprspm(REAL sa[], unsigned long ija[], REAL sb[], unsigned long ijb[],
	REAL sc[], unsigned long ijc[]);
void sprstm(REAL sa[], unsigned long ija[], REAL sb[], unsigned long ijb[],
	REAL thresh, unsigned long nmax, REAL sc[], unsigned long ijc[]);
void sprstp(REAL sa[], unsigned long ija[], REAL sb[], unsigned long ijb[]);
void sprstx(REAL sa[], unsigned long ija[], REAL x[], REAL b[],
	unsigned long n);
void stifbs(REAL y[], REAL dydx[], int nv, REAL *xx,
	REAL htry, REAL eps, REAL yscal[], REAL *hdid, REAL *hnext,
	void (*derivs)(REAL, REAL [], REAL []));
void stiff(REAL y[], REAL dydx[], int n, REAL *x,
	REAL htry, REAL eps, REAL yscal[], REAL *hdid, REAL *hnext,
	void (*derivs)(REAL, REAL [], REAL []));
void stoerm(REAL y[], REAL d2y[], int nv, REAL xs,
	REAL htot, int nstep, REAL yout[],
	void (*derivs)(REAL, REAL [], REAL []));
void svbksb(REAL **u, REAL w[], REAL **v, int m, int n, REAL b[],
	REAL x[]);
void svdcmp(REAL **a, int m, int n, REAL w[], REAL **v);
void svdfit(REAL x[], REAL y[], REAL sig[], int ndata, REAL a[],
	int ma, REAL **u, REAL **v, REAL w[], REAL *chisq,
	void (*funcs)(REAL, REAL [], int));
void svdvar(REAL **v, int ma, REAL w[], REAL **cvm);
void toeplz(REAL r[], REAL x[], REAL y[], int n);
void tptest(REAL data1[], REAL data2[], unsigned long n, REAL *t, REAL *prob);
void tqli(REAL d[], REAL e[], int n, REAL **z);
REAL trapzd(REAL (*func)(REAL), REAL a, REAL b, int n);
void tred2(REAL **a, int n, REAL d[], REAL e[]);
void tridag(REAL a[], REAL b[], REAL c[], REAL r[], REAL u[],
	unsigned long n);
REAL trncst(REAL x[], REAL y[], int iorder[], int ncity, int n[]);
void trnspt(int iorder[], int ncity, int n[]);
void ttest(REAL data1[], unsigned long n1, REAL data2[], unsigned long n2,
	REAL *t, REAL *prob);
void tutest(REAL data1[], unsigned long n1, REAL data2[], unsigned long n2,
	REAL *t, REAL *prob);
void twofft(REAL data1[], REAL data2[], REAL fft1[], REAL fft2[],
	unsigned long n);
void vander(double x[], double w[], double q[], int n);
void vegas(REAL regn[], int ndim, REAL (*fxn)(REAL [], REAL), int init,
	unsigned long ncall, int itmx, int nprn, REAL *tgral, REAL *sd,
	REAL *chi2a);
void voltra(int n, int m, REAL t0, REAL h, REAL *t, REAL **f,
	REAL (*g)(int, REAL), REAL (*ak)(int, int, REAL, REAL));
void wt1(REAL a[], unsigned long n, int isign,
	void (*wtstep)(REAL [], unsigned long, int));
void wtn(REAL a[], unsigned long nn[], int ndim, int isign,
	void (*wtstep)(REAL [], unsigned long, int));
void wwghts(REAL wghts[], int n, REAL h,
	void (*kermom)(double [], double ,int));
int zbrac(REAL (*func)(REAL), REAL *x1, REAL *x2);
void zbrak(REAL (*fx)(REAL), REAL x1, REAL x2, int n, REAL xb1[],
	REAL xb2[], int *nb);
REAL zbrent(REAL (*func)(REAL), REAL x1, REAL x2, REAL tol);
void zrhqr(REAL a[], int m, REAL rtr[], REAL rti[]);
REAL zriddr(REAL (*func)(REAL), REAL x1, REAL x2, REAL xacc);
void zroots(fcomplex a[], int m, fcomplex roots[], int polish);

#endif /* _NR_H_ */
