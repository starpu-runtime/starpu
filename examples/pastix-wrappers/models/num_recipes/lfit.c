#define NRANSI
#include <stdio.h>
#include <stdlib.h>
#include "nrutil.h"

#define REAL float

void lfit(REAL x[], REAL y[], REAL sig[], int ndat, REAL a[], int ia[],
	int ma, REAL **covar, REAL *chisq, void (*funcs)(REAL, REAL [], int))
{
	void covsrt(REAL **covar, int ma, int ia[], int mfit);
	void gaussj(REAL **a, int n, REAL **b, int m);
	int i,j,k,l,m,mfit=0;
	REAL ym,wt,sum,sig2i,**beta,*afunc;

	

	beta=matrix(1,ma,1,1);
	afunc=vector(1,ma);
	for (j=1;j<=ma;j++)
		if (ia[j]) mfit++;
	if (mfit == 0) nrerror("lfit: no parameters to be fitted");
	for (j=1;j<=mfit;j++) {
		for (k=1;k<=mfit;k++) covar[j][k]=0.0;
		beta[j][1]=0.0;
	}

	for (i=1;i<=ndat;i++) {
		(*funcs)(x[i],afunc,ma);
		ym=y[i];
		if (mfit < ma) {
			for (j=1;j<=ma;j++)
				if (!ia[j]) ym -= a[j]*afunc[j];
		}
		sig2i=1.0/SQR(sig[i]);
		for (j=0,l=1;l<=ma;l++) {
			if (ia[l]) {
				wt=afunc[l]*sig2i;
				for (j++,k=0,m=1;m<=l;m++)
					if (ia[m]) covar[j][++k] += wt*afunc[m];
				beta[j][1] += ym*wt;
			}
		}
	}
	
	for (j=2;j<=mfit;j++)
		for (k=1;k<j;k++)
			covar[k][j]=covar[j][k];
//	printf("lfit : gaussj\n");	
	gaussj(covar,mfit,beta,1);
//	printf("lfit1\n");
	for (j=0,l=1;l<=ma;l++)
		if (ia[l]) a[l]=beta[++j][1];
//	printf("lfit2\n");
	*chisq=0.0;

	for (i=1;i<=ndat;i++) {
		(*funcs)(x[i],afunc,ma);
		for (sum=0.0,j=1;j<=ma;j++) sum += a[j]*afunc[j];
		*chisq += SQR((y[i]-sum)/sig[i]);
	}
	covsrt(covar,ma,ia,mfit);
	free_vector(afunc,1,ma);
	free_matrix(beta,1,ma,1,1);
}
#undef NRANSI
