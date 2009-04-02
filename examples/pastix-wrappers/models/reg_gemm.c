/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "nr.h"

#define REAL float
#define SIZE 2000000
#define ma 6


typedef struct Coord
{
  REAL x1;
  REAL x2;
  REAL x3;
} coord, *pcoord;


coord tabcoord[SIZE];
REAL tabx[SIZE];
REAL taby[SIZE];  
REAL sig[SIZE];
REAL afunc[ma+1];
int ia[ma+1];
REAL a[ma+1];


void funcs( REAL i, REAL afunc[ma+1], int ma2)
{
  
  afunc[1]= 1;
  afunc[2]= tabcoord[(int)i].x1;
  afunc[3]= tabcoord[(int)i].x2;
  afunc[4]= tabcoord[(int)i].x1*tabcoord[(int)i].x2;
  afunc[5]= tabcoord[(int)i].x2*tabcoord[(int)i].x3;
  afunc[6]= tabcoord[(int)i].x1*tabcoord[(int)i].x2*tabcoord[(int)i].x3;
  //printf("%f %f %f \n",afunc[0],afunc[1],afunc[2]);
}


int main(int argc, char * argv[])
{
  REAL total=0.0;
  REAL ecart=0.0;
  int len=0;
  char str2[1000];
/*   long double total=0.0; */
/*   long double ecart=0.0; */
  char *filename = argv[1];
  char perf_h[1000];
  char str[1000];
  int k,i;
  FILE * res;
  FILE *out;
  FILE *perf;
  // FILE *tmpperf;
  int ndat=atoi(argv[2]);
  REAL ** covar;
  REAL *chisq = (REAL *)malloc(sizeof(REAL));
  res = fopen(filename,"r");

  covar = (REAL**) malloc((ma+1) *sizeof(REAL*));
  for (i=0;i<ma+1;i++)
    covar[i]=(REAL*)malloc((ma+1) *sizeof(REAL));


  for (k=1;k<ndat+1;k++)
    {
      int x0, y0, x1, y1, x2, y2;
      REAL tmpfloat;
      fscanf(res,"%f\t%d\t%d\t%d\t%d\t%d\t%d\n", &tmpfloat, &x0, &y0, &x1, &y1, &x2, &y2);
      tabcoord[k].x1= x0 - y0; 
      tabcoord[k].x2= y2;
      tabcoord[k].x3= y0;
      taby[k]=tmpfloat;
      sig[k]=1;
      tabx[k]=k;
      //fprintf(out,"%f %f %f\n",tabcoord[k].x1 ,tabcoord[k].x2 ,tabcoord[k].x3);
      //fprintf(out,"%f %f\n",tabx[k] , taby[k]);
    }
  for (k=1;k<ma+1;k++)
    ia[k]=1;


  lfit(tabx, taby, sig, ndat, a, ia, ma, covar, chisq, &funcs);
  
  for (k=1;k<ma+1;k++)  
    {    
//      printf("%.12lf\n", a[k]);
      //total+=a[k];
    }



  //calcul de l'ecart type
  for (k=1;k<ndat+1;k++)
    {
      double abs=0.0;
      abs += a[1];
      abs += tabcoord[k].x1*a[2]+tabcoord[k].x2*a[3];
      abs += tabcoord[k].x1*tabcoord[k].x2*a[4]+tabcoord[k].x2*tabcoord[k].x3*a[5];
      abs += tabcoord[k].x1*tabcoord[k].x2*tabcoord[k].x3*a[6];
    //  fprintf(stderr,"k=%i ; calcul : %lf ; reel : %lf ; ", k, abs, taby[k]);
      abs = abs - taby[k];
      if (abs < 0)
	abs = - abs;
    //  fprintf(stderr,"ecart : %lf\n ", abs);

      total += abs;
      //printf("%f %f %f\n",tabcoord[k].x1 ,tabcoord[k].x2 ,tabcoord[k].x3);
    }



  fprintf(stdout,"#define GEMM_A  %e\n#define GEMM_B  %e\n#define GEMM_C  %e\n#define GEMM_D  %e\n#define GEMM_E  %e\n#define GEMM_F  %e\n",a[6],a[4],a[5],a[2],a[3],a[1]);
  fprintf(stderr,"#define PERF_GEMM(i,j,k) (GEMM_A*(double)(i)*(double)(j)*(double)(k)+GEMM_B*(double)(i)*(double)(j)+GEMM_C*(double)(j)*(double)(k)+GEMM_D*(double)(i)+GEMM_E*(double)(j)+GEMM_F)\n");


  fprintf(stderr, "total %lf\n", total);
  ecart = total / ndat;  
  fprintf(stderr, "ecart moyen %lf\n", ecart);

  return 0;
}
