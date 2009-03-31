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

#define SIZE 100000
#define ma 3


typedef struct Coord
{
  float x1;
  float x2;
} coord, *pcoord;


coord tabcoord[SIZE];
float tabx[SIZE];
float taby[SIZE];  
float sig[SIZE];
float afunc[ma+1];
int ia[ma+1];
float a[ma+1];


void funcs( float i, float afunc[ma+1], int ma2)
{
  
  afunc[1]=1;
  afunc[2]=tabcoord[(int)i].x1;
  afunc[3]=tabcoord[(int)i].x1*tabcoord[(int)i].x1*tabcoord[(int)i].x2;
  
  //printf("%f %f %f \n",afunc[0],afunc[1],afunc[2]);
}


int main(int argc, char * argv[])
{
  float total=0.0;
  float ecart=0.0;
/*   long double total=0.0; */
/*   long double ecart=0.0; */
  char *filename = argv[1];
  int k,i;
  FILE * res;
  int ndat=atoi(argv[2]);
  float ** covar;
  float *chisq = (float *)malloc(sizeof(float));
  res = fopen(filename,"r");
  covar = (float**) malloc((ma+1) *sizeof(float*));
  for (i=0;i<ma+1;i++)
    covar[i]=(float*)malloc((ma+1) *sizeof(float));

  for (k=1;k<ndat+1;k++)
    {
      int i, j;
      float tmpfloat;
      fscanf(res,"%f\t%d\t%d\n", &tmpfloat, &i, &j);
      tabcoord[k].x1=i-j; 
      tabcoord[k].x2=j;
      taby[k]=tmpfloat;
//      printf("%d -> %f %d %d\n", k, tmpfloat, i-j, j);
      sig[k]=1;
      tabx[k]=k;
    }
  for (k=1;k<ma+1;k++)
    ia[k]=1;
  
  lfit(tabx, taby, sig, ndat, a, ia, ma, covar, chisq, &funcs);
  
  for (k=1;k<ma+1;k++)  
    {    
  //    printf("%.12lf\n", a[k]);
      //total+=a[k];
    }



  //calcul de l'ecart type
  for (k=1;k<ndat+1;k++)
    {
      double abs=0.0;
      abs += a[1];
      abs += tabcoord[k].x1*a[2];
      abs += tabcoord[k].x1*tabcoord[k].x1*tabcoord[k].x2*a[3];
//      fprintf(stderr,"k=%i ; calcul : %lf ; reel : %lf ; ", k, abs, taby[k]);
      abs = abs - taby[k];
      if (abs < 0)
	abs = - abs;
 //     fprintf(stderr,"ecart : %lf\n ", abs);

      total += abs;
      //printf("%f %f %f\n",tabcoord[k].x1 ,tabcoord[k].x2 ,tabcoord[k].x3);
    }
  
  fprintf(stdout,"#define TRSM_A %e\n#define TRSM_B %e\n#define TRSM_C %e\n", a[3], a[2], a[1]);
  fprintf(stderr,"#define PERF_TRSM(i,j)   (TRSM_A*(double)(i)*(double)(i)*(double)(j)+TRSM_B*(double)(i)+TRSM_C)\n");


  fprintf(stderr, "total %lf\n", total);
  ecart = total / ndat;  
  fprintf(stderr, "ecart moyen %lf\n", ecart);

  return 0;
}
