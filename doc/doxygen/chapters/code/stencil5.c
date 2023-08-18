/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <math.h>
#include "stencil5.h"

//! [To be included. You should update doxygen if you see this text.]
#define _(row,col,ld) ((row)+(col)*(ld))

void stencil5_cpu(double *xy, double *xm1y, double *xp1y, double *xym1, double *xyp1)
{
  *xy = (*xy + *xm1y + *xp1y + *xym1 + *xyp1) / 5;
}

int main(int argc, char **argv)
{
  int niter, n;
  int x, y, loop;

  read_params(argc, argv, &n, &niter);

  double *A = calloc(n*n, sizeof(*A));
  fill(A, n, n);

  for(loop=0 ; loop<niter; loop++)
  {
    for (x = 0; x < n; x++)
    {
      for (y = 0; y < n; y++)
      {
        int xm1 = (x==0) ? n-1 : x-1;
        int xp1 = (x==n-1) ? 0 : x+1;
        int ym1 = (y==0) ? n-1 : y-1;
        int yp1 = (y==n-1) ? 0 : y+1;
        stencil5_cpu(&A[_(x,y,n)],
		     &A[_(xm1,y,n)], &A[_(xp1,y,n)],
		     &A[_(x,ym1,n)], &A[_(x,yp1,n)]);
      }
    }
  }

  return 0;
}
//! [To be included. You should update doxygen if you see this text.]
