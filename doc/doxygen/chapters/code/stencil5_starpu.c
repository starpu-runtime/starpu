/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <math.h>
#include "stencil5.h"

//! [To be included. You should update doxygen if you see this text.]
//! [starpu_codelet. You should update doxygen if you see this text.]
void stencil5_cpu(void *descr[], void *_args)
{
  (void)_args;
  double *xy = (double *)STARPU_VARIABLE_GET_PTR(descr[0]);
  double *xm1y = (double *)STARPU_VARIABLE_GET_PTR(descr[1]);
  double *xp1y = (double *)STARPU_VARIABLE_GET_PTR(descr[2]);
  double *xym1 = (double *)STARPU_VARIABLE_GET_PTR(descr[3]);
  double *xyp1 = (double *)STARPU_VARIABLE_GET_PTR(descr[4]);

  *xy = (*xy + *xm1y + *xp1y + *xym1 + *xyp1) / 5;
}

struct starpu_codelet stencil5_cl =
{
  .cpu_funcs = {stencil5_cpu},
  .nbuffers = 5,
  .modes = {STARPU_RW, STARPU_R, STARPU_R, STARPU_R, STARPU_R},
  .model = &starpu_perfmodel_nop,
};
//! [starpu_codelet. You should update doxygen if you see this text.]

int main(int argc, char **argv)
{
  starpu_data_handle_t *data_handles;
  int ret;
  int niter, n;
  int x, y, loop;

  ret = starpu_init(NULL);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

  read_params(argc, argv, &verbose, &n, &niter);

  double *A = calloc(n*n, sizeof(*A));
  fill(A, n, n);

//! [starpu_register. You should update doxygen if you see this text.]
  data_handles = malloc(n*n*sizeof(*data_handles));
  for(x = 0; x < n; x++)
  {
    for (y = 0; y < n; y++)
    {
      starpu_variable_data_register(&data_handles[_(x,y,n)],
				    STARPU_MAIN_RAM,
				    (uintptr_t)&(A[_(x,y,n)]), sizeof(double));
    }
  }
//! [starpu_register. You should update doxygen if you see this text.]

  for(loop=0 ; loop<niter; loop++)
  {
    for (x = 0; x < n; x++)
    {
      for (y = 0; y < n; y++)
      {
//! [starpu_task. You should update doxygen if you see this text.]
        int xm1 = (x==0) ? n-1 : x-1;
        int xp1 = (x==n-1) ? 0 : x+1;
        int ym1 = (y==0) ? n-1 : y-1;
        int yp1 = (y==n-1) ? 0 : y+1;
        starpu_task_insert(&stencil5_cl,
			   STARPU_RW, data_handles[_(x,y,n)],
			   STARPU_R, data_handles[_(xm1,y,n)],
			   STARPU_R, data_handles[_(xp1,y,n)],
			   STARPU_R, data_handles[_(x,ym1,n)],
			   STARPU_R, data_handles[_(x,yp1,n)],
			   0);
//! [starpu_task. You should update doxygen if you see this text.]
      }
    }
  }
  starpu_task_wait_for_all();

//! [starpu_unregister. You should update doxygen if you see this text.]
  for(x = 0; x < n; x++)
  {
    for (y = 0; y < n; y++)
    {
      starpu_data_unregister(data_handles[_(x,y,n)]);
    }
  }
//! [starpu_unregister. You should update doxygen if you see this text.]

  starpu_shutdown();
  return 0;
}
//! [To be included. You should update doxygen if you see this text.]
