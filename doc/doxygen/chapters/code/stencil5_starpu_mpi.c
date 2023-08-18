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

#include <starpu_mpi.h>
#include <math.h>
#include "stencil5.h"

//! [To be included. You should update doxygen if you see this text.]
#define _(row,col,ld) ((row)+(col)*(ld))
void stencil5_cpu(void *descr[], void *_args); // Same as in sequential StarPU
struct starpu_codelet stencil5_cl;  // Same as in sequential StarPU

/* Returns the MPI node number where data indexes index is */
int my_distrib(int x, int y, int nb_nodes)
{
  return ((int)(x / sqrt(nb_nodes) + (y / sqrt(nb_nodes)) * sqrt(nb_nodes))) % nb_nodes;
}

int main(int argc, char **argv)
{
  starpu_data_handle_t *data_handles;
  int niter, n;
  int my_rank, size, x, y, loop;

//! [mpi_init. You should update doxygen if you see this text.]
  int ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
  starpu_mpi_comm_rank(MPI_COMM_WORLD, &my_rank);
  starpu_mpi_comm_size(MPI_COMM_WORLD, &size);
//! [mpi_init. You should update doxygen if you see this text.]

  read_params(argc, argv, &n, &niter);

  double *A = calloc(n*n, sizeof(*A));
  fill(A, n, n);

  data_handles = malloc(n*n*sizeof(*data_handles));
  for(x = 0; x < n; x++)
  {
    for (y = 0; y < n; y++)
    {
//! [mpi_register. You should update doxygen if you see this text.]
      starpu_variable_data_register(&data_handles[_(x,y,n)],
				    STARPU_MAIN_RAM,
				    (uintptr_t)&(A[_(x,y,n)]), sizeof(double));
      int mpi_rank = my_distrib(x, y, size);
      starpu_mpi_data_register(data_handles[_(x,y,n)], (y*n)+x, mpi_rank);
//! [mpi_register. You should update doxygen if you see this text.]
    }
  }

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
//! [mpi_insert. You should update doxygen if you see this text.]
        starpu_mpi_task_insert(MPI_COMM_WORLD, &stencil5_cl,
			       STARPU_RW, data_handles[_(x,y,n)],
			       STARPU_R, data_handles[_(xm1,y,n)],
			       STARPU_R, data_handles[_(xp1,y,n)],
			       STARPU_R, data_handles[_(x,ym1,n)],
			       STARPU_R, data_handles[_(x,yp1,n)],
			       0);
//! [mpi_insert. You should update doxygen if you see this text.]
      }
    }
  }
  starpu_task_wait_for_all();

  /* bring data back to node 0 and unregister it */
  for(x = 0; x < n; x++)
  {
    for (y = 0; y < n; y++)
    {
        starpu_mpi_data_migrate(MPI_COMM_WORLD, data_handles[_(x,y,n)], 0);
        starpu_data_unregister(data_handles[_(x,y,n)]);
    }
  }

  starpu_mpi_shutdown();

  return 0;
}
//! [To be included. You should update doxygen if you see this text.]
