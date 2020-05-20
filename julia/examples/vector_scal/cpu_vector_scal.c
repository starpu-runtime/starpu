/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <stdio.h>
#include <stdint.h>
#include <starpu.h>
#include <math.h>

struct params {
  int32_t m;
  float k;
  float l;
};

float cpu_vector_scal(void *buffers[], void *cl_arg)
{
  /* get scalar parameters from cl_arg */
  struct params *scalars = (struct params *) cl_arg;
  int m = scalars->m;
  float k = scalars->k;
  float l = scalars->l;

  struct starpu_vector_interface *vector = (struct starpu_vector_interface *) buffers[0];

  /* length of the vector */
  unsigned n = STARPU_VECTOR_GET_NX(vector);

  /* get a pointer to the local copy of the vector : note that we have to
   * cast it in (float *) since a vector could contain any type of
   * elements so that the .ptr field is actually a uintptr_t */
  float *val = (float *)STARPU_VECTOR_GET_PTR(vector);

  /* scale the vector */
  for (unsigned i = 0; i < n; i++)
    val[i] = val[i] * m + l + k;

  return 0.0;
}

char* CPU = "cpu_vector_scal";
char* GPU = "gpu_vector_scal";
extern char *starpu_find_function(char *name, char *device) {
	if (!strcmp(device,"gpu")) return GPU;
	return CPU;
}
