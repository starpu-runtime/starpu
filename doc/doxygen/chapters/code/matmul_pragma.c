/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2010-2013  Université de Bordeaux 1
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

//! [To be included. You should update doxygen if you see that text.]
/* This program is valid, whether or not StarPU's GCC plug-in
   is being used.  */

#include <stdlib.h>

/* The attribute below is ignored when GCC is not used.  */
static void matmul (const float *A, const float *B, float * C,
                    unsigned nx, unsigned ny, unsigned nz)
  __attribute__ ((task));

static void
matmul (const float *A, const float *B, float * C,
        unsigned nx, unsigned ny, unsigned nz)
{
  /* Code of the CPU kernel here...  */
}

#ifdef STARPU_GCC_PLUGIN
/* Optional OpenCL task implementation.  */

static void matmul_opencl (const float *A, const float *B, float * C,
                           unsigned nx, unsigned ny, unsigned nz)
  __attribute__ ((task_implementation ("opencl", matmul)));

static void
matmul_opencl (const float *A, const float *B, float * C,
               unsigned nx, unsigned ny, unsigned nz)
{
  /* Code that invokes the OpenCL kernel here...  */
}
#endif

int
main (int argc, char *argv[])
{
  /* The pragmas below are simply ignored when StarPU-GCC
     is not used.  */
#pragma starpu initialize

  float A[123][42][7], B[123][42][7], C[123][42][7];

#pragma starpu register A
#pragma starpu register B
#pragma starpu register C

  /* When StarPU-GCC is used, the call below is asynchronous;
     otherwise, it is synchronous.  */
  matmul ((float *) A, (float *) B, (float *) C, 123, 42, 7);

#pragma starpu wait
#pragma starpu shutdown

  return EXIT_SUCCESS;
}
//! [To be included. You should update doxygen if you see that text.]
