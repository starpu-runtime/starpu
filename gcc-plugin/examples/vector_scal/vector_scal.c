/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012 Institut National de Recherche en Informatique et Automatique
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
#ifndef STARPU_GCC_PLUGIN
# error must be compiled with the StarPU GCC plug-in
#endif

/* This is a simple example that is primarily meant to test the features
   offered by the StarPU GCC plug-in.

   Currently tested features :
	- multi-implementations
	- CPU codelet

   Features to test in a near future :
	- CUDA
	- OpenCL
	- Filters
 */
#include <stdio.h>
#include <stdlib.h>

/* Declare and define the standard CPU implementation.  */

static void vector_scal (size_t size, float vector[size], float factor)
  __attribute__ ((task));

static void vector_scal_cpu (size_t size, float vector[size], float factor)
  __attribute__ ((task_implementation ("cpu", vector_scal)));

static void
vector_scal_cpu (size_t size, float vector[size], float factor)
{
  size_t i;
  for (i = 0; i < size; i++)
    vector[i] *= factor;
}

#ifdef __SSE__
/* The SSE-capable CPU implementation.  */

#include <xmmintrin.h>

static void vector_scal_sse (size_t size, float vector[size], float factor)
  __attribute__ ((task_implementation ("cpu", vector_scal)));

static void
vector_scal_sse (size_t size, float vector[size], float factor)
{
  unsigned int n_iterations = size / 4;

  __m128 *VECTOR = (__m128 *) vector;
  __m128 _FACTOR __attribute__ ((aligned (16)));
  _FACTOR = _mm_set1_ps (factor);

  unsigned int i;
  for (i = 0; i < n_iterations; i++)
    VECTOR[i] = _mm_mul_ps (_FACTOR, VECTOR[i]);

  unsigned int remainder = size % 4;
  if (remainder != 0)
    {
      unsigned int start = 4 * n_iterations;
      for (i = start; i < start + remainder; ++i)
	vector[i] = factor * vector[i];
    }
}
#endif /* __SSE__ */

#define EPSILON 1e-3
static int
check (size_t size, float vector[size], float factor)
{
  size_t i;
  for (i = 0; i < size; i++)
    if (vector[i] - i * factor > EPSILON)
      return 1;

  return 0;
}


int
main (void)
{
#pragma starpu initialize

#define NX     0x100000
#define FACTOR 3.14

  float vector[NX] __attribute__ ((heap_allocated));

#pragma starpu register vector

  size_t i;
  for (i = 0; i < NX; i++)
    vector[i] = (float) i;

  vector_scal (NX, vector, FACTOR);

#pragma starpu wait
#pragma starpu shutdown

  return check (NX, vector, FACTOR) ? EXIT_FAILURE : EXIT_SUCCESS;
}
