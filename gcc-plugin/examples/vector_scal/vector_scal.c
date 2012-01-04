/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012 INRIA
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

/*
 * This is a very simple example that is primarily meant to test the
 * features offered by the StarPU GCC plug-in.
 *
 * Currently tested features :
 * 	- multi-implementations
 *	- CPU codelet
 *
 * Features to test in a near future :
 *	- CUDA
 *	- OpenCL
 *	- Filters
 */
#include <stdio.h>
#include <stdlib.h>

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#define NX     16
#define FACTOR 3.14

static float vector[NX];

static void vector_scal(float *v, size_t n, float factor)
__attribute__ ((task));

static void vector_scal_cpu(float *v, size_t n, float factor)
__attribute__ ((task_implementation ("cpu", vector_scal)));

#ifdef __SSE__
static void vector_scal_sse(float *v, size_t n, float factor)
__attribute__ ((task_implementation ("cpu", vector_scal)));
#endif /* !__SSE__ */

static void
vector_scal_cpu(float *v, size_t n, float factor)
{
	int i;
	for (i = 0; i < n; i++)
		v[i] *= factor;
}

#ifdef __SSE__
static void
vector_scal_sse(float *vector, size_t n, float factor)
{
	unsigned int n_iterations = n/4;

	__m128 *VECTOR = (__m128*) vector;
	__m128 _FACTOR __attribute__((aligned(16)));
	_FACTOR = _mm_set1_ps(factor);

	unsigned int i;	
	for (i = 0; i < n_iterations; i++)
		VECTOR[i] = _mm_mul_ps(_FACTOR, VECTOR[i]);

	unsigned int remainder = n%4;
	if (remainder != 0)
	{
		unsigned int start = 4 * n_iterations;
		for (i = start; i < start+remainder; ++i)
		{
			vector[i] = factor * vector[i];
		}
	}
}
#endif /* !__SSE__ */

static void
init_data(void)
{
	int i;
	for (i = 0; i < NX; i++)
		vector[i] = (float) i;
}

#define EPSILON 1e-3
static int
check(void)
{
	int i;
	for (i = 0; i < NX; i++)
		if (vector[i] - i*FACTOR > EPSILON)
			return 1;

	return 0;
}

int
main(void)
{
#pragma  starpu initialize

	init_data();

#pragma starpu register &vector NX

	vector_scal(vector, NX, FACTOR);

#pragma starpu wait
#pragma starpu unregister &vector
#pragma starpu shutdown

	return check()?EXIT_FAILURE:EXIT_SUCCESS;
}
