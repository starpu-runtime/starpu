/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* Trivial dot reduction OpenCL kernel */

#include "dot_product.h"

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void _redux_opencl(__global DOT_TYPE *dota,
			    __global DOT_TYPE *dotb)
{
	*dota += *dotb;
}

__kernel void _dot_opencl(__global float *x,
			  __global float *y,
			  __global DOT_TYPE *dot,
			  unsigned n)
{
/* FIXME: real parallel implementation */
	unsigned i;
	__local double tmp;
	tmp = 0.0;
	for (i = 0; i < n ; i++)
		tmp += x[i]*y[i];

	*dot += tmp;
}
