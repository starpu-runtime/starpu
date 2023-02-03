/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* Use the "double" type */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void complex_copy_opencl(__global double *o_real,
				  unsigned o_real_offset,
				  __global double *o_imaginary,
				  unsigned o_imaginary_offset,
				  __global double *i_real,
				  unsigned i_real_offset,
				  __global double *i_imaginary,
				  unsigned i_imaginary_offset,
				  unsigned nx)
{
	const int i = get_global_id(0);
	if (i < nx)
	{
		o_real = (__global char*) o_real + o_real_offset;
		o_imaginary = (__global char*) o_imaginary + o_imaginary_offset;
		i_real = (__global char*) i_real + i_real_offset;
		i_imaginary = (__global char*) i_imaginary + i_imaginary_offset;

		o_real[i] = i_real[i];
		o_imaginary[i] = i_imaginary[i];
	}
}
