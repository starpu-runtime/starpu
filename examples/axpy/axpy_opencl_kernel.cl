/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* OpenCL kernel implementing axpy */

#include "axpy.h"

__kernel void _axpy_opencl(__global TYPE *x,
			   unsigned x_offset,
			   __global TYPE *y,
			   unsigned y_offset,
			   unsigned nx,
			   TYPE alpha)
{
        const int i = get_global_id(0);
        x = (__global char*) x + x_offset;
        y = (__global char*) y + y_offset;
        if (i < nx)
                y[i] = alpha * x[i] + y[i];
}
