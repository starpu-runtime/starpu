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

#include "multiformat_types.h"
__kernel void multiformat_opencl(__global struct struct_of_arrays *soa,
				 unsigned int nx,
				 __global int *err,
				 int factor)
{
        const int i = get_global_id(0);
	if (i >= nx)
		return;

	if (soa->x[i] != i * factor || soa->y[i] != i * factor)
	{
		*err = i;
	}
	else
	{
		soa->x[i] = -soa->x[i];
		soa->y[i] = -soa->y[i];
	}
}
