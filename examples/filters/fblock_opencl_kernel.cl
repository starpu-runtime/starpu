/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* dumb OpenCL kernel to fill a 3D matrix */

__kernel void fblock_opencl(__global int* block, unsigned offset, int nx, int ny, int nz, unsigned ldy, unsigned ldz, int factor)
{
	const int idx = get_global_id(0);
	const int idy = get_global_id(1);
	const int idz = get_global_id(2);
	if (idx >= nx)
		return;
	if (idy >= ny)
		return;
	if (idz >= nz)
		return;

	block = (__global int*) ((__global char *)block + offset);
	int i = idz*ldz + idy*ldy + idx;
	block[i] = factor;
}
