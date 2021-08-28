/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
__kernel void arr4d_opencl(__global int *arr4d,
			   int nx, int ny, int nz, int nt,
			   int ldy, int ldz, int ldt,
			   int factor, __global int *err)
{
	const int idx = get_global_id(0);
	const int idy = get_global_id(1);
	const int idz = get_global_id(2) % nz;
	const int idt = get_global_id(2) / nz;
	if (idx >= nx)
		return;
	if (idy >= ny)
		return;
	if (idz >= nz)
		return;
	if (idt >= nt)
		return;

	int val = idt*nz*ny*nx+idz*ny*nx+idy*nx+idx;
	int i = (idt*ldt)+(idz*ldz)+(idy*ldy)+idx;

	if (arr4d[i] != factor * val)
		*err = 1;
	else
		arr4d[i] *= -1;
}
