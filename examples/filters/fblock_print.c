/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void print_block(int *block, int nx, int ny, int nz, unsigned ldy, unsigned ldz)
{
	int i, j, k;
	FPRINTF(stderr, "block=%p nx=%d ny=%d nz=%d ldy=%u ldz=%u\n", block, nx, ny, nz, ldy, ldz);
	for(k=0 ; k<nz ; k++)
	{
		for(j=0 ; j<ny ; j++)
		{
			for(i=0 ; i<nx ; i++)
			{
				FPRINTF(stderr, "%2d ", block[(k*ldz)+(j*ldy)+i]);
			}
			FPRINTF(stderr,"\n");
		}
		FPRINTF(stderr,"\n");
	}
	FPRINTF(stderr,"\n");
}

void print_block_data(starpu_data_handle_t block_handle)
{
	int *block = (int *)starpu_block_get_local_ptr(block_handle);
	int nx = starpu_block_get_nx(block_handle);
	int ny = starpu_block_get_ny(block_handle);
	int nz = starpu_block_get_nz(block_handle);
	unsigned ldy = starpu_block_get_local_ldy(block_handle);
	unsigned ldz = starpu_block_get_local_ldz(block_handle);

	starpu_data_acquire(block_handle, STARPU_R);
	print_block(block, nx, ny, nz, ldy, ldz);
	starpu_data_release(block_handle);
}

void print_3dim_data(starpu_data_handle_t ndim_handle)
{
	int *arr3d = (int *)starpu_ndim_get_local_ptr(ndim_handle);
	unsigned *nn = starpu_ndim_get_nn(ndim_handle);
	unsigned *ldn = starpu_ndim_get_local_ldn(ndim_handle);

	starpu_data_acquire(ndim_handle, STARPU_R);
	print_block(arr3d, nn[0], nn[1], nn[2], ldn[1], ldn[2]);
	starpu_data_release(ndim_handle);
}

void generate_block_data(int *block, int nx, int ny, int nz, unsigned ldy, unsigned ldz)
{
	int i, j, k, n = 0;
	for(k=0 ; k<nz ; k++)
	{
		for(j=0 ; j<ny ; j++)
		{
			for(i=0 ; i<nx ; i++)
			{
				block[(k*ldz)+(j*ldy)+i] = n++;
			}
		}
	}
}
