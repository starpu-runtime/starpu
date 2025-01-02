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

#include <starpu.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void print_block(int *block, size_t nx, size_t ny, size_t nz, size_t ldy, size_t ldz)
{
	size_t i, j, k;
	FPRINTF(stderr, "block=%p nx=%zu ny=%zu nz=%zu ldy=%zu ldz=%zu\n", block, nx, ny, nz, ldy, ldz);
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
	size_t nx = starpu_block_get_nx(block_handle);
	size_t ny = starpu_block_get_ny(block_handle);
	size_t nz = starpu_block_get_nz(block_handle);
	size_t ldy = starpu_block_get_local_ldy(block_handle);
	size_t ldz = starpu_block_get_local_ldz(block_handle);

	starpu_data_acquire(block_handle, STARPU_R);
	print_block(block, nx, ny, nz, ldy, ldz);
	starpu_data_release(block_handle);
}

void print_3dim_data(starpu_data_handle_t ndim_handle)
{
	int *arr3d = (int *)starpu_ndim_get_local_ptr(ndim_handle);
	size_t *nn = starpu_ndim_get_nn(ndim_handle);
	size_t *ldn = starpu_ndim_get_local_ldn(ndim_handle);

	starpu_data_acquire(ndim_handle, STARPU_R);
	print_block(arr3d, nn[0], nn[1], nn[2], ldn[1], ldn[2]);
	starpu_data_release(ndim_handle);
}

void generate_block_data(int *block, size_t nx, size_t ny, size_t nz, size_t ldy, size_t ldz)
{
	size_t i, j, k, n = 0;
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
