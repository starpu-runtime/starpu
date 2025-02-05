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

void print_tensor(int *tensor, size_t nx, size_t ny, size_t nz, size_t nt, size_t ldy, size_t ldz, size_t ldt)
{
	size_t i, j, k, l;
	FPRINTF(stderr, "tensor=%p nx=%zu ny=%zu nz=%zu nt=%zu ldy=%zu ldz=%zu ldt=%zu\n", tensor, nx, ny, nz, nt, ldy, ldz, ldt);
	for(l=0 ; l<nt ; l++)
	{
		for(k=0 ; k<nz ; k++)
		{
			for(j=0 ; j<ny ; j++)
			{
				for(i=0 ; i<nx ; i++)
				{
					FPRINTF(stderr, "%2d ", tensor[(l*ldt)+(k*ldz)+(j*ldy)+i]);
				}
				FPRINTF(stderr,"\n");
			}
			FPRINTF(stderr,"\n");
		}
		FPRINTF(stderr,"\n");
	}
	FPRINTF(stderr,"\n");
}

void print_tensor_data(starpu_data_handle_t tensor_handle)
{
	int *tensor = (int *)starpu_tensor_get_local_ptr(tensor_handle);
	size_t nx = starpu_tensor_get_nx(tensor_handle);
	size_t ny = starpu_tensor_get_ny(tensor_handle);
	size_t nz = starpu_tensor_get_nz(tensor_handle);
	size_t nt = starpu_tensor_get_nt(tensor_handle);
	size_t ldy = starpu_tensor_get_local_ldy(tensor_handle);
	size_t ldz = starpu_tensor_get_local_ldz(tensor_handle);
	size_t ldt = starpu_tensor_get_local_ldt(tensor_handle);

	starpu_data_acquire(tensor_handle, STARPU_R);
	print_tensor(tensor, nx, ny, nz, nt, ldy, ldz, ldt);
	starpu_data_release(tensor_handle);
}

void print_4dim_data(starpu_data_handle_t ndim_handle)
{
	int *arr4d = (int *)starpu_ndim_get_local_ptr(ndim_handle);
	size_t *nn = starpu_ndim_get_nn(ndim_handle);
	size_t *ldn = starpu_ndim_get_local_ldn(ndim_handle);

	starpu_data_acquire(ndim_handle, STARPU_R);
	print_tensor(arr4d, nn[0], nn[1], nn[2], nn[3], ldn[1], ldn[2], ldn[3]);
	starpu_data_release(ndim_handle);
}

void generate_tensor_data(int *tensor, size_t nx, size_t ny, size_t nz, size_t nt, size_t ldy, size_t ldz, size_t ldt)
{
	size_t i, j, k, l, n = 0;
	for(l=0 ; l<nt ; l++)
	{
		for(k=0 ; k<nz ; k++)
		{
			for(j=0 ; j<ny ; j++)
			{
				for(i=0 ; i<nx ; i++)
				{
					tensor[(l*ldt)+(k*ldz)+(j*ldy)+i] = n++;
				}
			}
		}
	}
}
