/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void print_tensor(int *tensor, int nx, int ny, int nz, int nt, unsigned ldy, unsigned ldz, unsigned ldt)
{
	int i, j, k, l;
	FPRINTF(stderr, "tensor=%p nx=%d ny=%d nz=%d nt=%d ldy=%u ldz=%u ldt=%u\n", tensor, nx, ny, nz, nt, ldy, ldz, ldt);
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
	int nx = starpu_tensor_get_nx(tensor_handle);
	int ny = starpu_tensor_get_ny(tensor_handle);
	int nz = starpu_tensor_get_nz(tensor_handle);
	int nt = starpu_tensor_get_nt(tensor_handle);
	unsigned ldy = starpu_tensor_get_local_ldy(tensor_handle);
	unsigned ldz = starpu_tensor_get_local_ldz(tensor_handle);
	unsigned ldt = starpu_tensor_get_local_ldt(tensor_handle);

	print_tensor(tensor, nx, ny, nz, nt, ldy, ldz, ldt);
}

void print_4dim_data(starpu_data_handle_t ndim_handle)
{
	int *arr4d = (int *)starpu_ndim_get_local_ptr(ndim_handle);
	unsigned *nn = starpu_ndim_get_nn(ndim_handle);
	unsigned *ldn = starpu_ndim_get_local_ldn(ndim_handle);

	print_tensor(arr4d, nn[0], nn[1], nn[2], nn[3], ldn[1], ldn[2], ldn[3]);
}

void generate_tensor_data(int *tensor, int nx, int ny, int nz, int nt, unsigned ldy, unsigned ldz, unsigned ldt)
{
	int i, j, k, l, n = 0;
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
