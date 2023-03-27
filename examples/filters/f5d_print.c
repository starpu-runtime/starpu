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

void print_5darr(int *arr5d, int nx, int ny, int nz, int nt, int ng, unsigned ldy, unsigned ldz, unsigned ldt, unsigned ldg)
{
	int i, j, k, l, m;
	FPRINTF(stderr, "5dim array=%p nx=%d ny=%d nz=%d nt=%d ng=%d ldy=%u ldz=%u ldt=%u ldg=%u\n", arr5d, nx, ny, nz, nt, ng, ldy, ldz, ldt, ldg);
	for(m=0 ; m<ng ; m++)
	{
		for(l=0 ; l<nt ; l++)
		{
			for(k=0 ; k<nz ; k++)
			{
				for(j=0 ; j<ny ; j++)
				{
					for(i=0 ; i<nx ; i++)
					{
						FPRINTF(stderr, "%2d ", arr5d[(m*ldg)+(l*ldt)+(k*ldz)+(j*ldy)+i]);
					}
					FPRINTF(stderr,"\n");
				}
				FPRINTF(stderr,"\n");
			}
			FPRINTF(stderr,"\n");
		}
		FPRINTF(stderr,"\n");
	}
	FPRINTF(stderr,"\n");
}

void print_5dim_data(starpu_data_handle_t ndim_handle)
{
	int *arr5d = (int *)starpu_ndim_get_local_ptr(ndim_handle);
	unsigned *nn = starpu_ndim_get_nn(ndim_handle);
	unsigned *ldn = starpu_ndim_get_local_ldn(ndim_handle);

	starpu_data_acquire(ndim_handle, STARPU_R);
	print_5darr(arr5d, nn[0], nn[1], nn[2], nn[3], nn[4], ldn[1], ldn[2], ldn[3], ldn[4]);
	starpu_data_release(ndim_handle);
}

void generate_5dim_data(int *arr5d, int nx, int ny, int nz, int nt, int ng, unsigned ldy, unsigned ldz, unsigned ldt, unsigned ldg)
{
	int i, j, k, l, m, n = 0;
	for(m=0 ; m<ng ; m++)
	{
		for(l=0 ; l<nt ; l++)
		{
			for(k=0 ; k<nz ; k++)
			{
				for(j=0 ; j<ny ; j++)
				{
					for(i=0 ; i<nx ; i++)
					{
						arr5d[(m*ldg)+(l*ldt)+(k*ldz)+(j*ldy)+i] = n++;
					}
				}
			}
		}
	}
}
