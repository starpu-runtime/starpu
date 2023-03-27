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

void print_matrix(int *matrix, int nx, int ny, unsigned ld)
{
	int i, j;
	FPRINTF(stderr, "matrix=%p nx=%d ny=%d ld=%u\n", matrix, nx, ny, ld);
	for(j=0 ; j<ny ; j++)
	{
		for(i=0 ; i<nx ; i++)
		{
			FPRINTF(stderr, "%4d ", matrix[(j*ld)+i]);
		}
		FPRINTF(stderr,"\n");
	}
	FPRINTF(stderr,"\n");
}

void print_matrix_data(starpu_data_handle_t matrix_handle)
{
	int *matrix = (int *)starpu_matrix_get_local_ptr(matrix_handle);
	int nx = starpu_matrix_get_nx(matrix_handle);
	int ny = starpu_matrix_get_ny(matrix_handle);
	unsigned ld = starpu_matrix_get_local_ld(matrix_handle);

	starpu_data_acquire(matrix_handle, STARPU_R);
	print_matrix(matrix, nx, ny, ld);
	starpu_data_release(matrix_handle);
}

void print_2dim_data(starpu_data_handle_t ndim_handle)
{
	int *arr2d = (int *)starpu_ndim_get_local_ptr(ndim_handle);
	unsigned *nn = starpu_ndim_get_nn(ndim_handle);
	unsigned *ldn = starpu_ndim_get_local_ldn(ndim_handle);

	starpu_data_acquire(ndim_handle, STARPU_R);
	print_matrix(arr2d, nn[0], nn[1], ldn[1]);
	starpu_data_release(ndim_handle);
}

void generate_matrix_data(int *matrix, int nx, int ny, unsigned ld)
{
	int i, j, n = 0;
	for(j=0 ; j<ny; j++)
	{
		for(i=0 ; i<nx; i++)
		{
			matrix[(j*ld)+i] = n++;
		}
	}
}

