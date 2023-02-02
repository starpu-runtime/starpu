/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2023-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define NX 2
#define NY 3
#define NZ 2
#define NT 2

void arr4d_cpu_func(void *buffers[], void *args)
{
	(void)args;
	int *arr4d = (int *) STARPU_NDIM_GET_PTR(buffers[0]);
	int *nn = (int *)STARPU_NDIM_GET_NN(buffers[0]);
	unsigned *ldn = STARPU_NDIM_GET_LDN(buffers[0]);
	int nx = nn[0];
	int ny = nn[1];
	int nz = nn[2];
	int nt = nn[3];
	unsigned ldy = ldn[1];
	unsigned ldz = ldn[2];
	unsigned ldt = ldn[3];
	int i, j, k, l;
	for (l = 0; l < nt; l++)
	{
		for (k = 0; k < nz; k++)
		{
			for (j = 0; j < ny; j++)
			{
				for (i = 0; i < nx; i++)
				{
					arr4d[(l*ldt)+(k*ldz)+(j*ldy)+i] *= 10;
				}
			}
		}
	}
}

struct starpu_codelet arr4d_cl =
{
	.cpu_funcs = {arr4d_cpu_func},
	.cpu_funcs_name = {"arr4d_cpu_func"},
	.nbuffers = 1,
	.modes = { STARPU_RW },
	.name = "arr4d_cl"
};

int main(void)
{
	int ret;
	int arr4d[NX*NY*NZ*NT];
	int val = 0;
	int i, j, k, l;
	starpu_data_handle_t arr4d_handle;
	unsigned nn[4] = {NX, NY, NZ, NT};
	unsigned ldn[4] = {1, NX, NX*NY, NX*NY*NZ};

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	for (l = 0; l < NT; l++)
		for (k = 0; k < NZ; k++)
			for (j = 0; j < NY; j++)
				for (i = 0; i < NX; i++)
					arr4d[(l*NX*NY*NZ)+(k*NX*NY)+(j*NX)+i] = val++;

	starpu_ndim_data_register(&arr4d_handle, STARPU_MAIN_RAM, (uintptr_t)arr4d, ldn, nn, 4, sizeof(arr4d[0]));

	ret = starpu_task_insert(&arr4d_cl,
				 STARPU_RW, arr4d_handle,
				 0);
	if (ret == -ENODEV)
		goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_data_unregister(arr4d_handle);
	for (l = 0; l < NT; l++)
	{
		fprintf(stderr, "------\n");
		for (k = 0; k < NZ; k++)
		{
			for (j = 0; j < NY; j++)
			{
				fprintf(stderr, "|\t");
				for (i = 0; i < NX; i++)
					fprintf(stderr, "%d\t", arr4d[(l*NX*NY*NZ)+(k*NX*NY)+(j*NX)+i]);
				fprintf(stderr, " |");
			}
			fprintf(stderr, "\n");
		}
		fprintf(stderr, "------\n");
	}
	starpu_shutdown();

	return 0;

enodev:
	starpu_data_unregister(arr4d_handle);
	starpu_shutdown();
	return 77;
}
