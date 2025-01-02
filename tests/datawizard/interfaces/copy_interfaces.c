/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../../helper.h"
#include <datawizard/coherency.h>

static int check_copy(starpu_data_handle_t handle, char *header)
{
	void *old_interface, *new_interface;
	starpu_data_handle_t new_handle;
	int ret=0;

	starpu_data_register_same(&new_handle, handle);

	if (!getenv("STARPU_SSILENT"))
	{
		if (new_handle->ops->display)
		{
			fprintf(stderr, "%s: ", header);
			new_handle->ops->display(new_handle, stderr);
			fprintf(stderr, "\n");
		}
		else
		{
			fprintf(stderr, "%s does not define a display ops\n", header);
		}
	}

	old_interface = starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	new_interface = starpu_data_get_interface_on_node(new_handle, STARPU_MAIN_RAM);

	if (new_handle->ops->compare(old_interface, new_interface) == 0)
	{
		FPRINTF(stderr, "Error when copying %s data\n", header);
		ret = 1;
	}
	starpu_data_unregister(handle);
	starpu_data_unregister(new_handle);
	return ret;
}

int main(int argc, char **argv)
{
	int ret;
	starpu_data_handle_t handle;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (ret == 0)
	{
		int NX=3;
		int NY=2;
		int matrix[NX][NY];
		starpu_matrix_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)matrix, NX, NX, NY, sizeof(matrix[0][0]));
		ret = check_copy(handle, "matrix");
	}

	if (ret == 0)
	{
		int NX=3;
		int NY=2;
		int NZ=4;
		int block[NX*NY*NZ];
		starpu_block_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)block, NX, NX*NY, NX, NY, NZ, sizeof(block[0]));
		ret = check_copy(handle, "block");
	}

	if (ret == 0)
	{
		int xx[] = {12, 23, 45};
		starpu_vector_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)xx, 3, sizeof(xx[0]));
		ret = check_copy(handle, "vector");
	}

	if (ret == 0)
	{
		uint32_t nnz = 2;
		unsigned nrow = 5;
		float nzvalA[nnz];
		uint32_t colind[nnz];
		uint32_t rowptr[nrow+1];
		starpu_csr_data_register(&handle, STARPU_MAIN_RAM, nnz, nrow, (uintptr_t)nzvalA, colind, rowptr, 0, sizeof(float));
		ret = check_copy(handle, "csr");
	}

	if (ret == 0)
	{
		uint32_t nnz = 2;
		unsigned nrow = 5;
		float nzvalA[nnz];
		uint32_t colind[nnz];
		uint32_t rowptr[nrow+1];
		starpu_bcsr_data_register(&handle, STARPU_MAIN_RAM, nnz, nrow, (uintptr_t)nzvalA, colind, rowptr, 0, 1, 1, sizeof(float));
		ret = check_copy(handle, "bcsr");
	}

	if (ret == 0)
	{
		int x=42;
		starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&x, sizeof(x));
		ret = check_copy(handle, "variable");
	}

	if (ret == 0)
	{
		int NX=3;
		int NY=2;
		int NZ=4;
		int NT=3;
		int tensor[NX*NY*NZ*NT];
		starpu_tensor_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)tensor, NX, NX*NY, NX*NY*NZ, NX, NY, NZ, NT, sizeof(tensor[0]));
		ret = check_copy(handle, "tensor");
	}

	if (ret == 0)
	{
		int NX=3;
		int NY=2;
		int array2d[NX*NY];
		unsigned nn[2] = {NX, NY};
		unsigned ldn[2] = {1, NX};
		starpu_ndim_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)array2d, ldn, nn, 2, sizeof(int));
		ret = check_copy(handle, "ndim");
	}

	starpu_shutdown();
	return ret;
}
