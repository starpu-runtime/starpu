/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Centre National de la Recherche Scientifique
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

#include <starpu_mpi.h>
#include <stdlib.h>
#include "helper.h"

typedef void (*check_func)(starpu_data_handle_t handle_s, starpu_data_handle_t handle_r, int *error);

void check_variable(starpu_data_handle_t handle_s, starpu_data_handle_t handle_r, int *error)
{
	int ret;
	float *v_s, *v_r;

	STARPU_ASSERT(starpu_variable_get_elemsize(handle_s) == starpu_variable_get_elemsize(handle_r));

	v_s = (float *)starpu_variable_get_local_ptr(handle_s);
	v_r = (float *)starpu_variable_get_local_ptr(handle_r);

	if (*v_s == *v_r)
	{
		FPRINTF_MPI("Success with variable value: %f == %f\n", *v_s, *v_r);
	}
	else
	{
		*error = 1;
		FPRINTF_MPI("Error with variable value: %f != %f\n", *v_s, *v_r);
	}
}

void check_vector(starpu_data_handle_t handle_s, starpu_data_handle_t handle_r, int *error)
{
	int ret, i;
	int nx;
	int *v_r, *v_s;

	STARPU_ASSERT(starpu_vector_get_elemsize(handle_s) == starpu_vector_get_elemsize(handle_r));
	STARPU_ASSERT(starpu_vector_get_nx(handle_s) == starpu_vector_get_nx(handle_r));

	nx = starpu_vector_get_nx(handle_r);
	v_r = (int *)starpu_vector_get_local_ptr(handle_r);
	v_s = (int *)starpu_vector_get_local_ptr(handle_s);

	for(i=0 ; i<nx ; i++)
	{
		if (v_s[i] == v_r[i])
		{
			FPRINTF_MPI("Success with vector[%d] value: %d == %d\n", i, v_s[i], v_r[i]);
		}
		else
		{
			*error = 1;
			FPRINTF_MPI("Error with vector[%d] value: %d != %d\n", i, v_s[i], v_r[i]);
		}
	}
}

void check_matrix(starpu_data_handle_t handle_s, starpu_data_handle_t handle_r, int *error)
{
	STARPU_ASSERT(starpu_matrix_get_elemsize(handle_s) == starpu_matrix_get_elemsize(handle_r));
	STARPU_ASSERT(starpu_matrix_get_nx(handle_s) == starpu_matrix_get_nx(handle_r));
	STARPU_ASSERT(starpu_matrix_get_ny(handle_s) == starpu_matrix_get_ny(handle_r));
	STARPU_ASSERT(starpu_matrix_get_local_ld(handle_s) == starpu_matrix_get_local_ld(handle_r));

	char *matrix_s = (char *)starpu_matrix_get_local_ptr(handle_s);
	char *matrix_r = (char *)starpu_matrix_get_local_ptr(handle_r);

	int nx = starpu_matrix_get_nx(handle_s);
	int ny = starpu_matrix_get_ny(handle_s);
	int ldy = starpu_matrix_get_local_ld(handle_s);

	int x, y;

	for(y=0 ; y<ny ; y++)
		for(x=0 ; x<nx ; x++)
		{
			int index=(y*ldy)+x;
			if (matrix_s[index] == matrix_r[index])
			{
				FPRINTF_MPI("Success with matrix[%d,%d --> %d] value: %c == %c\n", x, y, index, matrix_s[index], matrix_r[index]);
			}
			else
			{
				*error = 1;
				FPRINTF_MPI("Error with matrix[%d,%d --> %d] value: %c != %c\n", x, y, index, matrix_s[index], matrix_r[index]);
			}
		}
}

void check_block(starpu_data_handle_t handle_s, starpu_data_handle_t handle_r, int *error)
{
	STARPU_ASSERT(starpu_block_get_elemsize(handle_s) == starpu_block_get_elemsize(handle_r));
	STARPU_ASSERT(starpu_block_get_nx(handle_s) == starpu_block_get_nx(handle_r));
	STARPU_ASSERT(starpu_block_get_ny(handle_s) == starpu_block_get_ny(handle_r));
	STARPU_ASSERT(starpu_block_get_nz(handle_s) == starpu_block_get_nz(handle_r));
	STARPU_ASSERT(starpu_block_get_local_ldy(handle_s) == starpu_block_get_local_ldy(handle_r));
	STARPU_ASSERT(starpu_block_get_local_ldz(handle_s) == starpu_block_get_local_ldz(handle_r));

	float *block_s = (float *)starpu_block_get_local_ptr(handle_s);
	float *block_r = (float *)starpu_block_get_local_ptr(handle_r);

	int nx = starpu_block_get_nx(handle_s);
	int ny = starpu_block_get_ny(handle_s);
	int nz = starpu_block_get_nz(handle_s);

	int ldy = starpu_block_get_local_ldy(handle_s);
	int ldz = starpu_block_get_local_ldz(handle_s);

	int x, y, z;

	for(z=0 ; z<nz ; z++)
		for(y=0 ; y<ny ; y++)
			for(x=0 ; x<nx ; x++)
			{
				int index=(z*ldz)+(y*ldy)+x;
				if (block_s[index] == block_r[index])
				{
					FPRINTF_MPI("Success with block[%d,%d,%d --> %d] value: %f == %f\n", x, y, z, index, block_s[index], block_r[index]);
				}
				else
				{
					*error = 1;
					FPRINTF_MPI("Error with block[%d,%d,%d --> %d] value: %f != %f\n", x, y, z, index, block_s[index], block_r[index]);
				}
			}
}

void send_recv_and_check(int rank, int node, starpu_data_handle_t handle_s, int tag_s, starpu_data_handle_t handle_r, int tag_r, int *error, check_func func)
{
	int ret;
	MPI_Status status;

	if (rank == 0)
	{
		ret = starpu_mpi_send(handle_s, node, tag_s, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
		ret = starpu_mpi_recv(handle_r, node, tag_r, MPI_COMM_WORLD, &status);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");

		func(handle_s, handle_r, error);
	}
	else
	{
		ret = starpu_mpi_recv(handle_s, node, tag_s, MPI_COMM_WORLD, &status);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		ret = starpu_mpi_send(handle_s, node, tag_r, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
	}
}

int main(int argc, char **argv)
{
	int ret, rank, size;
	int error=0;

	int nx=3;
	int ny=2;
	int nz=4;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size < 2)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 2 processes.\n");

		MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ret = starpu_mpi_init(NULL, NULL, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");

	if (rank == 0)
	{
		MPI_Status status;

		{
			float v = 42.12;
			starpu_data_handle_t variable_handle[2];
			starpu_variable_data_register(&variable_handle[0], STARPU_MAIN_RAM, (uintptr_t)&v, sizeof(v));
			starpu_variable_data_register(&variable_handle[1], -1, (uintptr_t)NULL, sizeof(v));

			send_recv_and_check(rank, 1, variable_handle[0], 0x42, variable_handle[1], 0x1337, &error, check_variable);

			starpu_data_unregister(variable_handle[0]);
			starpu_data_unregister(variable_handle[1]);
		}

		{
			int vector[4] = {1, 2, 3, 4};
			starpu_data_handle_t vector_handle[2];

			starpu_vector_data_register(&vector_handle[0], STARPU_MAIN_RAM, (uintptr_t)vector, 4, sizeof(vector[0]));
			starpu_vector_data_register(&vector_handle[1], -1, (uintptr_t)NULL, 4, sizeof(vector[0]));

			send_recv_and_check(rank, 1, vector_handle[0], 0x43, vector_handle[1], 0x2337, &error, check_vector);

			starpu_data_unregister(vector_handle[0]);
			starpu_data_unregister(vector_handle[1]);
		}

		{
			char *matrix, n='a';
			int x, y;
			starpu_data_handle_t matrix_handle[2];

			matrix = (char*)malloc(nx*ny*nz*sizeof(char));
			assert(matrix);
			for(y=0 ; y<ny ; y++)
			{
				for(x=0 ; x<nx ; x++)
				{
					matrix[(y*nx)+x] = n++;
				}
			}

			starpu_matrix_data_register(&matrix_handle[0], STARPU_MAIN_RAM, (uintptr_t)matrix, nx, nx, ny, sizeof(char));
			starpu_matrix_data_register(&matrix_handle[1], -1, (uintptr_t)NULL, nx, nx, ny, sizeof(char));

			send_recv_and_check(rank, 1, matrix_handle[0], 0x75, matrix_handle[1], 0x8555, &error, check_matrix);

			starpu_data_unregister(matrix_handle[0]);
			starpu_data_unregister(matrix_handle[1]);
			free(matrix);
		}

		{
			float *block, n=1.0;
			int x, y, z;
			starpu_data_handle_t block_handle[2];

			block = (float*)malloc(nx*ny*nz*sizeof(float));
			assert(block);
			for(z=0 ; z<nz ; z++)
			{
				for(y=0 ; y<ny ; y++)
				{
					for(x=0 ; x<nx ; x++)
					{
						block[(z*nx*ny)+(y*nx)+x] = n++;
					}
				}
			}

			starpu_block_data_register(&block_handle[0], STARPU_MAIN_RAM, (uintptr_t)block, nx, nx*ny, nx, ny, nz, sizeof(float));
			starpu_block_data_register(&block_handle[1], -1, (uintptr_t)NULL, nx, nx*ny, nx, ny, nz, sizeof(float));

			send_recv_and_check(rank, 1, block_handle[0], 0x73, block_handle[1], 0x8337, &error, check_block);

			starpu_data_unregister(block_handle[0]);
			starpu_data_unregister(block_handle[1]);
			free(block);
		}
	}
	else if (rank == 1)
	{
		MPI_Status status;

		{
			starpu_data_handle_t variable_handle;
			starpu_variable_data_register(&variable_handle, -1, (uintptr_t)NULL, sizeof(float));
			send_recv_and_check(rank, 0, variable_handle, 0x42, NULL, 0x1337, NULL, NULL);
			starpu_data_unregister(variable_handle);
		}

		{
			starpu_data_handle_t vector_handle;
			starpu_vector_data_register(&vector_handle, -1, (uintptr_t)NULL, 4, sizeof(int));
			send_recv_and_check(rank, 0, vector_handle, 0x43, NULL, 0x2337, NULL, NULL);
			starpu_data_unregister(vector_handle);
		}

		{
			starpu_data_handle_t matrix_handle;
			starpu_matrix_data_register(&matrix_handle, -1, (uintptr_t)NULL, nx, nx, ny, sizeof(char));
			send_recv_and_check(rank, 0, matrix_handle, 0x75, NULL, 0x8555, NULL, NULL);
			starpu_data_unregister(matrix_handle);
		}

		{
			starpu_data_handle_t block_handle;
			starpu_block_data_register(&block_handle, -1, (uintptr_t)NULL, nx, nx*ny, nx, ny, nz, sizeof(float));
			send_recv_and_check(rank, 0, block_handle, 0x73, NULL, 0x8337, NULL, NULL);
			starpu_data_unregister(block_handle);
		}
	}

	starpu_mpi_shutdown();
	starpu_shutdown();

	MPI_Finalize();

	return rank == 0 ? error : 0;
}
