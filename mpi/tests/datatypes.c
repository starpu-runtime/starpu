/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

		assert(func);
		func(handle_s, handle_r, error);
	}
	else if (rank == 1)
	{
		ret = starpu_mpi_recv(handle_s, node, tag_s, MPI_COMM_WORLD, &status);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_recv");
		ret = starpu_mpi_send(handle_s, node, tag_r, MPI_COMM_WORLD);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send");
	}
}

/*
 * Void
 */
void check_void(starpu_data_handle_t handle_s, starpu_data_handle_t handle_r, int *error)
{
	(void)error;
	(void)handle_s;
	(void)handle_r;
	FPRINTF_MPI(stderr, "Success with void value\n");
}

void exchange_void(int rank, int *error)
{
	STARPU_SKIP_IF_VALGRIND;

	if (rank == 0)
	{
		starpu_data_handle_t void_handle[2];
		starpu_void_data_register(&void_handle[0]);
		starpu_void_data_register(&void_handle[1]);

		send_recv_and_check(rank, 1, void_handle[0], 0x42, void_handle[1], 0x1337, error, check_void);

		starpu_data_unregister(void_handle[0]);
		starpu_data_unregister(void_handle[1]);
	}
	else if (rank == 1)
	{
		starpu_data_handle_t void_handle;
		starpu_void_data_register(&void_handle);
		send_recv_and_check(rank, 0, void_handle, 0x42, NULL, 0x1337, NULL, NULL);
		starpu_data_unregister(void_handle);
	}
}

/*
 * Variable
 */
void check_variable(starpu_data_handle_t handle_s, starpu_data_handle_t handle_r, int *error)
{
	float *v_s, *v_r;

	STARPU_ASSERT(starpu_variable_get_elemsize(handle_s) == starpu_variable_get_elemsize(handle_r));

	v_s = (float *)starpu_variable_get_local_ptr(handle_s);
	v_r = (float *)starpu_variable_get_local_ptr(handle_r);

	if (*v_s == *v_r)
	{
		FPRINTF_MPI(stderr, "Success with variable value: %f == %f\n", *v_s, *v_r);
	}
	else
	{
		*error = 1;
		FPRINTF_MPI(stderr, "Error with variable value: %f != %f\n", *v_s, *v_r);
	}
}

void exchange_variable(int rank, int *error)
{
	if (rank == 0)
	{
		float v = 42.12;
		starpu_data_handle_t variable_handle[2];
		starpu_variable_data_register(&variable_handle[0], STARPU_MAIN_RAM, (uintptr_t)&v, sizeof(v));
		starpu_variable_data_register(&variable_handle[1], -1, (uintptr_t)NULL, sizeof(v));

		send_recv_and_check(rank, 1, variable_handle[0], 0x42, variable_handle[1], 0x1337, error, check_variable);

		starpu_data_unregister(variable_handle[0]);
		starpu_data_unregister(variable_handle[1]);
	}
	else if (rank == 1)
	{
		starpu_data_handle_t variable_handle;
		starpu_variable_data_register(&variable_handle, -1, (uintptr_t)NULL, sizeof(float));
		send_recv_and_check(rank, 0, variable_handle, 0x42, NULL, 0x1337, NULL, NULL);
		starpu_data_unregister(variable_handle);
	}
}

/*
 * Vector
 */
void check_vector(starpu_data_handle_t handle_s, starpu_data_handle_t handle_r, int *error)
{
	int i;
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
			FPRINTF_MPI(stderr, "Success with vector[%d] value: %d == %d\n", i, v_s[i], v_r[i]);
		}
		else
		{
			*error = 1;
			FPRINTF_MPI(stderr, "Error with vector[%d] value: %d != %d\n", i, v_s[i], v_r[i]);
		}
	}
}

void exchange_vector(int rank, int *error)
{
	if (rank == 0)
	{
		int vector[4] = {1, 2, 3, 4};
		starpu_data_handle_t vector_handle[2];

		starpu_vector_data_register(&vector_handle[0], STARPU_MAIN_RAM, (uintptr_t)vector, 4, sizeof(vector[0]));
		starpu_vector_data_register(&vector_handle[1], -1, (uintptr_t)NULL, 4, sizeof(vector[0]));

		send_recv_and_check(rank, 1, vector_handle[0], 0x43, vector_handle[1], 0x2337, error, check_vector);

		starpu_data_unregister(vector_handle[0]);
		starpu_data_unregister(vector_handle[1]);
	}
	else if (rank == 1)
	{
		starpu_data_handle_t vector_handle;
		starpu_vector_data_register(&vector_handle, -1, (uintptr_t)NULL, 4, sizeof(int));
		send_recv_and_check(rank, 0, vector_handle, 0x43, NULL, 0x2337, NULL, NULL);
		starpu_data_unregister(vector_handle);
	}
}

/*
 * Matrix
 */
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
	{
		for(x=0 ; x<nx ; x++)
		{
			int index=(y*ldy)+x;
			if (matrix_s[index] == matrix_r[index])
			{
				FPRINTF_MPI(stderr, "Success with matrix[%d,%d --> %d] value: %c == %c\n", x, y, index, matrix_s[index], matrix_r[index]);
			}
			else
			{
				*error = 1;
				FPRINTF_MPI(stderr, "Error with matrix[%d,%d --> %d] value: %c != %c\n", x, y, index, matrix_s[index], matrix_r[index]);
			}
		}
	}
}

void exchange_matrix(int rank, int *error)
{
	int nx=3;
	int ny=2;

	if (rank == 0)
	{
		char *matrix, n='a';
		int x, y;
		starpu_data_handle_t matrix_handle[2];

		matrix = (char*)malloc(nx*ny*sizeof(char));
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

		send_recv_and_check(rank, 1, matrix_handle[0], 0x75, matrix_handle[1], 0x8555, error, check_matrix);

		starpu_data_unregister(matrix_handle[0]);
		starpu_data_unregister(matrix_handle[1]);
		free(matrix);
	}
	else if (rank == 1)
	{
		starpu_data_handle_t matrix_handle;
		starpu_matrix_data_register(&matrix_handle, -1, (uintptr_t)NULL, nx, nx, ny, sizeof(char));
		send_recv_and_check(rank, 0, matrix_handle, 0x75, NULL, 0x8555, NULL, NULL);
		starpu_data_unregister(matrix_handle);
	}
}

/*
 * Block
 */
void check_block(starpu_data_handle_t handle_s, starpu_data_handle_t handle_r, int *error)
{
	STARPU_ASSERT(starpu_block_get_elemsize(handle_s) == starpu_block_get_elemsize(handle_r));
	STARPU_ASSERT(starpu_block_get_nx(handle_s) == starpu_block_get_nx(handle_r));
	STARPU_ASSERT(starpu_block_get_ny(handle_s) == starpu_block_get_ny(handle_r));
	STARPU_ASSERT(starpu_block_get_nz(handle_s) == starpu_block_get_nz(handle_r));
	STARPU_ASSERT(starpu_block_get_local_ldy(handle_s) == starpu_block_get_local_ldy(handle_r));
	STARPU_ASSERT(starpu_block_get_local_ldz(handle_s) == starpu_block_get_local_ldz(handle_r));

	starpu_data_acquire(handle_s, STARPU_R);
	starpu_data_acquire(handle_r, STARPU_R);

	float *block_s = (float *)starpu_block_get_local_ptr(handle_s);
	float *block_r = (float *)starpu_block_get_local_ptr(handle_r);

	int nx = starpu_block_get_nx(handle_s);
	int ny = starpu_block_get_ny(handle_s);
	int nz = starpu_block_get_nz(handle_s);

	int ldy = starpu_block_get_local_ldy(handle_s);
	int ldz = starpu_block_get_local_ldz(handle_s);

	int x, y, z;

	for(z=0 ; z<nz ; z++)
	{
		for(y=0 ; y<ny ; y++)
			for(x=0 ; x<nx ; x++)
			{
				int index=(z*ldz)+(y*ldy)+x;
				if (block_s[index] == block_r[index])
				{
					FPRINTF_MPI(stderr, "Success with block[%d,%d,%d --> %d] value: %f == %f\n", x, y, z, index, block_s[index], block_r[index]);
				}
				else
				{
					*error = 1;
					FPRINTF_MPI(stderr, "Error with block[%d,%d,%d --> %d] value: %f != %f\n", x, y, z, index, block_s[index], block_r[index]);
				}
			}
	}

	starpu_data_release(handle_s);
	starpu_data_release(handle_r);
}

void exchange_block(int rank, int *error)
{
	int nx=3;
	int ny=2;
	int nz=4;

	if (rank == 0)
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

		send_recv_and_check(rank, 1, block_handle[0], 0x73, block_handle[1], 0x8337, error, check_block);

		starpu_data_unregister(block_handle[0]);
		starpu_data_unregister(block_handle[1]);
		free(block);
	}
	else if (rank == 1)
	{
		starpu_data_handle_t block_handle;
		starpu_block_data_register(&block_handle, -1, (uintptr_t)NULL, nx, nx*ny, nx, ny, nz, sizeof(float));
		send_recv_and_check(rank, 0, block_handle, 0x73, NULL, 0x8337, NULL, NULL);
		starpu_data_unregister(block_handle);
	}
}

/*
 * BCSR
 */
void check_bcsr(starpu_data_handle_t handle_s, starpu_data_handle_t handle_r, int *error)
{
	STARPU_ASSERT(starpu_bcsr_get_elemsize(handle_s) == starpu_bcsr_get_elemsize(handle_r));
	STARPU_ASSERT(starpu_bcsr_get_nnz(handle_s) == starpu_bcsr_get_nnz(handle_r));
	STARPU_ASSERT(starpu_bcsr_get_nrow(handle_s) == starpu_bcsr_get_nrow(handle_r));
	STARPU_ASSERT(starpu_bcsr_get_firstentry(handle_s) == starpu_bcsr_get_firstentry(handle_r));
	STARPU_ASSERT(starpu_bcsr_get_r(handle_s) == starpu_bcsr_get_r(handle_r));
	STARPU_ASSERT(starpu_bcsr_get_c(handle_s) == starpu_bcsr_get_c(handle_r));

	starpu_data_acquire(handle_s, STARPU_R);
	starpu_data_acquire(handle_r, STARPU_R);

	uint32_t *colind_s = starpu_bcsr_get_local_colind(handle_s);
	uint32_t *colind_r = starpu_bcsr_get_local_colind(handle_r);
	uint32_t *rowptr_s = starpu_bcsr_get_local_rowptr(handle_s);
	uint32_t *rowptr_r = starpu_bcsr_get_local_rowptr(handle_r);

	int *bcsr_s = (int *)starpu_bcsr_get_local_nzval(handle_s);
	int *bcsr_r = (int *)starpu_bcsr_get_local_nzval(handle_r);

	int r = starpu_bcsr_get_r(handle_s);
	int c = starpu_bcsr_get_c(handle_s);
	int nnz = starpu_bcsr_get_nnz(handle_s);
	int nrows = starpu_bcsr_get_nrow(handle_s);

	int x;

	for(x=0 ; x<nnz ; x++)
	{
		if (colind_s[x] == colind_r[x])
		{
			FPRINTF_MPI(stderr, "Success with colind[%d] value: %u == %u\n", x, colind_s[x], colind_r[x]);
		}
		else
		{
			*error = 1;
			FPRINTF_MPI(stderr, "Error with colind[%d] value: %u != %u\n", x, colind_s[x], colind_r[x]);
		}
	}

	for(x=0 ; x<nrows+1 ; x++)
	{
		if (rowptr_s[x] == rowptr_r[x])
		{
			FPRINTF_MPI(stderr, "Success with rowptr[%d] value: %u == %u\n", x, rowptr_s[x], rowptr_r[x]);
		}
		else
		{
			*error = 1;
			FPRINTF_MPI(stderr, "Error with rowptr[%d] value: %u != %u\n", x, rowptr_s[x], rowptr_r[x]);
		}
	}

	for(x=0 ; x<r*c*nnz ; x++)
	{
		if (bcsr_s[x] == bcsr_r[x])
		{
			FPRINTF_MPI(stderr, "Success with bcsr[%d] value: %d == %d\n", x, bcsr_s[x], bcsr_r[x]);
		}
		else
		{
			*error = 1;
			FPRINTF_MPI(stderr, "Error with bcsr[%d] value: %d != %d\n", x, bcsr_s[x], bcsr_r[x]);
		}
	}

	starpu_data_release(handle_s);
	starpu_data_release(handle_r);
}

void exchange_bcsr(int rank, int *error)
{
	/*
	 * We use the following matrix:
	 *
	 *   +----------------+
	 *   |  0   1   0   0 |
	 *   |  2   3   0   0 |
	 *   |  4   5   8   9 |
	 *   |  6   7  10  11 |
	 *   +----------------+
	 *
	 * nzval  = [0, 1, 2, 3] ++ [4, 5, 6, 7] ++ [8, 9, 10, 11]
	 * colind = [0, 0, 1]
	 * rowptr = [0, 1, 3]
	 * r = c = 2
	 */

	/* Size of the blocks */
#define BCSR_R 2
#define BCSR_C 2
#define BCSR_NROWS 2
#define BCSR_NNZ_BLOCKS 3     /* out of 4 */
#define BCSR_NZVAL_SIZE (BCSR_R*BCSR_C*BCSR_NNZ_BLOCKS)

	if (rank == 0)
	{
		starpu_data_handle_t bcsr_handle[2];
		uint32_t colind[BCSR_NNZ_BLOCKS] = {0, 0, 1};
		uint32_t rowptr[BCSR_NROWS+1] = {0, 1, BCSR_NNZ_BLOCKS};
		int nzval[BCSR_NZVAL_SIZE]  =
		{
			0, 1, 2, 3,    /* First block  */
			4, 5, 6, 7,    /* Second block */
			8, 9, 10, 11   /* Third block  */
		};

		starpu_bcsr_data_register(&bcsr_handle[0], STARPU_MAIN_RAM, BCSR_NNZ_BLOCKS, BCSR_NROWS, (uintptr_t) nzval, colind, rowptr, 0, BCSR_R, BCSR_C, sizeof(nzval[0]));
		starpu_bcsr_data_register(&bcsr_handle[1], -1, BCSR_NNZ_BLOCKS, BCSR_NROWS, (uintptr_t) NULL, (uint32_t *) NULL, (uint32_t *) NULL, 0, BCSR_R, BCSR_C, sizeof(nzval[0]));

		send_recv_and_check(rank, 1, bcsr_handle[0], 0x73, bcsr_handle[1], 0x8337, error, check_bcsr);

		starpu_data_unregister(bcsr_handle[0]);
		starpu_data_unregister(bcsr_handle[1]);
	}
	else if (rank == 1)
	{
		starpu_data_handle_t bcsr_handle;
		starpu_bcsr_data_register(&bcsr_handle, -1, BCSR_NNZ_BLOCKS, BCSR_NROWS, (uintptr_t) NULL, (uint32_t *) NULL, (uint32_t *) NULL, 0, BCSR_R, BCSR_C, sizeof(int));
		send_recv_and_check(rank, 0, bcsr_handle, 0x73, NULL, 0x8337, NULL, NULL);
		starpu_data_unregister(bcsr_handle);
	}
}

/*
 * CSR
 */
void check_csr(starpu_data_handle_t handle_s, starpu_data_handle_t handle_r, int *error)
{
	STARPU_ASSERT(starpu_csr_get_elemsize(handle_s) == starpu_csr_get_elemsize(handle_r));
	STARPU_ASSERT(starpu_csr_get_nnz(handle_s) == starpu_csr_get_nnz(handle_r));
	STARPU_ASSERT(starpu_csr_get_nrow(handle_s) == starpu_csr_get_nrow(handle_r));
	STARPU_ASSERT(starpu_csr_get_firstentry(handle_s) == starpu_csr_get_firstentry(handle_r));

	starpu_data_acquire(handle_s, STARPU_R);
	starpu_data_acquire(handle_r, STARPU_R);

	uint32_t *colind_s = starpu_csr_get_local_colind(handle_s);
	uint32_t *colind_r = starpu_csr_get_local_colind(handle_r);
	uint32_t *rowptr_s = starpu_csr_get_local_rowptr(handle_s);
	uint32_t *rowptr_r = starpu_csr_get_local_rowptr(handle_r);

	int *csr_s = (int *)starpu_csr_get_local_nzval(handle_s);
	int *csr_r = (int *)starpu_csr_get_local_nzval(handle_r);

	int nnz = starpu_csr_get_nnz(handle_s);
	int nrows = starpu_csr_get_nrow(handle_s);

	int x;

	for(x=0 ; x<nnz ; x++)
	{
		if (colind_s[x] == colind_r[x])
		{
			FPRINTF_MPI(stderr, "Success with colind[%d] value: %u == %u\n", x, colind_s[x], colind_r[x]);
		}
		else
		{
			*error = 1;
			FPRINTF_MPI(stderr, "Error with colind[%d] value: %u != %u\n", x, colind_s[x], colind_r[x]);
		}
	}

	for(x=0 ; x<nrows+1 ; x++)
	{
		if (rowptr_s[x] == rowptr_r[x])
		{
			FPRINTF_MPI(stderr, "Success with rowptr[%d] value: %u == %u\n", x, rowptr_s[x], rowptr_r[x]);
		}
		else
		{
			*error = 1;
			FPRINTF_MPI(stderr, "Error with rowptr[%d] value: %u != %u\n", x, rowptr_s[x], rowptr_r[x]);
		}
	}

	for(x=0 ; x<nnz ; x++)
	{
		if (csr_s[x] == csr_r[x])
		{
			FPRINTF_MPI(stderr, "Success with csr[%d] value: %d == %d\n", x, csr_s[x], csr_r[x]);
		}
		else
		{
			*error = 1;
			FPRINTF_MPI(stderr, "Error with csr[%d] value: %d != %d\n", x, csr_s[x], csr_r[x]);
		}
	}

	starpu_data_release(handle_s);
	starpu_data_release(handle_r);
}

void exchange_csr(int rank, int *error)
{
	// the values are completely wrong, we just want to test that the communication is done correctly
#define CSR_NROWS 2
#define CSR_NNZ   5

	if (rank == 0)
	{
		starpu_data_handle_t csr_handle[2];
		uint32_t colind[CSR_NNZ] = {0, 1, 2, 3, 4};
		uint32_t rowptr[CSR_NROWS+1] = {0, 1, CSR_NNZ};
		int nzval[CSR_NNZ] = { 11, 22, 33, 44, 55 };

		starpu_csr_data_register(&csr_handle[0], STARPU_MAIN_RAM, CSR_NNZ, CSR_NROWS, (uintptr_t) nzval, colind, rowptr, 0, sizeof(nzval[0]));
		starpu_csr_data_register(&csr_handle[1], -1, CSR_NNZ, CSR_NROWS, (uintptr_t) NULL, (uint32_t *) NULL, (uint32_t *) NULL, 0, sizeof(nzval[0]));

		send_recv_and_check(rank, 1, csr_handle[0], 0x84, csr_handle[1], 0x8765, error, check_csr);

		starpu_data_unregister(csr_handle[0]);
		starpu_data_unregister(csr_handle[1]);
	}
	else if (rank == 1)
	{
		starpu_data_handle_t csr_handle;
		starpu_csr_data_register(&csr_handle, -1, CSR_NNZ, CSR_NROWS, (uintptr_t) NULL, (uint32_t *) NULL, (uint32_t *) NULL, 0, sizeof(int));
		send_recv_and_check(rank, 0, csr_handle, 0x84, NULL, 0x8765, NULL, NULL);
		starpu_data_unregister(csr_handle);
	}
}

int main(int argc, char **argv)
{
	int ret, rank, size;
	int error=0;
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

	if (size < 2)
	{
		if (rank == 0)
			FPRINTF(stderr, "We need at least 2 processes.\n");

		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

	exchange_void(rank, &error);
	exchange_variable(rank, &error);
	exchange_vector(rank, &error);
	exchange_matrix(rank, &error);
	exchange_block(rank, &error);
	exchange_bcsr(rank, &error);
	exchange_csr(rank, &error);

	starpu_mpi_shutdown();

	if (!mpi_init)
		MPI_Finalize();

	return rank == 0 ? error : 0;
}
