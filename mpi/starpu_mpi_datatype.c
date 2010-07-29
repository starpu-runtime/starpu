/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu_mpi_datatype.h>

/*
 *	MPI_* functions usually requires both a pointer to the first element of
 *	a datatype and the datatype itself, so we need to provide both.
 */

typedef int (*handle_to_datatype_func)(starpu_data_handle, MPI_Datatype *);
typedef void *(*handle_to_ptr_func)(starpu_data_handle);

/*
 * 	Matrix
 */

static int handle_to_datatype_matrix(starpu_data_handle data_handle, MPI_Datatype *datatype)
{
	int ret;

	unsigned nx = starpu_matrix_get_nx(data_handle);
	unsigned ny = starpu_matrix_get_ny(data_handle);
	unsigned ld = starpu_matrix_get_local_ld(data_handle);
	size_t elemsize = starpu_matrix_get_elemsize(data_handle);

	ret = MPI_Type_vector(ny, nx*elemsize, ld*elemsize, MPI_BYTE, datatype);
	STARPU_ASSERT(ret == MPI_SUCCESS);

	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT(ret == MPI_SUCCESS);

	return 0;
}

static void *handle_to_ptr_matrix(starpu_data_handle data_handle)
{
	return (void *)starpu_matrix_get_local_ptr(data_handle);
}

/*
 * 	Block
 */

static int handle_to_datatype_block(starpu_data_handle data_handle, MPI_Datatype *datatype)
{
	int ret;

	unsigned nx = starpu_block_get_nx(data_handle);
	unsigned ny = starpu_block_get_ny(data_handle);
	unsigned nz = starpu_block_get_nz(data_handle);
	unsigned ldy = starpu_block_get_local_ldy(data_handle);
	unsigned ldz = starpu_block_get_local_ldz(data_handle);
	size_t elemsize = starpu_block_get_elemsize(data_handle);

	MPI_Datatype datatype_2dlayer;
	ret = MPI_Type_vector(ny, nx*elemsize, ldy*elemsize, MPI_BYTE, &datatype_2dlayer);
	STARPU_ASSERT(ret == MPI_SUCCESS);

	ret = MPI_Type_commit(&datatype_2dlayer);
	STARPU_ASSERT(ret == MPI_SUCCESS);

	ret = MPI_Type_hvector(nz, 1, ldz*elemsize, datatype_2dlayer, datatype);
	STARPU_ASSERT(ret == MPI_SUCCESS);

	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT(ret == MPI_SUCCESS);

	return 0;
}

static void *handle_to_ptr_block(starpu_data_handle data_handle)
{
	return (void *)starpu_block_get_local_ptr(data_handle);
}

/*
 * 	Vector
 */

static int handle_to_datatype_vector(starpu_data_handle data_handle, MPI_Datatype *datatype)
{
	int ret;

	unsigned nx = starpu_vector_get_nx(data_handle);
	size_t elemsize = starpu_vector_get_elemsize(data_handle);

	ret = MPI_Type_contiguous(nx*elemsize, MPI_BYTE, datatype);
	STARPU_ASSERT(ret == MPI_SUCCESS);

	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT(ret == MPI_SUCCESS);

	return 0;
}

static void *handle_to_ptr_vector(starpu_data_handle data_handle)
{
	return (void *)starpu_vector_get_local_ptr(data_handle);
}

/*
 *	Generic
 */

static handle_to_datatype_func handle_to_datatype_funcs[STARPU_NINTERFACES_ID] = {
	[STARPU_MATRIX_INTERFACE_ID]	= handle_to_datatype_matrix,
	[STARPU_BLOCK_INTERFACE_ID]	= handle_to_datatype_block,
	[STARPU_VECTOR_INTERFACE_ID]	= handle_to_datatype_vector,
	[STARPU_CSR_INTERFACE_ID]	= NULL,
	[STARPU_BCSR_INTERFACE_ID]	= NULL
};

static handle_to_ptr_func handle_to_ptr_funcs[STARPU_NINTERFACES_ID] = {
	[STARPU_MATRIX_INTERFACE_ID]	= handle_to_ptr_matrix,
	[STARPU_BLOCK_INTERFACE_ID]	= handle_to_ptr_block,
	[STARPU_VECTOR_INTERFACE_ID]	= handle_to_ptr_vector,
	[STARPU_CSR_INTERFACE_ID]	= NULL,
	[STARPU_BCSR_INTERFACE_ID]	= NULL
};

int starpu_mpi_handle_to_datatype(starpu_data_handle data_handle, MPI_Datatype *datatype)
{
	unsigned id = starpu_get_handle_interface_id(data_handle);

	handle_to_datatype_func func = handle_to_datatype_funcs[id];

	STARPU_ASSERT(func);

	return func(data_handle, datatype);
}

void *starpu_mpi_handle_to_ptr(starpu_data_handle data_handle)
{
	unsigned id = starpu_get_handle_interface_id(data_handle);

	handle_to_ptr_func func = handle_to_ptr_funcs[id];
	
	STARPU_ASSERT(func);

	return func(data_handle);
}
