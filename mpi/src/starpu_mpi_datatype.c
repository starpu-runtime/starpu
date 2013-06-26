/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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

#include <starpu_mpi_datatype.h>

typedef void (*handle_to_datatype_func)(starpu_data_handle_t, MPI_Datatype *);
typedef void (*handle_free_datatype_func)(MPI_Datatype *);

/*
 * 	Matrix
 */

static void handle_to_datatype_matrix(starpu_data_handle_t data_handle, MPI_Datatype *datatype)
{
	int ret;

	unsigned nx = starpu_matrix_get_nx(data_handle);
	unsigned ny = starpu_matrix_get_ny(data_handle);
	unsigned ld = starpu_matrix_get_local_ld(data_handle);
	size_t elemsize = starpu_matrix_get_elemsize(data_handle);

	ret = MPI_Type_vector(ny, nx*elemsize, ld*elemsize, MPI_BYTE, datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_vector failed");

	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");
}

/*
 * 	Block
 */

static void handle_to_datatype_block(starpu_data_handle_t data_handle, MPI_Datatype *datatype)
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
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_vector failed");

	ret = MPI_Type_commit(&datatype_2dlayer);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");

	ret = MPI_Type_hvector(nz, 1, ldz*elemsize, datatype_2dlayer, datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_hvector failed");

	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");
}

/*
 * 	Vector
 */

static void handle_to_datatype_vector(starpu_data_handle_t data_handle, MPI_Datatype *datatype)
{
	int ret;

	unsigned nx = starpu_vector_get_nx(data_handle);
	size_t elemsize = starpu_vector_get_elemsize(data_handle);

	ret = MPI_Type_contiguous(nx*elemsize, MPI_BYTE, datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_contiguous failed");

	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");
}

/*
 * 	Variable
 */

static void handle_to_datatype_variable(starpu_data_handle_t data_handle, MPI_Datatype *datatype)
{
	int ret;

	size_t elemsize = starpu_variable_get_elemsize(data_handle);

	ret = MPI_Type_contiguous(elemsize, MPI_BYTE, datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_contiguous failed");

	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");
}

/*
 *	Generic
 */

static handle_to_datatype_func handle_to_datatype_funcs[STARPU_MAX_INTERFACE_ID] =
{
	[STARPU_MATRIX_INTERFACE_ID]	= handle_to_datatype_matrix,
	[STARPU_BLOCK_INTERFACE_ID]	= handle_to_datatype_block,
	[STARPU_VECTOR_INTERFACE_ID]	= handle_to_datatype_vector,
	[STARPU_CSR_INTERFACE_ID]	= NULL,
	[STARPU_BCSR_INTERFACE_ID]	= NULL,
	[STARPU_VARIABLE_INTERFACE_ID]	= handle_to_datatype_variable,
	[STARPU_VOID_INTERFACE_ID]      = NULL,
	[STARPU_MULTIFORMAT_INTERFACE_ID] = NULL,
};

void _starpu_mpi_handle_allocate_datatype(starpu_data_handle_t data_handle, MPI_Datatype *datatype, int *user_datatype)
{
	enum starpu_data_interface_id id = starpu_data_get_interface_id(data_handle);

	if (id < STARPU_MAX_INTERFACE_ID)
	{
		handle_to_datatype_func func = handle_to_datatype_funcs[id];
		STARPU_ASSERT_MSG(func, "Handle To Datatype Function not defined for StarPU data interface %d", id);
		func(data_handle, datatype);
		*user_datatype = 0;
	}
	else
	{
		/* The datatype is not predefined by StarPU */
		*datatype = MPI_BYTE;
		*user_datatype = 1;
	}
}

static void _starpu_mpi_handle_free_simple_datatype(MPI_Datatype *datatype)
{
	MPI_Type_free(datatype);
}

static void _starpu_mpi_handle_free_complex_datatype(MPI_Datatype *datatype)
{
	int num_ints, num_adds, num_datatypes, combiner, i;
	int *array_of_ints;
	MPI_Aint *array_of_adds;
	MPI_Datatype *array_of_datatypes;

	MPI_Type_get_envelope(*datatype, &num_ints, &num_adds, &num_datatypes, &combiner);
	if (combiner != MPI_COMBINER_NAMED)
	{
		array_of_ints = (int *) malloc(num_ints * sizeof(int));
		array_of_adds = (MPI_Aint *) malloc(num_adds * sizeof(MPI_Aint));
		array_of_datatypes = (MPI_Datatype *) malloc(num_datatypes * sizeof(MPI_Datatype));
		MPI_Type_get_contents(*datatype, num_ints, num_adds, num_datatypes, array_of_ints, array_of_adds, array_of_datatypes);
		for(i=0 ; i<num_datatypes ; i++)
		{
			_starpu_mpi_handle_free_complex_datatype(&array_of_datatypes[i]);
		}
		MPI_Type_free(datatype);
		free(array_of_ints);
		free(array_of_adds);
		free(array_of_datatypes);
	}
}

static handle_free_datatype_func handle_free_datatype_funcs[STARPU_MAX_INTERFACE_ID] =
{
	[STARPU_MATRIX_INTERFACE_ID]	= _starpu_mpi_handle_free_simple_datatype,
	[STARPU_BLOCK_INTERFACE_ID]	= _starpu_mpi_handle_free_complex_datatype,
	[STARPU_VECTOR_INTERFACE_ID]	= _starpu_mpi_handle_free_simple_datatype,
	[STARPU_CSR_INTERFACE_ID]	= NULL,
	[STARPU_BCSR_INTERFACE_ID]	= NULL,
	[STARPU_VARIABLE_INTERFACE_ID]	= _starpu_mpi_handle_free_simple_datatype,
	[STARPU_VOID_INTERFACE_ID]      = NULL,
	[STARPU_MULTIFORMAT_INTERFACE_ID] = NULL,
};

void _starpu_mpi_handle_free_datatype(starpu_data_handle_t data_handle, MPI_Datatype *datatype)
{
	enum starpu_data_interface_id id = starpu_data_get_interface_id(data_handle);

	if (id < STARPU_MAX_INTERFACE_ID)
	{
		handle_free_datatype_func func = handle_free_datatype_funcs[id];
		STARPU_ASSERT_MSG(func, "Handle free datatype function not defined for StarPU data interface %d", id);
		func(datatype);
	}
	/* else the datatype is not predefined by StarPU */
}

char *_starpu_mpi_datatype(MPI_Datatype datatype)
{
     if (datatype == MPI_DATATYPE_NULL) return "MPI_DATATYPE_NULL";
     if (datatype == MPI_CHAR) return "MPI_CHAR";
     if (datatype == MPI_UNSIGNED_CHAR) return "MPI_UNSIGNED_CHAR";
     if (datatype == MPI_BYTE) return "MPI_BYTE";
     if (datatype == MPI_SHORT) return "MPI_SHORT";
     if (datatype == MPI_UNSIGNED_SHORT) return "MPI_UNSIGNED_SHORT";
     if (datatype == MPI_INT) return "MPI_INT";
     if (datatype == MPI_UNSIGNED) return "MPI_UNSIGNED";
     if (datatype == MPI_LONG) return "MPI_LONG";
     if (datatype == MPI_UNSIGNED_LONG) return "MPI_UNSIGNED_LONG";
     if (datatype == MPI_FLOAT) return "MPI_FLOAT";
     if (datatype == MPI_DOUBLE) return "MPI_DOUBLE";
     if (datatype == MPI_LONG_DOUBLE) return "MPI_LONG_DOUBLE";
     if (datatype == MPI_LONG_LONG) return "MPI_LONG_LONG";
     if (datatype == MPI_LONG_INT) return "MPI_LONG_INT";
     if (datatype == MPI_SHORT_INT) return "MPI_SHORT_INT";
     if (datatype == MPI_FLOAT_INT) return "MPI_FLOAT_INT";
     if (datatype == MPI_DOUBLE_INT) return "MPI_DOUBLE_INT";
     if (datatype == MPI_2INT) return "MPI_2INT";
     if (datatype == MPI_2DOUBLE_PRECISION) return "MPI_2DOUBLE_PRECISION";
     if (datatype == MPI_COMPLEX) return "MPI_COMPLEX";
     if (datatype == MPI_DOUBLE_COMPLEX) return "MPI_DOUBLE_COMPLEX";
     if (datatype == MPI_LOGICAL) return "MPI_LOGICAL";
     if (datatype == MPI_REAL) return "MPI_REAL";
     if (datatype == MPI_REAL4) return "MPI_REAL4";
     if (datatype == MPI_REAL8) return "MPI_REAL8";
     if (datatype == MPI_DOUBLE_PRECISION) return "MPI_DOUBLE_PRECISION";
     if (datatype == MPI_INTEGER) return "MPI_INTEGER";
     if (datatype == MPI_INTEGER4) return "MPI_INTEGER4";
     if (datatype == MPI_INTEGER8) return "MPI_INTEGER8";
     if (datatype == MPI_PACKED) return "MPI_PACKED";
     if (datatype == 0) return "Unknown datatype";
     return "User defined MPI Datatype";
}
