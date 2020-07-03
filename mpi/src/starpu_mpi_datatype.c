/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <common/uthash.h>
#include <datawizard/coherency.h>

struct _starpu_mpi_datatype_funcs
{
	enum starpu_data_interface_id id;
	starpu_mpi_datatype_allocate_func_t allocate_datatype_func;
	starpu_mpi_datatype_free_func_t free_datatype_func;
	UT_hash_handle hh;
};

/* We want to allow applications calling starpu_mpi_interface_datatype_register/unregister as constructor/destructor */
static starpu_pthread_mutex_t _starpu_mpi_datatype_funcs_table_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static struct _starpu_mpi_datatype_funcs *_starpu_mpi_datatype_funcs_table = NULL;

void _starpu_mpi_datatype_init(void)
{
}

void _starpu_mpi_datatype_shutdown(void)
{
}

/*
 * 	Matrix
 */

static int handle_to_datatype_matrix(starpu_data_handle_t data_handle, MPI_Datatype *datatype)
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

	return 0;
}

/*
 * 	Block
 */

static int handle_to_datatype_block(starpu_data_handle_t data_handle, MPI_Datatype *datatype)
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

	ret = MPI_Type_create_hvector(nz, 1, ldz*elemsize, datatype_2dlayer, datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_hvector failed");

	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");

	return 0;
}

/*
 * 	Vector
 */

static int handle_to_datatype_vector(starpu_data_handle_t data_handle, MPI_Datatype *datatype)
{
	int ret;

	unsigned nx = starpu_vector_get_nx(data_handle);
	size_t elemsize = starpu_vector_get_elemsize(data_handle);

	ret = MPI_Type_contiguous(nx*elemsize, MPI_BYTE, datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_contiguous failed");

	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");

	return 0;
}

/*
 * 	Variable
 */

static int handle_to_datatype_variable(starpu_data_handle_t data_handle, MPI_Datatype *datatype)
{
	int ret;

	size_t elemsize = starpu_variable_get_elemsize(data_handle);

	ret = MPI_Type_contiguous(elemsize, MPI_BYTE, datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_contiguous failed");

	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");

	return 0;
}

/*
 * 	Void
 */

static int handle_to_datatype_void(starpu_data_handle_t data_handle, MPI_Datatype *datatype)
{
	int ret;
	(void)data_handle;

	ret = MPI_Type_contiguous(0, MPI_BYTE, datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_contiguous failed");

	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");

	return 0;
}

/*
 *	Generic
 */

static starpu_mpi_datatype_allocate_func_t handle_to_datatype_funcs[STARPU_MAX_INTERFACE_ID] =
{
	[STARPU_MATRIX_INTERFACE_ID]	= handle_to_datatype_matrix,
	[STARPU_BLOCK_INTERFACE_ID]	= handle_to_datatype_block,
	[STARPU_VECTOR_INTERFACE_ID]	= handle_to_datatype_vector,
	[STARPU_CSR_INTERFACE_ID]	= NULL, /* Sent through pack/unpack operations */
	[STARPU_BCSR_INTERFACE_ID]	= NULL, /* Sent through pack/unpack operations */
	[STARPU_VARIABLE_INTERFACE_ID]	= handle_to_datatype_variable,
	[STARPU_VOID_INTERFACE_ID]	= handle_to_datatype_void,
	[STARPU_MULTIFORMAT_INTERFACE_ID] = NULL,
};

MPI_Datatype _starpu_mpi_datatype_get_user_defined_datatype(starpu_data_handle_t data_handle)
{
	enum starpu_data_interface_id id = starpu_data_get_interface_id(data_handle);
	if (id < STARPU_MAX_INTERFACE_ID) return 0;

	struct _starpu_mpi_datatype_funcs *table;
	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_datatype_funcs_table_mutex);
	HASH_FIND_INT(_starpu_mpi_datatype_funcs_table, &id, table);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_datatype_funcs_table_mutex);
	if (table && table->allocate_datatype_func)
	{
		MPI_Datatype datatype;
		int ret = table->allocate_datatype_func(data_handle, &datatype);
		if (ret == 0)
			return datatype;
		else
			return 0;
	}
	return 0;
}

void _starpu_mpi_datatype_allocate(starpu_data_handle_t data_handle, struct _starpu_mpi_req *req)
{
	enum starpu_data_interface_id id = starpu_data_get_interface_id(data_handle);

	if (id < STARPU_MAX_INTERFACE_ID)
	{
		starpu_mpi_datatype_allocate_func_t func = handle_to_datatype_funcs[id];
		if (func)
		{
			func(data_handle, &req->datatype);
			req->registered_datatype = 1;
		}
		else
		{
			/* The datatype is predefined by StarPU but it will be sent as a memory area */
			req->datatype = MPI_BYTE;
			req->registered_datatype = 0;
		}
	}
	else
	{
		struct _starpu_mpi_datatype_funcs *table;
		STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_datatype_funcs_table_mutex);
		HASH_FIND_INT(_starpu_mpi_datatype_funcs_table, &id, table);
		STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_datatype_funcs_table_mutex);
		if (table)
		{
			STARPU_ASSERT_MSG(table->allocate_datatype_func, "Handle To Datatype Function not defined for StarPU data interface %d", id);
			int ret = table->allocate_datatype_func(data_handle, &req->datatype);
			if (ret == 0)
				req->registered_datatype = 1;
			else
			{
				/* Couldn't register, probably complex data which needs packing. */
				req->datatype = MPI_BYTE;
				req->registered_datatype = 0;
			}
		}
		else
		{
			/* The datatype is not predefined by StarPU */
			req->datatype = MPI_BYTE;
			req->registered_datatype = 0;
		}
	}
#ifdef STARPU_VERBOSE
	{
		char datatype_name[MPI_MAX_OBJECT_NAME];
		int datatype_name_len;
		MPI_Type_get_name(req->datatype, datatype_name, &datatype_name_len);
		if (datatype_name_len == 0)
			req->datatype_name = strdup("User defined datatype");
		else
			req->datatype_name = strdup(datatype_name);
	}
#endif
}

static void _starpu_mpi_handle_free_simple_datatype(MPI_Datatype *datatype)
{
	MPI_Type_free(datatype);
}

static void _starpu_mpi_handle_free_complex_datatype(MPI_Datatype *datatype)
{
	int num_ints, num_adds, num_datatypes, combiner;

	MPI_Type_get_envelope(*datatype, &num_ints, &num_adds, &num_datatypes, &combiner);
	if (combiner != MPI_COMBINER_NAMED)
	{
		int *array_of_ints;
		MPI_Aint *array_of_adds;
		MPI_Datatype *array_of_datatypes;
		int i;

		_STARPU_MPI_MALLOC(array_of_ints, num_ints * sizeof(int));
		_STARPU_MPI_MALLOC(array_of_adds, num_adds * sizeof(MPI_Aint));
		_STARPU_MPI_MALLOC(array_of_datatypes, num_datatypes * sizeof(MPI_Datatype));

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

static starpu_mpi_datatype_free_func_t handle_free_datatype_funcs[STARPU_MAX_INTERFACE_ID] =
{
	[STARPU_MATRIX_INTERFACE_ID]	= _starpu_mpi_handle_free_simple_datatype,
	[STARPU_BLOCK_INTERFACE_ID]	= _starpu_mpi_handle_free_complex_datatype,
	[STARPU_VECTOR_INTERFACE_ID]	= _starpu_mpi_handle_free_simple_datatype,
	[STARPU_CSR_INTERFACE_ID]	= NULL,  /* Sent through pack/unpack operations */
	[STARPU_BCSR_INTERFACE_ID]	= NULL,  /* Sent through pack/unpack operations */
	[STARPU_VARIABLE_INTERFACE_ID]	= _starpu_mpi_handle_free_simple_datatype,
	[STARPU_VOID_INTERFACE_ID]      = _starpu_mpi_handle_free_simple_datatype,
	[STARPU_MULTIFORMAT_INTERFACE_ID] = NULL,
};

void _starpu_mpi_datatype_free(starpu_data_handle_t data_handle, MPI_Datatype *datatype)
{
	enum starpu_data_interface_id id = starpu_data_get_interface_id(data_handle);

	if (id < STARPU_MAX_INTERFACE_ID)
	{
		starpu_mpi_datatype_free_func_t func = handle_free_datatype_funcs[id];
		if (func)
			func(datatype);
	}
	else
	{
		struct _starpu_mpi_datatype_funcs *table;
		STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_datatype_funcs_table_mutex);
		HASH_FIND_INT(_starpu_mpi_datatype_funcs_table, &id, table);
		STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_datatype_funcs_table_mutex);
		if (table)
		{
			STARPU_ASSERT_MSG(table->free_datatype_func, "Free Datatype Function not defined for StarPU data interface %d", id);
			if (*datatype != MPI_BYTE)
				table->free_datatype_func(datatype);
		}

	}
	/* else the datatype is not predefined by StarPU */
}

int starpu_mpi_interface_datatype_register(enum starpu_data_interface_id id, starpu_mpi_datatype_allocate_func_t allocate_datatype_func, starpu_mpi_datatype_free_func_t free_datatype_func)
{
	struct _starpu_mpi_datatype_funcs *table;

	STARPU_ASSERT_MSG(id >= STARPU_MAX_INTERFACE_ID, "Cannot redefine the MPI datatype for a predefined StarPU datatype");

	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_datatype_funcs_table_mutex);
	HASH_FIND_INT(_starpu_mpi_datatype_funcs_table, &id, table);
	if (table)
	{
		table->allocate_datatype_func = allocate_datatype_func;
		table->free_datatype_func = free_datatype_func;
	}
	else
	{
		_STARPU_MPI_MALLOC(table, sizeof(struct _starpu_mpi_datatype_funcs));
		table->id = id;
		table->allocate_datatype_func = allocate_datatype_func;
		table->free_datatype_func = free_datatype_func;
		HASH_ADD_INT(_starpu_mpi_datatype_funcs_table, id, table);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_datatype_funcs_table_mutex);
	return 0;
}

int starpu_mpi_datatype_register(starpu_data_handle_t handle, starpu_mpi_datatype_allocate_func_t allocate_datatype_func, starpu_mpi_datatype_free_func_t free_datatype_func)
{
	enum starpu_data_interface_id id = starpu_data_get_interface_id(handle);
	int ret;
	ret = starpu_mpi_interface_datatype_register(id, allocate_datatype_func, free_datatype_func);
	STARPU_ASSERT_MSG(handle->ops->handle_to_pointer || handle->ops->to_pointer, "The data interface must define the operation 'to_pointer'\n");
	return ret;
}

int starpu_mpi_interface_datatype_unregister(enum starpu_data_interface_id id)
{
	struct _starpu_mpi_datatype_funcs *table;

	STARPU_ASSERT_MSG(id >= STARPU_MAX_INTERFACE_ID, "Cannot redefine the MPI datatype for a predefined StarPU datatype");

	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_datatype_funcs_table_mutex);
	HASH_FIND_INT(_starpu_mpi_datatype_funcs_table, &id, table);
	if (table)
	{
		HASH_DEL(_starpu_mpi_datatype_funcs_table, table);
		free(table);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_datatype_funcs_table_mutex);
	return 0;
}

int starpu_mpi_datatype_unregister(starpu_data_handle_t handle)
{
	enum starpu_data_interface_id id = starpu_data_get_interface_id(handle);
	return starpu_mpi_interface_datatype_unregister(id);
}
