/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
	starpu_mpi_datatype_node_allocate_func_t allocate_datatype_node_func;
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

static void _make_lower_datatype(size_t block_size, size_t begin, size_t nx, size_t elemsize, int *block_lengths, MPI_Datatype *block_types, MPI_Aint *displacements, int *block_count, int nb_blocks)
{
	size_t end_line = begin+nx*elemsize;
	_STARPU_MPI_DEBUG(1200, "  processing lower layer from pos %zu to %zu\n", begin, end_line);
	do
	{
		size_t block_end = begin+block_size;
		if (block_end > end_line) block_end = end_line;
		if (*block_count > 0 && (begin == (size_t)block_lengths[*block_count - 1]+displacements[*block_count-1]) && (block_lengths[*block_count - 1] + block_end-begin < block_size))
		{
			// the new block is directly after the previous one and both do not exceed maximum size, merge them
			block_lengths[*block_count - 1] += (block_end-begin);
			_STARPU_MPI_DEBUG(1200, "    updating previous block %d with length %u from %zu to %zu with displacement %zu\n", *block_count-1, block_lengths[*block_count-1], displacements[*block_count-1], block_end, displacements[*block_count]);
		}
		else
		{
			block_lengths[*block_count] = (block_end-begin);
			block_types[*block_count] = MPI_BYTE;
			displacements[*block_count] = begin;
			_STARPU_MPI_DEBUG(1200, "    creating block %d with length %u from %zu to %zu with displacement %zu\n", *block_count, block_lengths[*block_count], begin, block_end, displacements[*block_count]);
			*block_count = *block_count + 1;
		}
		STARPU_ASSERT_MSG(*block_count < nb_blocks, "MPI Datatype creation failed");
		begin = block_end;
	} while(begin < end_line);
}

static void _make_recursive_datatype(size_t *layers, size_t *steps, int nb_dims, int current_dim, size_t elemsize, size_t block_size, size_t begin, int *block_lengths, MPI_Datatype *block_types, MPI_Aint *displacements, int *block_count, int nb_blocks)
{
	size_t i;
	for(i=0 ; i<layers[current_dim-1] ; i++)
	{
		if (current_dim-1 == 1)
			// this is the lowest dimension
			_make_lower_datatype(block_size, begin, layers[0], elemsize, block_lengths, block_types, displacements, block_count, nb_blocks);
		else
			_make_recursive_datatype(layers, steps, nb_dims, current_dim-1, elemsize, block_size, begin, block_lengths, block_types, displacements, block_count, nb_blocks);
		begin += steps[current_dim-1] * elemsize;
	}
}

static void _make_datatype(MPI_Datatype *datatype, size_t *layers, size_t *steps, int nb_dims, size_t elemsize)
{
#define BLOCK_SIZE (size_t)INT_MAX
#define NB_BLOCKS 10000
	int block_count=0;
	int block_lengths[NB_BLOCKS];
	MPI_Aint displacements[NB_BLOCKS];
	MPI_Datatype block_types[NB_BLOCKS];
	int ret;

	_STARPU_MPI_DEBUG(1200, "creating datatype for a data with %d dimensions with elements of size %zu and a block_size %zu\n", nb_dims, elemsize, BLOCK_SIZE);

	size_t begin = 0;
	if (nb_dims == 1)
		_make_lower_datatype(BLOCK_SIZE, begin, layers[0], elemsize, block_lengths, block_types, displacements, &block_count, NB_BLOCKS);
	else
		_make_recursive_datatype(layers, steps, nb_dims, nb_dims, elemsize, BLOCK_SIZE, begin, block_lengths, block_types, displacements, &block_count, NB_BLOCKS);

	ret = MPI_Type_create_struct(block_count, block_lengths, displacements, block_types, datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_create_struct failed");

	{
		MPI_Aint lb, extent;
		MPI_Type_get_extent(*datatype, &lb, &extent);
		_STARPU_MPI_DEBUG(1200, "using type with size %zu, lb %zu and %d block(s)\n", extent, lb, block_count);
		STARPU_ASSERT_MSG(extent-displacements[block_count-1]-block_lengths[block_count-1] == 0, "MPI Datatype creation failed. Your MPI implementation does not seem to manage large data sets. You should use a MPI which implements MPI_Type_vector_c");
	}

	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");
}

/*
 * 	Matrix
 */

static int handle_to_datatype_matrix(starpu_data_handle_t data_handle, unsigned node, MPI_Datatype *datatype)
{
	struct starpu_matrix_interface *matrix_interface = starpu_data_get_interface_on_node(data_handle, node);

	size_t nx = STARPU_MATRIX_GET_NX(matrix_interface);
	size_t ny = STARPU_MATRIX_GET_NY(matrix_interface);
	size_t ld = STARPU_MATRIX_GET_LD(matrix_interface);
	size_t elemsize = STARPU_MATRIX_GET_ELEMSIZE(matrix_interface);

#ifdef STARPU_HAVE_MPI_TYPE_VECTOR_C
	int ret;
	_STARPU_MPI_DEBUG(1200, "creating datatype for matrix using MPI_Type_vector_c\n");
	ret = MPI_Type_vector_c(ny, nx*elemsize, ld*elemsize, MPI_BYTE, datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_vector_c failed");
	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");
#else
	_STARPU_MPI_DEBUG(1200, "creating datatype for matrix using MPI_Type_create_struct\n");
	size_t layers[2] = {nx, ny};
	size_t steps[2] = {1, ld};
	_make_datatype(datatype, layers, steps, 2, elemsize);
#endif

	return 0;
}

/*
 * 	Block
 */

static int handle_to_datatype_block(starpu_data_handle_t data_handle, unsigned node, MPI_Datatype *datatype)
{
	struct starpu_block_interface *block_interface = starpu_data_get_interface_on_node(data_handle, node);

	size_t nx = STARPU_BLOCK_GET_NX(block_interface);
	size_t ny = STARPU_BLOCK_GET_NY(block_interface);
	size_t nz = STARPU_BLOCK_GET_NZ(block_interface);
	size_t ldy = STARPU_BLOCK_GET_LDY(block_interface);
	size_t ldz = STARPU_BLOCK_GET_LDZ(block_interface);
	size_t elemsize = STARPU_BLOCK_GET_ELEMSIZE(block_interface);

#ifdef STARPU_HAVE_MPI_TYPE_VECTOR_C
	int ret;
	_STARPU_MPI_DEBUG(1200, "creating datatype for block using MPI_Type_vector_c\n");
	MPI_Datatype datatype_2dlayer;
	ret = MPI_Type_vector_c(ny, nx*elemsize, ldy*elemsize, MPI_BYTE, &datatype_2dlayer);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_vector failed");

	ret = MPI_Type_create_hvector(nz, 1, ldz*elemsize, datatype_2dlayer, datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_hvector failed");

	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");

	ret = MPI_Type_free(&datatype_2dlayer);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_free failed");
#else
	size_t layers[3] = {nx, ny, nz};
	size_t steps[3] = {1, ldy, ldz};
	_make_datatype(datatype, layers, steps, 3, elemsize);
#endif

	return 0;
}

/*
 * 	Tensor
 */

static int handle_to_datatype_tensor(starpu_data_handle_t data_handle, unsigned node, MPI_Datatype *datatype)
{
	struct starpu_tensor_interface *tensor_interface = starpu_data_get_interface_on_node(data_handle, node);

	size_t nx = STARPU_TENSOR_GET_NX(tensor_interface);
	size_t ny = STARPU_TENSOR_GET_NY(tensor_interface);
	size_t nz = STARPU_TENSOR_GET_NZ(tensor_interface);
	size_t nt = STARPU_TENSOR_GET_NT(tensor_interface);
	size_t ldy = STARPU_TENSOR_GET_LDY(tensor_interface);
	size_t ldz = STARPU_TENSOR_GET_LDZ(tensor_interface);
	size_t ldt = STARPU_TENSOR_GET_LDT(tensor_interface);
	size_t elemsize = STARPU_TENSOR_GET_ELEMSIZE(tensor_interface);

	size_t layers[4] = {nx, ny, nz, nt};
	size_t steps[4] = {1, ldy, ldz, ldt};
	_make_datatype(datatype, layers, steps, 4, elemsize);

	return 0;
}

/*
 * 	Ndim
 */

static int handle_to_datatype_ndim(starpu_data_handle_t data_handle, unsigned node, MPI_Datatype *datatype)
{
	struct starpu_ndim_interface *ndim_interface = starpu_data_get_interface_on_node(data_handle, node);

	size_t *nn = STARPU_NDIM_GET_NN(ndim_interface);
	size_t *ldn = STARPU_NDIM_GET_LDN(ndim_interface);
	size_t ndim = STARPU_NDIM_GET_NDIM(ndim_interface);
	size_t elemsize = STARPU_NDIM_GET_ELEMSIZE(ndim_interface);

	_make_datatype(datatype, nn, ldn, ndim, elemsize);
	return 0;
}

/*
 * 	Vector
 */

static int handle_to_datatype_vector(starpu_data_handle_t data_handle, unsigned node, MPI_Datatype *datatype)
{
	struct starpu_vector_interface *vector_interface = starpu_data_get_interface_on_node(data_handle, node);

	size_t nx = STARPU_VECTOR_GET_NX(vector_interface);
	size_t elemsize = STARPU_VECTOR_GET_ELEMSIZE(vector_interface);

#ifdef STARPU_HAVE_MPI_TYPE_VECTOR_C
	int ret;
	_STARPU_MPI_DEBUG(1200, "creating datatype for vector using MPI_Type_vector_c\n");
	ret = MPI_Type_vector_c(1, nx*elemsize, 0, MPI_BYTE, datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_vector_c failed");
	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");
#else
	_STARPU_MPI_DEBUG(1200, "creating datatype for vector using MPI_Type_create_struct\n");
	size_t layers[1] = {nx};
	_make_datatype(datatype, layers, NULL, 1, elemsize);
#endif

	return 0;
}

/*
 * 	Variable
 */

static int handle_to_datatype_variable(starpu_data_handle_t data_handle, unsigned node, MPI_Datatype *datatype)
{
	struct starpu_variable_interface *variable_interface = starpu_data_get_interface_on_node(data_handle, node);

	int ret;

	size_t elemsize = STARPU_VARIABLE_GET_ELEMSIZE(variable_interface);

	ret = MPI_Type_contiguous(elemsize, MPI_BYTE, datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_contiguous failed");

	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");

	return 0;
}

/*
 * 	Void
 */

static int handle_to_datatype_void(starpu_data_handle_t data_handle, unsigned node, MPI_Datatype *datatype)
{
	int ret;
	(void)data_handle;
	(void)node;

	ret = MPI_Type_contiguous(0, MPI_BYTE, datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_contiguous failed");

	ret = MPI_Type_commit(datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_commit failed");

	return 0;
}

/*
 *	Generic
 */

static starpu_mpi_datatype_node_allocate_func_t handle_to_datatype_funcs[STARPU_MAX_INTERFACE_ID] =
{
//#define DYNAMIC_MATRICES
#ifndef DYNAMIC_MATRICES
	[STARPU_MATRIX_INTERFACE_ID]	= handle_to_datatype_matrix,
#endif
	[STARPU_BLOCK_INTERFACE_ID]	= handle_to_datatype_block,
	[STARPU_TENSOR_INTERFACE_ID]	= handle_to_datatype_tensor,
	[STARPU_NDIM_INTERFACE_ID]	= handle_to_datatype_ndim,
	[STARPU_VECTOR_INTERFACE_ID]	= handle_to_datatype_vector,
	[STARPU_CSR_INTERFACE_ID]	= NULL, /* Sent through pack/unpack operations */
	[STARPU_BCSR_INTERFACE_ID]	= NULL, /* Sent through pack/unpack operations */
	[STARPU_VARIABLE_INTERFACE_ID]	= handle_to_datatype_variable,
	[STARPU_VOID_INTERFACE_ID]	= handle_to_datatype_void,
	[STARPU_MULTIFORMAT_INTERFACE_ID] = NULL,
};

MPI_Datatype _starpu_mpi_datatype_get_user_defined_datatype(starpu_data_handle_t data_handle, unsigned node)
{
	enum starpu_data_interface_id id = starpu_data_get_interface_id(data_handle);
	if (id < STARPU_MAX_INTERFACE_ID) return 0;

	struct _starpu_mpi_datatype_funcs *table;
	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_datatype_funcs_table_mutex);
	HASH_FIND_INT(_starpu_mpi_datatype_funcs_table, &id, table);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_datatype_funcs_table_mutex);
	if (table && (table->allocate_datatype_node_func || table->allocate_datatype_func))
	{
		MPI_Datatype datatype;
		int ret;
		if (table->allocate_datatype_node_func)
			ret = table->allocate_datatype_node_func(data_handle, node, &datatype);
		else
			ret = table->allocate_datatype_func(data_handle, &datatype);
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
		starpu_mpi_datatype_node_allocate_func_t func = handle_to_datatype_funcs[id];
		if (func)
		{
			func(data_handle, req->node, &req->datatype);
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
			STARPU_ASSERT_MSG(table->allocate_datatype_node_func || table->allocate_datatype_func, "Handle To Datatype Function not defined for StarPU data interface %d", id);
			int ret;
			if (table->allocate_datatype_node_func)
				ret = table->allocate_datatype_node_func(data_handle, req->node, &req->datatype);
			else
				ret = table->allocate_datatype_func(data_handle, &req->datatype);
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
#ifdef STARPU_MPI_VERBOSE
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
	int ret = MPI_Type_free(datatype);
	STARPU_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Type_free failed");
}

static starpu_mpi_datatype_free_func_t handle_free_datatype_funcs[STARPU_MAX_INTERFACE_ID] =
{
#ifndef DYNAMIC_MATRICES
	[STARPU_MATRIX_INTERFACE_ID]	= _starpu_mpi_handle_free_simple_datatype,
#endif
	[STARPU_BLOCK_INTERFACE_ID]	= _starpu_mpi_handle_free_simple_datatype,
	[STARPU_TENSOR_INTERFACE_ID]	= _starpu_mpi_handle_free_simple_datatype,
	[STARPU_VECTOR_INTERFACE_ID]	= _starpu_mpi_handle_free_simple_datatype,
	[STARPU_NDIM_INTERFACE_ID]	= _starpu_mpi_handle_free_simple_datatype,
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

int _starpu_mpi_interface_datatype_register(enum starpu_data_interface_id id, starpu_mpi_datatype_node_allocate_func_t allocate_datatype_node_func, starpu_mpi_datatype_allocate_func_t allocate_datatype_func, starpu_mpi_datatype_free_func_t free_datatype_func)
{
	struct _starpu_mpi_datatype_funcs *table;

	STARPU_ASSERT_MSG(id >= STARPU_MAX_INTERFACE_ID, "Cannot redefine the MPI datatype for a predefined StarPU datatype");

	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_mpi_datatype_funcs_table_mutex);
	HASH_FIND_INT(_starpu_mpi_datatype_funcs_table, &id, table);
	if (table)
	{
		table->allocate_datatype_node_func = allocate_datatype_node_func;
		table->allocate_datatype_func = allocate_datatype_func;
		table->free_datatype_func = free_datatype_func;
	}
	else
	{
		_STARPU_MPI_MALLOC(table, sizeof(struct _starpu_mpi_datatype_funcs));
		table->id = id;
		table->allocate_datatype_node_func = allocate_datatype_node_func;
		table->allocate_datatype_func = allocate_datatype_func;
		table->free_datatype_func = free_datatype_func;
		HASH_ADD_INT(_starpu_mpi_datatype_funcs_table, id, table);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_mpi_datatype_funcs_table_mutex);
	return 0;
}

int starpu_mpi_interface_datatype_node_register(enum starpu_data_interface_id id, starpu_mpi_datatype_node_allocate_func_t allocate_datatype_node_func, starpu_mpi_datatype_free_func_t free_datatype_func)
{
	return _starpu_mpi_interface_datatype_register(id, allocate_datatype_node_func, NULL, free_datatype_func);
}

int starpu_mpi_interface_datatype_register(enum starpu_data_interface_id id, starpu_mpi_datatype_allocate_func_t allocate_datatype_func, starpu_mpi_datatype_free_func_t free_datatype_func)
{
	return _starpu_mpi_interface_datatype_register(id, NULL, allocate_datatype_func, free_datatype_func);
}

int starpu_mpi_datatype_node_register(starpu_data_handle_t handle, starpu_mpi_datatype_node_allocate_func_t allocate_datatype_node_func, starpu_mpi_datatype_free_func_t free_datatype_func)
{
	enum starpu_data_interface_id id = starpu_data_get_interface_id(handle);
	int ret;
	ret = starpu_mpi_interface_datatype_node_register(id, allocate_datatype_node_func, free_datatype_func);
	STARPU_ASSERT_MSG(handle->ops->handle_to_pointer || handle->ops->to_pointer, "The data interface must define the operation 'to_pointer'\n");
	return ret;
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
