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
#include <common/config.h>
#include <common/utils.h>
#include <datawizard/filters.h>

static void _interface_assignment_ndim_to_tensor(void *ndim_interface, void *child_interface);
static void _interface_assignment_ndim_to_block(void *ndim_interface, void *child_interface);
static void _interface_assignment_ndim_to_matrix(void *ndim_interface, void *child_interface);
static void _interface_assignment_ndim_to_vector(void *ndim_interface, void *child_interface);
static void _interface_assignment_ndim_to_variable(void *ndim_interface, void *child_interface);

static void _interface_deallocate(void * ndim_interface);

static void _starpu_ndim_filter_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				      unsigned id, unsigned nparts, uintptr_t shadow_size)
{
	struct starpu_ndim_interface *ndim_father = (struct starpu_ndim_interface *) father_interface;
	struct starpu_ndim_interface *ndim_child = (struct starpu_ndim_interface *) child_interface;

	STARPU_ASSERT_MSG(ndim_father->id == STARPU_NDIM_INTERFACE_ID, "%s can only be applied on a ndim array data", __func__);

	size_t ndim = ndim_father->ndim;
	STARPU_ASSERT_MSG(ndim > 0, "ndim %u must be greater than 0!\n", (unsigned) ndim);

	unsigned dim = 0;
	if (ndim > 1)
		dim = f->filter_arg;

	STARPU_ASSERT_MSG(dim < ndim, "dim %u must be less than %u!\n", dim, (unsigned) ndim);

	uint32_t father_nn = 0;
	uint32_t ni[ndim];
	unsigned i;
	for (i=0; i<ndim; i++)
	{
		if(i==dim)
		{
			ni[i] = ndim_father->nn[i] - 2 * shadow_size;
			father_nn = ni[i];
		}
		else
		{
			ni[i] = ndim_father->nn[i];
		}
	}

	STARPU_ASSERT_MSG(nparts <= father_nn, "cannot split %u elements in %u parts", father_nn, nparts);

	unsigned blocksize = ndim_father->ldn[dim];
	size_t elemsize = ndim_father->elemsize;
	uint32_t child_nn;
	size_t offset;

	starpu_filter_nparts_compute_chunk_size_and_offset(father_nn, nparts, elemsize, id, blocksize, &child_nn, &offset);
	child_nn += 2 * shadow_size;
	ndim_child->id = ndim_father->id;

	_STARPU_MALLOC(ndim_child->nn, ndim*sizeof(uint32_t));
	for (i=0; i<ndim; i++)
	{
		if (i!=dim)
		{
			ndim_child->nn[i] = ni[i];
		}
		else
		{
			ndim_child->nn[i] = child_nn;
		}
	}

	_STARPU_MALLOC(ndim_child->ldn, ndim*sizeof(uint32_t));
	ndim_child->ndim = ndim;
	ndim_child->elemsize = elemsize;
	ndim_child->allocsize = elemsize;

	if (ndim_father->dev_handle)
	{
		if (ndim_father->ptr)
			ndim_child->ptr = ndim_father->ptr + offset;
		for (i=0; i<ndim; i++)
		{
			ndim_child->ldn[i] = ndim_father->ldn[i];
		}

		if (ndim >= 1)
			ndim_child->allocsize *= ndim_child->ldn[ndim-1] * ndim_child->nn[ndim-1];

		ndim_child->dev_handle = ndim_father->dev_handle;
		ndim_child->offset = ndim_father->offset + offset;
	}
	else
	{
		for (i=0; i<ndim; i++)
			ndim_child->allocsize *= ndim_child->nn[i];
	}
}

void starpu_ndim_filter_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
			      unsigned id, unsigned nparts)
{
	_starpu_ndim_filter_block(father_interface, child_interface, f, id, nparts, 0);
}

void starpu_ndim_filter_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				     unsigned id, unsigned nparts)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_ndim_filter_block(father_interface, child_interface, f, id, nparts, shadow_size);
}

void starpu_ndim_filter_to_tensor(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				  unsigned id, unsigned nparts)
{
	struct starpu_ndim_interface *ndim_father = (struct starpu_ndim_interface *) father_interface;
	STARPU_ASSERT_MSG(ndim_father->ndim == 4, "can only be applied on a 4-dim array");
	if (ndim_father->dev_handle)
		STARPU_ASSERT_MSG(ndim_father->ldn[0]==1, "cannot transfer to a tensor if ldn[0] does not equal to 1");

	struct starpu_ndim_interface ndim_child;
	_starpu_ndim_filter_block(father_interface, &ndim_child, f, id, nparts, 0);

	_interface_assignment_ndim_to_tensor(&ndim_child, child_interface);

	_interface_deallocate(&ndim_child);
}

void starpu_ndim_filter_to_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				 unsigned id, unsigned nparts)
{
	struct starpu_ndim_interface *ndim_father = (struct starpu_ndim_interface *) father_interface;
	STARPU_ASSERT_MSG(ndim_father->ndim == 3, "can only be applied on a 3-dim array");
	if (ndim_father->dev_handle)
		STARPU_ASSERT_MSG(ndim_father->ldn[0]==1, "cannot transfer to a block if ldn[0] does not equal to 1");

	struct starpu_ndim_interface ndim_child;
	_starpu_ndim_filter_block(father_interface, &ndim_child, f, id, nparts, 0);

	_interface_assignment_ndim_to_block(&ndim_child, child_interface);

	_interface_deallocate(&ndim_child);
}

void starpu_ndim_filter_to_matrix(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				  unsigned id, unsigned nparts)
{
	struct starpu_ndim_interface *ndim_father = (struct starpu_ndim_interface *) father_interface;
	STARPU_ASSERT_MSG(ndim_father->ndim == 2, "can only be applied on a 2-dim array");
	if (ndim_father->dev_handle)
		STARPU_ASSERT_MSG(ndim_father->ldn[0]==1, "cannot transfer to a matrix if ldn[0] does not equal to 1");

	struct starpu_ndim_interface ndim_child;
	_starpu_ndim_filter_block(father_interface, &ndim_child, f, id, nparts, 0);

	_interface_assignment_ndim_to_matrix(&ndim_child, child_interface);

	_interface_deallocate(&ndim_child);
}

void starpu_ndim_filter_to_vector(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				  unsigned id, unsigned nparts)
{
	struct starpu_ndim_interface *ndim_father = (struct starpu_ndim_interface *) father_interface;
	STARPU_ASSERT_MSG(ndim_father->ndim == 1, "can only be applied on a 1-dim array");
	if (ndim_father->dev_handle)
		STARPU_ASSERT_MSG(ndim_father->ldn[0]==1, "cannot transfer to a vector if ldn[0] does not equal to 1");

	struct starpu_ndim_interface ndim_child;
	_starpu_ndim_filter_block(father_interface, &ndim_child, f, id, nparts, 0);

	_interface_assignment_ndim_to_vector(&ndim_child, child_interface);

	_interface_deallocate(&ndim_child);
}

void starpu_ndim_filter_to_variable(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				    unsigned id, unsigned nparts)
{
	struct starpu_ndim_interface *ndim_father = (struct starpu_ndim_interface *) father_interface;
	STARPU_ASSERT_MSG(ndim_father->ndim == 0, "can only be applied on a 0-dim array (a variable)");
	STARPU_ASSERT_MSG(id == 0 && nparts == 1, "cannot split a variable");

	_interface_assignment_ndim_to_variable(father_interface, child_interface);
}

void starpu_ndim_filter_pick_ndim(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				  unsigned id, unsigned nparts)
{
	struct starpu_ndim_interface *ndim_father = (struct starpu_ndim_interface *) father_interface;
	struct starpu_ndim_interface *ndim_child = (struct starpu_ndim_interface *) child_interface;

	STARPU_ASSERT_MSG(ndim_father->id == STARPU_NDIM_INTERFACE_ID, "%s can only be applied on a ndim array data", __func__);
	ndim_child->id = STARPU_NDIM_INTERFACE_ID;

	size_t ndim = ndim_father->ndim;
	STARPU_ASSERT_MSG(ndim > 0, "ndim %u must be greater than 0!\n", (unsigned) ndim);

	unsigned dim = 0;
	if (ndim > 1)
		dim = f->filter_arg;

	STARPU_ASSERT_MSG(dim < ndim, "dim %u must be less than %u!\n", dim, (unsigned) ndim);

	uint32_t father_nn = 0;
	uint32_t ni[ndim];
	unsigned i;
	for (i=0; i<ndim; i++)
	{
		ni[i] = ndim_father->nn[i];
		if(i==dim)
			father_nn = ni[i];
	}

	STARPU_ASSERT_MSG(nparts <= father_nn, "cannot split %u elements in %u parts", father_nn, nparts);

	unsigned blocksize = ndim_father->ldn[dim];
	size_t elemsize = ndim_father->elemsize;
	size_t chunk_pos = (size_t)f->filter_arg_ptr;

	STARPU_ASSERT_MSG((chunk_pos + id) < father_nn, "the chosen sub (n-1)dim array should be in the ndim array");

	size_t offset = (chunk_pos + id) * blocksize * elemsize;
	int j;
	_STARPU_MALLOC(ndim_child->nn, (ndim-1)*sizeof(uint32_t));
	if (ndim > 1)
	{
		j = 0;
		for (i=0; i<ndim; i++)
		{
			if (i!=dim)
			{
				ndim_child->nn[j] = ni[i];
				j++;
			}
		}
	}

	_STARPU_MALLOC(ndim_child->ldn, (ndim-1)*sizeof(uint32_t));
	ndim_child->ndim = ndim-1;
	ndim_child->elemsize = elemsize;
	ndim_child->allocsize = elemsize;

	if (ndim_father->dev_handle)
	{
		if (ndim_father->ptr)
			ndim_child->ptr = ndim_father->ptr + offset;
		if (ndim > 1)
		{
			j = 0;
			for (i=0; i<ndim; i++)
			{
				if (i!=dim)
				{
					ndim_child->ldn[j] = ndim_father->ldn[i];
					j++;
				}
			}

			ndim_child->allocsize *= ndim_child->ldn[ndim-2] * ndim_child->nn[ndim-2];
		}

		ndim_child->dev_handle = ndim_father->dev_handle;
		ndim_child->offset = ndim_father->offset + offset;
	}
	else
	{
		for (i=0; i<ndim-1; i++)
			ndim_child->allocsize *= ndim_child->nn[i];
	}
}

void starpu_ndim_filter_5d_pick_tensor(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				       unsigned id, unsigned nparts)
{
	struct starpu_ndim_interface *ndim_father = (struct starpu_ndim_interface *) father_interface;
	STARPU_ASSERT_MSG(ndim_father->ndim == 5, "can only be applied on a 5-dim array");
	if (ndim_father->dev_handle)
		STARPU_ASSERT_MSG(ndim_father->ldn[0]==1, "cannot pick a tensor if ldn[0] does not equal to 1");

	struct starpu_ndim_interface ndim_child;
	starpu_ndim_filter_pick_ndim(father_interface, &ndim_child, f, id, nparts);

	_interface_assignment_ndim_to_tensor(&ndim_child, child_interface);

	_interface_deallocate(&ndim_child);
}

void starpu_ndim_filter_4d_pick_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				      unsigned id, unsigned nparts)
{
	struct starpu_ndim_interface *ndim_father = (struct starpu_ndim_interface *) father_interface;
	STARPU_ASSERT_MSG(ndim_father->ndim == 4, "can only be applied on a 4-dim array");
	if (ndim_father->dev_handle)
		STARPU_ASSERT_MSG(ndim_father->ldn[0]==1, "cannot pick a block if ldn[0] does not equal to 1");

	struct starpu_ndim_interface ndim_child;
	starpu_ndim_filter_pick_ndim(father_interface, &ndim_child, f, id, nparts);

	_interface_assignment_ndim_to_block(&ndim_child, child_interface);

	_interface_deallocate(&ndim_child);
}

void starpu_ndim_filter_3d_pick_matrix(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				       unsigned id, unsigned nparts)
{
	struct starpu_ndim_interface *ndim_father = (struct starpu_ndim_interface *) father_interface;
	STARPU_ASSERT_MSG(ndim_father->ndim == 3, "can only be applied on a 3-dim array");
	if (ndim_father->dev_handle)
		STARPU_ASSERT_MSG(ndim_father->ldn[0]==1, "cannot pick a matrix if ldn[0] does not equal to 1");

	struct starpu_ndim_interface ndim_child;
	starpu_ndim_filter_pick_ndim(father_interface, &ndim_child, f, id, nparts);

	_interface_assignment_ndim_to_matrix(&ndim_child, child_interface);

	_interface_deallocate(&ndim_child);
}

void starpu_ndim_filter_2d_pick_vector(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				       unsigned id, unsigned nparts)
{
	struct starpu_ndim_interface *ndim_father = (struct starpu_ndim_interface *) father_interface;
	STARPU_ASSERT_MSG(ndim_father->ndim == 2, "can only be applied on a 2-dim array");
	if (ndim_father->dev_handle)
		STARPU_ASSERT_MSG(ndim_father->ldn[0]==1, "cannot pick a vector if ldn[0] does not equal to 1");

	struct starpu_ndim_interface ndim_child;
	starpu_ndim_filter_pick_ndim(father_interface, &ndim_child, f, id, nparts);

	_interface_assignment_ndim_to_vector(&ndim_child, child_interface);

	_interface_deallocate(&ndim_child);
}

void starpu_ndim_filter_1d_pick_variable(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					 unsigned id, unsigned nparts)
{
	struct starpu_ndim_interface *ndim_father = (struct starpu_ndim_interface *) father_interface;
	STARPU_ASSERT_MSG(ndim_father->ndim == 1, "can only be applied on a 1-dim array");
	if (ndim_father->dev_handle)
		STARPU_ASSERT_MSG(ndim_father->ldn[0]==1, "cannot pick a variable if ldn[0] does not equal to 1");

	struct starpu_ndim_interface ndim_child;
	starpu_ndim_filter_pick_ndim(father_interface, &ndim_child, f, id, nparts);

	_interface_assignment_ndim_to_variable(&ndim_child, child_interface);

	_interface_deallocate(&ndim_child);
}

static void _interface_deallocate(void *ndim_interface)
{
	struct starpu_ndim_interface *ndarr = (struct starpu_ndim_interface *) ndim_interface;

	free(ndarr->nn);
	free(ndarr->ldn);
}

struct starpu_data_interface_ops *starpu_ndim_filter_pick_tensor_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_tensor_ops;
}

struct starpu_data_interface_ops *starpu_ndim_filter_pick_block_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_block_ops;
}

struct starpu_data_interface_ops *starpu_ndim_filter_pick_matrix_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_matrix_ops;
}

struct starpu_data_interface_ops *starpu_ndim_filter_pick_vector_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_vector_ops;
}

struct starpu_data_interface_ops *starpu_ndim_filter_pick_variable_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_variable_ops;
}

struct starpu_data_interface_ops *starpu_ndim_filter_to_tensor_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_tensor_ops;
}

struct starpu_data_interface_ops *starpu_ndim_filter_to_block_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_block_ops;
}

struct starpu_data_interface_ops *starpu_ndim_filter_to_matrix_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_matrix_ops;
}

struct starpu_data_interface_ops *starpu_ndim_filter_to_vector_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_vector_ops;
}

struct starpu_data_interface_ops *starpu_ndim_filter_to_variable_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_variable_ops;
}

static void _interface_assignment_ndim_to_tensor(void *ndim_interface, void *child_interface)
{
	struct starpu_tensor_interface *tensor = (struct starpu_tensor_interface *) child_interface;
	struct starpu_ndim_interface *ndarr = (struct starpu_ndim_interface *) ndim_interface;

	tensor->id = STARPU_TENSOR_INTERFACE_ID;
	tensor->nx = ndarr->nn[0];
	tensor->ny = ndarr->nn[1];
	tensor->nz = ndarr->nn[2];
	tensor->nt = ndarr->nn[3];
	tensor->elemsize = ndarr->elemsize;
	tensor->ptr = ndarr->ptr;
	tensor->ldy = ndarr->ldn[1];
	tensor->ldz = ndarr->ldn[2];
	tensor->ldt = ndarr->ldn[3];
	tensor->dev_handle = ndarr->dev_handle;
	tensor->offset = ndarr->offset;
}

static void _interface_assignment_ndim_to_block(void *ndim_interface, void *child_interface)
{
	struct starpu_block_interface *block = (struct starpu_block_interface *) child_interface;
	struct starpu_ndim_interface *ndarr = (struct starpu_ndim_interface *) ndim_interface;

	block->id = STARPU_BLOCK_INTERFACE_ID;
	block->nx = ndarr->nn[0];
	block->ny = ndarr->nn[1];
	block->nz = ndarr->nn[2];
	block->elemsize = ndarr->elemsize;
	block->ptr = ndarr->ptr;
	block->ldy = ndarr->ldn[1];
	block->ldz = ndarr->ldn[2];
	block->dev_handle = ndarr->dev_handle;
	block->offset = ndarr->offset;
}

static void _interface_assignment_ndim_to_matrix(void *ndim_interface, void *child_interface)
{
	struct starpu_matrix_interface *matrix = (struct starpu_matrix_interface *) child_interface;
	struct starpu_ndim_interface *ndarr = (struct starpu_ndim_interface *) ndim_interface;

	matrix->id = STARPU_MATRIX_INTERFACE_ID;
	matrix->nx = ndarr->nn[0];
	matrix->ny = ndarr->nn[1];
	matrix->elemsize = ndarr->elemsize;
	matrix->ptr = ndarr->ptr;
	matrix->ld = ndarr->ldn[1];
	if (matrix->ptr)
		matrix->allocsize = matrix->ld * matrix->ny * matrix->elemsize;
	else
		matrix->allocsize = matrix->nx * matrix->ny * matrix->elemsize;
	matrix->dev_handle = ndarr->dev_handle;
	matrix->offset = ndarr->offset;
}

static void _interface_assignment_ndim_to_vector(void *ndim_interface, void *child_interface)
{
	struct starpu_vector_interface *vector = (struct starpu_vector_interface *) child_interface;
	struct starpu_ndim_interface *ndarr = (struct starpu_ndim_interface *) ndim_interface;

	vector->id = STARPU_VECTOR_INTERFACE_ID;
	vector->nx = ndarr->nn[0];
	vector->elemsize = ndarr->elemsize;
	vector->allocsize = vector->nx * vector->elemsize;
	vector->ptr = ndarr->ptr;
	vector->dev_handle = ndarr->dev_handle;
	vector->offset = ndarr->offset;
}

static void _interface_assignment_ndim_to_variable(void *ndim_interface, void *child_interface)
{
	struct starpu_variable_interface *variable = (struct starpu_variable_interface *) child_interface;
	struct starpu_ndim_interface *ndarr = (struct starpu_ndim_interface *) ndim_interface;

	variable->id = STARPU_VARIABLE_INTERFACE_ID;
	variable->elemsize = ndarr->elemsize;
	variable->ptr = ndarr->ptr;
	variable->dev_handle = ndarr->dev_handle;
	variable->offset = ndarr->offset;
}

void starpu_ndim_filter_pick_variable(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nchunks)
{
	struct starpu_ndim_interface *ndim_father = (struct starpu_ndim_interface *) father_interface;
	struct starpu_variable_interface *variable_child = (struct starpu_variable_interface *) child_interface;

	STARPU_ASSERT_MSG(ndim_father->id == STARPU_NDIM_INTERFACE_ID, "%s can only be applied on a ndim array data", __func__);

	size_t ndim = ndim_father->ndim;
	STARPU_ASSERT_MSG(ndim > 0, "ndim %u must be greater than 0!\n", (unsigned) ndim);

	uint32_t nn[ndim];
	unsigned ldn[ndim];
	unsigned i;
	for (i=0; i<ndim; i++)
	{
		nn[i] = ndim_father->nn[i];
		ldn[i] = ndim_father->ldn[i];
	}

	size_t elemsize = ndim_father->elemsize;
	uint32_t* chunk_pos = (uint32_t*)f->filter_arg_ptr;
	int b = 1;
	size_t offset = 0;
	for (i = 0; i < ndim; i++)
	{
		if(chunk_pos[i] >= nn[i])
		{
			b = 0;
			break;
		}
		offset += chunk_pos[i]*ldn[i]*elemsize;
	}

	STARPU_ASSERT_MSG(b == 1, "the chosen variable should be in the ndim array");

	/* update the child's interface */
	variable_child->id = STARPU_VARIABLE_INTERFACE_ID;
	variable_child->elemsize = elemsize;

	/* is the information on this node valid ? */
	if (ndim_father->dev_handle)
	{
		if (ndim_father->ptr)
			variable_child->ptr = ndim_father->ptr + offset;
		variable_child->dev_handle = ndim_father->dev_handle;
		variable_child->offset = ndim_father->offset + offset;
	}
}
