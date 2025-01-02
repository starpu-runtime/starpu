/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010-2010  Mehdi Juhoor
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
#include <datawizard/filters.h>

/*
 * an example of a dummy partition function : blocks ...
 */

static void _starpu_matrix_filter_block(int dim, void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks, uintptr_t shadow_size)
{
	struct starpu_matrix_interface *matrix_parent = (struct starpu_matrix_interface *) parent_interface;
	struct starpu_matrix_interface *matrix_child = (struct starpu_matrix_interface *) child_interface;

	unsigned blocksize;
	/* the element will be split, in case horizontal, it's nx, in case vertical, it's ny*/
	size_t nn;
	size_t nx;
	size_t ny;

	switch(dim)
	{
		/* horizontal*/
		case 1:
			/* actual number of elements */
			nx = matrix_parent->nx - 2 * shadow_size;
			ny = matrix_parent->ny;
			nn = nx;
			blocksize = 1;
			break;
		/* vertical*/
		case 2:
			nx = matrix_parent->nx;
			/* actual number of elements */
			ny = matrix_parent->ny - 2 * shadow_size;
			nn = ny;
			blocksize = matrix_parent->ld;
			break;
		default:
			STARPU_ASSERT_MSG(0, "Unknown value for dim");
	}

	size_t elemsize = matrix_parent->elemsize;

	STARPU_ASSERT_MSG(nchunks <= nn, "cannot split %zu elements in %u parts", nn, nchunks);

	size_t child_nn;
	size_t offset;

	starpu_filter_nparts_compute_chunk_size_and_offset(nn, nchunks, elemsize, id, blocksize, &child_nn, &offset);

	child_nn += 2 * shadow_size;

	STARPU_ASSERT_MSG(matrix_parent->id == STARPU_MATRIX_INTERFACE_ID, "%s can only be applied on a matrix data", __func__);

	/* update the child's interface */
	matrix_child->id = matrix_parent->id;

	switch(dim)
	{
		case 1:
			matrix_child->nx = child_nn;
			matrix_child->ny = ny;
			break;
		case 2:
			matrix_child->nx = nx;
			matrix_child->ny = child_nn;
			break;
		default:
			STARPU_ASSERT_MSG(0, "Unknown value for dim");
	}

	matrix_child->elemsize = elemsize;

	/* is the information on this node valid ? */
	if (matrix_parent->dev_handle)
	{
		if (matrix_parent->ptr)
			matrix_child->ptr = matrix_parent->ptr + offset;
		matrix_child->ld = matrix_parent->ld;
		matrix_child->dev_handle = matrix_parent->dev_handle;
		matrix_child->offset = matrix_parent->offset + offset;
		matrix_child->allocsize = matrix_child->ld * matrix_child->ny * elemsize;
	}
	else
		matrix_child->allocsize = matrix_child->nx * matrix_child->ny * elemsize;
}

void starpu_matrix_filter_block(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	_starpu_matrix_filter_block(1, parent_interface, child_interface, f, id, nchunks, 0);
}

/*
 * an example of a dummy partition function : blocks ...
 */
void starpu_matrix_filter_block_shadow(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_matrix_filter_block(1, parent_interface, child_interface, f, id, nchunks, shadow_size);
}

void starpu_matrix_filter_vertical_block(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	_starpu_matrix_filter_block(2, parent_interface, child_interface, f, id, nchunks, 0);
}

void starpu_matrix_filter_vertical_block_shadow(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_matrix_filter_block(2, parent_interface, child_interface, f, id, nchunks, shadow_size);
}

void starpu_matrix_filter_pick_vector_y(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	struct starpu_matrix_interface *matrix_parent = (struct starpu_matrix_interface *) parent_interface;
	/* each chunk becomes a vector */
	struct starpu_vector_interface *vector_child = (struct starpu_vector_interface *) child_interface;

	unsigned blocksize;

	size_t nx;
	size_t ny;

	/* actual number of elements */
	nx = matrix_parent->nx;
	ny = matrix_parent->ny;
	blocksize = nx;

	size_t elemsize = matrix_parent->elemsize;

	uintptr_t chunk_pos = (uintptr_t)f->filter_arg_ptr;

	STARPU_ASSERT_MSG(nchunks <= nx, "cannot get %u vectors", nchunks);
	STARPU_ASSERT_MSG((chunk_pos + id) < ny, "the chosen vector should be in the matrix");

	size_t offset = (chunk_pos + id) * blocksize * elemsize;

	STARPU_ASSERT_MSG(matrix_parent->id == STARPU_MATRIX_INTERFACE_ID, "%s can only be applied on a matrix data", __func__);

	/* update the child's interface */
	vector_child->id = STARPU_VECTOR_INTERFACE_ID;
	vector_child->nx = nx;
	vector_child->elemsize = elemsize;
	vector_child->allocsize = vector_child->nx * elemsize;

	/* is the information on this node valid ? */
	if (matrix_parent->dev_handle)
	{
		if (matrix_parent->ptr)
			vector_child->ptr = matrix_parent->ptr + offset;
		vector_child->dev_handle = matrix_parent->dev_handle;
		vector_child->offset = matrix_parent->offset + offset;
	}
}

struct starpu_data_interface_ops *starpu_matrix_filter_pick_vector_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_vector_ops;
}

void starpu_matrix_filter_pick_variable(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nchunks)
{
	struct starpu_matrix_interface *matrix_parent = (struct starpu_matrix_interface *) parent_interface;
	/* each chunk becomes a variable */
	struct starpu_variable_interface *variable_child = (struct starpu_variable_interface *) child_interface;

	unsigned blocksize;

	size_t nx;
	size_t ld;
	size_t ny;

	/* actual number of elements */
	nx = matrix_parent->nx;
	ld = matrix_parent->ld;
	ny = matrix_parent->ny;
	blocksize = ld;

	size_t elemsize = matrix_parent->elemsize;

	uint32_t* chunk_pos = (uint32_t*)f->filter_arg_ptr;
	// int i;
	// for(i=0; i<2; i++)
	// {
	// 	printf("pos is %d\n", chunk_pos[i]);
	// }

	STARPU_ASSERT_MSG((chunk_pos[0] < nx)&&(chunk_pos[1] < ny), "the chosen variable should be in the matrix");

	size_t offset = (((chunk_pos[1]) * blocksize) + chunk_pos[0]) * elemsize;

	STARPU_ASSERT_MSG(matrix_parent->id == STARPU_MATRIX_INTERFACE_ID, "%s can only be applied on a matrix data", __func__);

	/* update the child's interface */
	variable_child->id = STARPU_VARIABLE_INTERFACE_ID;
	variable_child->elemsize = elemsize;

	/* is the information on this node valid ? */
	if (matrix_parent->dev_handle)
	{
		if (matrix_parent->ptr)
			variable_child->ptr = matrix_parent->ptr + offset;
		variable_child->dev_handle = matrix_parent->dev_handle;
		variable_child->offset = matrix_parent->offset + offset;
	}
}

struct starpu_data_interface_ops *starpu_matrix_filter_pick_variable_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_variable_ops;
}
