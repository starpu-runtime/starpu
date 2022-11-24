/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
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

static void _starpu_matrix_filter_block(int dim, void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks, uintptr_t shadow_size)
{
	struct starpu_matrix_interface *matrix_father = (struct starpu_matrix_interface *) father_interface;
	struct starpu_matrix_interface *matrix_child = (struct starpu_matrix_interface *) child_interface;

	unsigned blocksize;
	/* the element will be split, in case horizontal, it's nx, in case vertical, it's ny*/
	uint32_t nn;
	uint32_t nx;
	uint32_t ny;

	switch(dim)
	{
		/* horizontal*/
		case 1:
			/* actual number of elements */
			nx = matrix_father->nx - 2 * shadow_size;
			ny = matrix_father->ny;
			nn = nx;
			blocksize = 1;
			break;
		/* vertical*/
		case 2:
			nx = matrix_father->nx;
			/* actual number of elements */
			ny = matrix_father->ny - 2 * shadow_size;
			nn = ny;
			blocksize = matrix_father->ld;
			break;
		default:
			STARPU_ASSERT_MSG(0, "Unknown value for dim");
	}

	size_t elemsize = matrix_father->elemsize;

	STARPU_ASSERT_MSG(nchunks <= nn, "cannot split %u elements in %u parts", nn, nchunks);

	uint32_t child_nn;
	size_t offset;

	starpu_filter_nparts_compute_chunk_size_and_offset(nn, nchunks, elemsize, id, blocksize, &child_nn, &offset);

	child_nn += 2 * shadow_size;

	STARPU_ASSERT_MSG(matrix_father->id == STARPU_MATRIX_INTERFACE_ID, "%s can only be applied on a matrix data", __func__);

	/* update the child's interface */
	matrix_child->id = matrix_father->id;

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
	if (matrix_father->dev_handle)
	{
		if (matrix_father->ptr)
			matrix_child->ptr = matrix_father->ptr + offset;
		matrix_child->ld = matrix_father->ld;
		matrix_child->dev_handle = matrix_father->dev_handle;
		matrix_child->offset = matrix_father->offset + offset;
		matrix_child->allocsize = matrix_child->ld * matrix_child->ny * elemsize;
	}
	else
		matrix_child->allocsize = matrix_child->nx * matrix_child->ny * elemsize;
}

void starpu_matrix_filter_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	_starpu_matrix_filter_block(1, father_interface, child_interface, f, id, nchunks, 0);
}

/*
 * an example of a dummy partition function : blocks ...
 */
void starpu_matrix_filter_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_matrix_filter_block(1, father_interface, child_interface, f, id, nchunks, shadow_size);
}

void starpu_matrix_filter_vertical_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	_starpu_matrix_filter_block(2, father_interface, child_interface, f, id, nchunks, 0);
}

void starpu_matrix_filter_vertical_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_matrix_filter_block(2, father_interface, child_interface, f, id, nchunks, shadow_size);
}

void starpu_matrix_filter_pick_vector_y(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	struct starpu_matrix_interface *matrix_father = (struct starpu_matrix_interface *) father_interface;
	/* each chunk becomes a vector */
	struct starpu_vector_interface *vector_child = (struct starpu_vector_interface *) child_interface;

	unsigned blocksize;

	uint32_t nx;
	uint32_t ny;
	
	/* actual number of elements */
	nx = matrix_father->nx;
	ny = matrix_father->ny;
	blocksize = nx;

	size_t elemsize = matrix_father->elemsize;

	uintptr_t chunk_pos = (uintptr_t)f->filter_arg_ptr;

	STARPU_ASSERT_MSG(nchunks <= nx, "cannot get %u vectors", nchunks);
	STARPU_ASSERT_MSG((chunk_pos + id) < ny, "the chosen vector should be in the matrix");

	size_t offset = (chunk_pos + id) * blocksize * elemsize;

	STARPU_ASSERT_MSG(matrix_father->id == STARPU_MATRIX_INTERFACE_ID, "%s can only be applied on a matrix data", __func__);

	/* update the child's interface */
	vector_child->id = STARPU_VECTOR_INTERFACE_ID;
	vector_child->nx = nx;
	vector_child->elemsize = elemsize;
	vector_child->allocsize = vector_child->nx * elemsize;

	/* is the information on this node valid ? */
	if (matrix_father->dev_handle)
	{
		if (matrix_father->ptr)
			vector_child->ptr = matrix_father->ptr + offset;
		vector_child->dev_handle = matrix_father->dev_handle;
		vector_child->offset = matrix_father->offset + offset;
	}
}

struct starpu_data_interface_ops *starpu_matrix_filter_pick_vector_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_vector_ops;
}

void starpu_matrix_filter_pick_variable(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nchunks)
{
	struct starpu_matrix_interface *matrix_father = (struct starpu_matrix_interface *) father_interface;
	/* each chunk becomes a variable */
	struct starpu_variable_interface *variable_child = (struct starpu_variable_interface *) child_interface;

	unsigned blocksize;

	uint32_t nx;
	uint32_t ny;
	
	/* actual number of elements */
	nx = matrix_father->nx;
	ny = matrix_father->ny;
	blocksize = nx;

	size_t elemsize = matrix_father->elemsize;

	uint32_t* chunk_pos = (uint32_t*)f->filter_arg_ptr;
	// for(int i=0; i<2; i++)
	// {
	// 	printf("pos is %d\n", chunk_pos[i]);
	// }

	STARPU_ASSERT_MSG((chunk_pos[0] < nx)&&(chunk_pos[1] < ny), "the chosen variable should be in the matrix");

	size_t offset = (((chunk_pos[1]) * blocksize) + chunk_pos[0]) * elemsize;

	STARPU_ASSERT_MSG(matrix_father->id == STARPU_MATRIX_INTERFACE_ID, "%s can only be applied on a matrix data", __func__);

	/* update the child's interface */
	variable_child->id = STARPU_VARIABLE_INTERFACE_ID;
	variable_child->elemsize = elemsize;

	/* is the information on this node valid ? */
	if (matrix_father->dev_handle)
	{
		if (matrix_father->ptr)
			variable_child->ptr = matrix_father->ptr + offset;
		variable_child->dev_handle = matrix_father->dev_handle;
		variable_child->offset = matrix_father->offset + offset;
	}
}

struct starpu_data_interface_ops *starpu_matrix_filter_pick_variable_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_variable_ops;
}
