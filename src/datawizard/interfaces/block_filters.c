/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static void _starpu_block_filter_block(int dim, void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				       unsigned id, unsigned nparts, uintptr_t shadow_size)
{
	struct starpu_block_interface *block_parent = (struct starpu_block_interface *) parent_interface;
	struct starpu_block_interface *block_child = (struct starpu_block_interface *) child_interface;

	unsigned blocksize;
	/* the element will be split, in case horizontal, it's nx, in case vertical, it's ny, in case depth, it's nz*/
	size_t nn;
	size_t nx;
	size_t ny;
	size_t nz;

	switch(dim)
	{
		/* horizontal*/
		case 1:
			/* actual number of elements */
			nx = block_parent->nx - 2 * shadow_size;
			ny = block_parent->ny;
			nz = block_parent->nz;
			nn = nx;
			blocksize = 1;
			break;
		/* vertical*/
		case 2:
			nx = block_parent->nx;
			/* actual number of elements */
			ny = block_parent->ny - 2 * shadow_size;
			nz = block_parent->nz;
			nn = ny;
			blocksize = block_parent->ldy;
			break;
		/* depth*/
		case 3:
			nx = block_parent->nx;
			ny = block_parent->ny;
			/* actual number of elements */
			nz = block_parent->nz - 2 * shadow_size;
			nn = nz;
			blocksize = block_parent->ldz;
			break;
		default:
			STARPU_ASSERT_MSG(0, "Unknown value for dim");
	}

	size_t elemsize = block_parent->elemsize;

	STARPU_ASSERT_MSG(nparts <= nn, "cannot split %zu elements in %u parts", nn, nparts);

	size_t child_nn;
	size_t offset;
	starpu_filter_nparts_compute_chunk_size_and_offset(nn, nparts, elemsize, id, blocksize, &child_nn, &offset);

	child_nn += 2 * shadow_size;

	STARPU_ASSERT_MSG(block_parent->id == STARPU_BLOCK_INTERFACE_ID, "%s can only be applied on a block data", __func__);
	block_child->id = block_parent->id;

	switch(dim)
	{
		case 1:
			block_child->nx = child_nn;
			block_child->ny = ny;
			block_child->nz = nz;
			break;
		case 2:
			block_child->nx = nx;
			block_child->ny = child_nn;
			block_child->nz = nz;
			break;
		case 3:
			block_child->nx = nx;
			block_child->ny = ny;
			block_child->nz = child_nn;
			break;
	}

	block_child->elemsize = elemsize;

	if (block_parent->dev_handle)
	{
		if (block_parent->ptr)
			block_child->ptr = block_parent->ptr + offset;
		block_child->ldy = block_parent->ldy;
		block_child->ldz = block_parent->ldz;
		block_child->dev_handle = block_parent->dev_handle;
		block_child->offset = block_parent->offset + offset;
	}
}

void starpu_block_filter_block(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
			       unsigned id, unsigned nparts)
{
	_starpu_block_filter_block(1, parent_interface, child_interface, f, id, nparts, 0);
}

void starpu_block_filter_block_shadow(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				      unsigned id, unsigned nparts)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_block_filter_block(1, parent_interface, child_interface, f, id, nparts, shadow_size);
}

void starpu_block_filter_vertical_block(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					unsigned id, unsigned nparts)
{
	_starpu_block_filter_block(2, parent_interface, child_interface, f, id, nparts, 0);
}

void starpu_block_filter_vertical_block_shadow(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					       unsigned id, unsigned nparts)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_block_filter_block(2, parent_interface, child_interface, f, id, nparts, shadow_size);
}

void starpu_block_filter_depth_block(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				     unsigned id, unsigned nparts)
{
	_starpu_block_filter_block(3, parent_interface, child_interface, f, id, nparts, 0);
}

void starpu_block_filter_depth_block_shadow(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					    unsigned id, unsigned nparts)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_block_filter_block(3, parent_interface, child_interface, f, id, nparts, shadow_size);
}

static void _starpu_block_filter_pick_matrix(int dim, void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					    unsigned id, unsigned nparts)
{
	struct starpu_block_interface *block_parent = (struct starpu_block_interface *) parent_interface;
	struct starpu_matrix_interface *matrix_child = (struct starpu_matrix_interface *) child_interface;

	unsigned blocksize;

	size_t nn;
	size_t nx = block_parent->nx;
	size_t ny = block_parent->ny;
	size_t nz = block_parent->nz;

	switch(dim)
	{
		/* along y-axis */
		case 1:
			nn = ny;
			blocksize = block_parent->ldy;
			break;
		/* along z-axis */
		case 2:
			nn = nz;
			blocksize = block_parent->ldz;
			break;
		default:
			STARPU_ASSERT_MSG(0, "Unknown value for dim");
	}

	size_t elemsize = block_parent->elemsize;

	size_t chunk_pos = (size_t)f->filter_arg_ptr;

	STARPU_ASSERT_MSG(nparts <= nn, "cannot get %u matrix", nparts);
	STARPU_ASSERT_MSG((chunk_pos + id) < nn, "the chosen matrix should be in the block");

	size_t offset = (chunk_pos + id) * blocksize * elemsize;

	STARPU_ASSERT_MSG(block_parent->id == STARPU_BLOCK_INTERFACE_ID, "%s can only be applied on a block data", __func__);
	matrix_child->id = STARPU_MATRIX_INTERFACE_ID;

	switch(dim)
	{
		/* along y-axis */
		case 1:
			matrix_child->nx = nx;
			matrix_child->ny = nz;
			break;
		/* along z-axis */
		case 2:
			matrix_child->nx = nx;
			matrix_child->ny = ny;
			break;
		default:
			STARPU_ASSERT_MSG(0, "Unknown value for dim");
	}

	matrix_child->elemsize = elemsize;
	matrix_child->allocsize = matrix_child->nx * matrix_child->ny * elemsize;

	if (block_parent->dev_handle)
	{
		if (block_parent->ptr)
			matrix_child->ptr = block_parent->ptr + offset;
		switch(dim)
		{
			/* along y-axis */
			case 1:
				matrix_child->ld = block_parent->ldz;
				break;
			/* along z-axis */
			case 2:
				matrix_child->ld = block_parent->ldy;
				break;
			default:
				STARPU_ASSERT_MSG(0, "Unknown value for dim");
		}
		matrix_child->dev_handle = block_parent->dev_handle;
		matrix_child->offset = block_parent->offset + offset;
	}
}

void starpu_block_filter_pick_matrix_z(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					    unsigned id, unsigned nparts)
{
	_starpu_block_filter_pick_matrix(2, parent_interface, child_interface, f, id, nparts);
}

void starpu_block_filter_pick_matrix_y(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					    unsigned id, unsigned nparts)
{
	_starpu_block_filter_pick_matrix(1, parent_interface, child_interface, f, id, nparts);
}

struct starpu_data_interface_ops *starpu_block_filter_pick_matrix_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_matrix_ops;
}


void starpu_block_filter_pick_variable(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nchunks)
{
	struct starpu_block_interface *block_parent = (struct starpu_block_interface *) parent_interface;
	/* each chunk becomes a variable */
	struct starpu_variable_interface *variable_child = (struct starpu_variable_interface *) child_interface;

	size_t nx = block_parent->nx;
	size_t ny = block_parent->ny;
	size_t nz = block_parent->nz;

	size_t ldy = block_parent->ldy;
	size_t ldz = block_parent->ldz;
	size_t elemsize = block_parent->elemsize;

	uint32_t* chunk_pos = (uint32_t*)f->filter_arg_ptr;
	// int i;
	// for(i=0; i<3; i++)
	// {
	// 	printf("pos is %d\n", chunk_pos[i]);
	// }

	STARPU_ASSERT_MSG((chunk_pos[0] < nx)&&(chunk_pos[1] < ny)&&(chunk_pos[2] < nz), "the chosen variable should be in the block");

	size_t offset = (chunk_pos[2] * ldz + chunk_pos[1] * ldy + chunk_pos[0]) * elemsize;

	STARPU_ASSERT_MSG(block_parent->id == STARPU_BLOCK_INTERFACE_ID, "%s can only be applied on a block data", __func__);

	/* update the child's interface */
	variable_child->id = STARPU_VARIABLE_INTERFACE_ID;
	variable_child->elemsize = elemsize;

	/* is the information on this node valid ? */
	if (block_parent->dev_handle)
	{
		if (block_parent->ptr)
			variable_child->ptr = block_parent->ptr + offset;
		variable_child->dev_handle = block_parent->dev_handle;
		variable_child->offset = block_parent->offset + offset;
	}
}

struct starpu_data_interface_ops *starpu_block_filter_pick_variable_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_variable_ops;
}
