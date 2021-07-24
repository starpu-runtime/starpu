/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static void _starpu_block_filter_block(int dim, void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
			       unsigned id, unsigned nparts, uintptr_t shadow_size)
{
	struct starpu_block_interface *block_father = (struct starpu_block_interface *) father_interface;
        struct starpu_block_interface *block_child = (struct starpu_block_interface *) child_interface;

        unsigned blocksize;
	/* the element will be split, in case horizontal, it's nx, in case vertical, it's ny, in case depth, it's nz*/
	uint32_t nn;
	uint32_t nx;
	uint32_t ny;
	uint32_t nz;

	switch(dim)
	{
		/* horizontal*/
		case 1:
			/* actual number of elements */
			nx = block_father->nx - 2 * shadow_size;
			ny = block_father->ny;
			nz = block_father->nz;
			nn = nx;
			blocksize = 1;
			break;
		/* vertical*/
		case 2:
			nx = block_father->nx;
			/* actual number of elements */
			ny = block_father->ny - 2 * shadow_size;
			nz = block_father->nz;
			nn = ny;
			blocksize = block_father->ldy;
			break;
		/* depth*/
		case 3:
			nx = block_father->nx;
			ny = block_father->ny;
			/* actual number of elements */
			nz = block_father->nz - 2 * shadow_size;
			nn = nz;
			blocksize = block_father->ldz;
			break;
	}

	size_t elemsize = block_father->elemsize;

	STARPU_ASSERT_MSG(nparts <= nn, "cannot split %u elements in %u parts", nn, nparts);

	uint32_t child_nn;
	size_t offset;
	starpu_filter_nparts_compute_chunk_size_and_offset(nn, nparts, elemsize, id, blocksize, &child_nn, &offset);

	child_nn += 2 * shadow_size;

	STARPU_ASSERT_MSG(block_father->id == STARPU_BLOCK_INTERFACE_ID, "%s can only be applied on a block data", __func__);
	block_child->id = block_father->id;

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

	if (block_father->dev_handle)
	{
		if (block_father->ptr)
                	block_child->ptr = block_father->ptr + offset;
                block_child->ldy = block_father->ldy;
                block_child->ldz = block_father->ldz;
                block_child->dev_handle = block_father->dev_handle;
                block_child->offset = block_father->offset + offset;
	}
}

void starpu_block_filter_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
			       unsigned id, unsigned nparts)
{
	_starpu_block_filter_block(1, father_interface, child_interface, f, id, nparts, 0);
}

void starpu_block_filter_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				      unsigned id, unsigned nparts)
{
        uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

        _starpu_block_filter_block(1, father_interface, child_interface, f, id, nparts, shadow_size);
}

void starpu_block_filter_vertical_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					unsigned id, unsigned nparts)
{
	_starpu_block_filter_block(2, father_interface, child_interface, f, id, nparts, 0);
}

void starpu_block_filter_vertical_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					       unsigned id, unsigned nparts)
{
        uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

        _starpu_block_filter_block(2, father_interface, child_interface, f, id, nparts, shadow_size);
}

void starpu_block_filter_depth_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				     unsigned id, unsigned nparts)
{
	_starpu_block_filter_block(3, father_interface, child_interface, f, id, nparts, 0);
}

void starpu_block_filter_depth_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					    unsigned id, unsigned nparts)
{
        uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

        _starpu_block_filter_block(3, father_interface, child_interface, f, id, nparts, shadow_size);
}
