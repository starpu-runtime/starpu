/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static void _starpu_tensor_filter_block(int dim, void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
		   unsigned id, unsigned nparts, uintptr_t shadow_size)
{
	struct starpu_tensor_interface *tensor_parent = (struct starpu_tensor_interface *) parent_interface;
	struct starpu_tensor_interface *tensor_child = (struct starpu_tensor_interface *) child_interface;

	unsigned blocksize;
	/* the element will be split, in case horizontal, it's nx, in case vertical, it's ny, in case depth, it's nz, in case time, it's nt*/
	uint32_t nn;
	uint32_t nx;
	uint32_t ny;
	uint32_t nz;
	uint32_t nt;

	switch(dim)
	{
	case 1: /* horizontal*/
		/* actual number of elements */
		nx = tensor_parent->nx - 2 * shadow_size;
		ny = tensor_parent->ny;
		nz = tensor_parent->nz;
		nt = tensor_parent->nt;
		nn = nx;
		blocksize = 1;
		break;

	case 2: /* vertical*/
		nx = tensor_parent->nx;
		/* actual number of elements */
		ny = tensor_parent->ny - 2 * shadow_size;
		nz = tensor_parent->nz;
		nt = tensor_parent->nt;
		nn = ny;
		blocksize = tensor_parent->ldy;
		break;

	case 3: /* depth*/
		nx = tensor_parent->nx;
		ny = tensor_parent->ny;
		/* actual number of elements */
		nz = tensor_parent->nz - 2 * shadow_size;
		nt = tensor_parent->nt;
		nn = nz;
		blocksize = tensor_parent->ldz;
		break;

	case 4: /* time*/
		nx = tensor_parent->nx;
		ny = tensor_parent->ny;
		nz = tensor_parent->nz;
		/* actual number of elements */
		nt = tensor_parent->nt - 2 * shadow_size;
		nn = nt;
		blocksize = tensor_parent->ldt;
		break;
	default:
		STARPU_ASSERT_MSG(0, "Unknown value for dim");
	}

	size_t elemsize = tensor_parent->elemsize;

	STARPU_ASSERT_MSG(nparts <= nn, "cannot split %u elements in %u parts", nn, nparts);

	uint32_t child_nn;
	size_t offset;
	starpu_filter_nparts_compute_chunk_size_and_offset(nn, nparts, elemsize, id, blocksize, &child_nn, &offset);

	child_nn += 2 * shadow_size;

	STARPU_ASSERT_MSG(tensor_parent->id == STARPU_TENSOR_INTERFACE_ID, "%s can only be applied on a tensor data", __func__);
	tensor_child->id = tensor_parent->id;

	switch(dim)
	{
	case 1:
		tensor_child->nx = child_nn;
		tensor_child->ny = ny;
		tensor_child->nz = nz;
		tensor_child->nt = nt;
		break;
	case 2:
		tensor_child->nx = nx;
		tensor_child->ny = child_nn;
		tensor_child->nz = nz;
		tensor_child->nt = nt;
		break;
	case 3:
		tensor_child->nx = nx;
		tensor_child->ny = ny;
		tensor_child->nz = child_nn;
		tensor_child->nt = nt;
		break;
	case 4:
		tensor_child->nx = nx;
		tensor_child->ny = ny;
		tensor_child->nz = nz;
		tensor_child->nt = child_nn;
		break;
	default:
		STARPU_ASSERT_MSG(0, "Unknown value for dim");
	}

	tensor_child->elemsize = elemsize;

	if (tensor_parent->dev_handle)
	{
		if (tensor_parent->ptr)
			tensor_child->ptr = tensor_parent->ptr + offset;
		tensor_child->ldy = tensor_parent->ldy;
		tensor_child->ldz = tensor_parent->ldz;
		tensor_child->ldt = tensor_parent->ldt;
		tensor_child->dev_handle = tensor_parent->dev_handle;
		tensor_child->offset = tensor_parent->offset + offset;
	}
}

void starpu_tensor_filter_block(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				unsigned id, unsigned nparts)
{
	_starpu_tensor_filter_block(1, parent_interface, child_interface, f, id, nparts, 0);
}

void starpu_tensor_filter_block_shadow(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				       unsigned id, unsigned nparts)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_tensor_filter_block(1, parent_interface, child_interface, f, id, nparts, shadow_size);
}

void starpu_tensor_filter_vertical_block(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					 unsigned id, unsigned nparts)
{
	_starpu_tensor_filter_block(2, parent_interface, child_interface, f, id, nparts, 0);
}

void starpu_tensor_filter_vertical_block_shadow(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
						unsigned id, unsigned nparts)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_tensor_filter_block(2, parent_interface, child_interface, f, id, nparts, shadow_size);
}

void starpu_tensor_filter_depth_block(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				      unsigned id, unsigned nparts)
{
	_starpu_tensor_filter_block(3, parent_interface, child_interface, f, id, nparts, 0);
}

void starpu_tensor_filter_depth_block_shadow(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					     unsigned id, unsigned nparts)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_tensor_filter_block(3, parent_interface, child_interface, f, id, nparts, shadow_size);
}

void starpu_tensor_filter_time_block(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				     unsigned id, unsigned nparts)
{
	_starpu_tensor_filter_block(4, parent_interface, child_interface, f, id, nparts, 0);
}

void starpu_tensor_filter_time_block_shadow(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					    unsigned id, unsigned nparts)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_tensor_filter_block(4, parent_interface, child_interface, f, id, nparts, shadow_size);
}

static void _starpu_tensor_filter_pick_block(int dim, void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					     unsigned id, unsigned nparts)
{
	struct starpu_tensor_interface *tensor_parent = (struct starpu_tensor_interface *) parent_interface;
	struct starpu_block_interface *block_child = (struct starpu_block_interface *) child_interface;

	unsigned blocksize;
	uint32_t nn;
	uint32_t nx = tensor_parent->nx;
	uint32_t ny = tensor_parent->ny;
	uint32_t nz = tensor_parent->nz;
	uint32_t nt = tensor_parent->nt;

	switch(dim)
	{
		/* along y-axis */
		case 1:
			nn = ny;
			blocksize = tensor_parent->ldy;
			break;
		/* along z-axis */
		case 2:
			nn = nz;
			blocksize = tensor_parent->ldz;
			break;
		/* along t-axis */
		case 3:
			nn = nt;
			blocksize = tensor_parent->ldt;
			break;
		default:
			STARPU_ASSERT_MSG(0, "Unknown value for dim");
	}

	size_t elemsize = tensor_parent->elemsize;

	size_t chunk_pos = (size_t)f->filter_arg_ptr;

	STARPU_ASSERT_MSG(nparts <= nn, "cannot get %u blocks", nparts);
	STARPU_ASSERT_MSG((chunk_pos + id) < nn, "the chosen block should be in the tensor");

	size_t offset = (chunk_pos + id) * blocksize * elemsize;

	STARPU_ASSERT_MSG(tensor_parent->id == STARPU_TENSOR_INTERFACE_ID, "%s can only be applied on a tensor data", __func__);
	block_child->id = STARPU_BLOCK_INTERFACE_ID;

	switch(dim)
	{
		/* along y-axis */
		case 1:
			block_child->nx = nx;
			block_child->ny = nz;
			block_child->nz = nt;
			break;
		/* along z-axis */
		case 2:
			block_child->nx = nx;
			block_child->ny = ny;
			block_child->nz = nt;
			break;
		/* along t-axis */
		case 3:
			block_child->nx = nx;
			block_child->ny = ny;
			block_child->nz = nz;
			break;
		default:
			STARPU_ASSERT_MSG(0, "Unknown value for dim");
	}

	block_child->elemsize = elemsize;

	if (tensor_parent->dev_handle)
	{
		if (tensor_parent->ptr)
			block_child->ptr = tensor_parent->ptr + offset;
		switch(dim)
		{
			/* along y-axis */
			case 1:
				block_child->ldy = tensor_parent->ldz;
				block_child->ldz = tensor_parent->ldt;
				break;
			/* along z-axis */
			case 2:
				block_child->ldy = tensor_parent->ldy;
				block_child->ldz = tensor_parent->ldt;
				break;
			/* along t-axis */
			case 3:
				block_child->ldy = tensor_parent->ldy;
				block_child->ldz = tensor_parent->ldz;
				break;
			default:
				STARPU_ASSERT_MSG(0, "Unknown value for dim");
		}
		block_child->dev_handle = tensor_parent->dev_handle;
		block_child->offset = tensor_parent->offset + offset;
	}
}
void starpu_tensor_filter_pick_block_t(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				       unsigned id, unsigned nparts)
{
	_starpu_tensor_filter_pick_block(3, parent_interface, child_interface, f, id, nparts);
}

void starpu_tensor_filter_pick_block_z(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				       unsigned id, unsigned nparts)
{
	_starpu_tensor_filter_pick_block(2, parent_interface, child_interface, f, id, nparts);
}

void starpu_tensor_filter_pick_block_y(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				       unsigned id, unsigned nparts)
{
	_starpu_tensor_filter_pick_block(1, parent_interface, child_interface, f, id, nparts);
}

struct starpu_data_interface_ops *starpu_tensor_filter_pick_block_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_block_ops;
}

void starpu_tensor_filter_pick_variable(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nchunks)
{
	struct starpu_tensor_interface *tensor_parent = (struct starpu_tensor_interface *) parent_interface;
	/* each chunk becomes a variable */
	struct starpu_variable_interface *variable_child = (struct starpu_variable_interface *) child_interface;

	uint32_t nx = tensor_parent->nx;
	uint32_t ny = tensor_parent->ny;
	uint32_t nz = tensor_parent->nz;
	uint32_t nt = tensor_parent->nt;

	unsigned ldy = tensor_parent->ldy;
	unsigned ldz = tensor_parent->ldz;
	unsigned ldt = tensor_parent->ldt;

	size_t elemsize = tensor_parent->elemsize;

	uint32_t* chunk_pos = (uint32_t*)f->filter_arg_ptr;
	// int i;
	// for(i=0; i<4; i++)
	// {
	// 	printf("pos is %d\n", chunk_pos[i]);
	// }

	STARPU_ASSERT_MSG((chunk_pos[0] < nx)&&(chunk_pos[1] < ny)&&(chunk_pos[2] < nz)&&(chunk_pos[3] < nt), "the chosen variable should be in the tensor");

	size_t offset = (chunk_pos[3] * ldt + chunk_pos[2] * ldz + chunk_pos[1] * ldy + chunk_pos[0]) * elemsize;

	STARPU_ASSERT_MSG(tensor_parent->id == STARPU_TENSOR_INTERFACE_ID, "%s can only be applied on a tensor data", __func__);

	/* update the child's interface */
	variable_child->id = STARPU_VARIABLE_INTERFACE_ID;
	variable_child->elemsize = elemsize;

	/* is the information on this node valid ? */
	if (tensor_parent->dev_handle)
	{
		if (tensor_parent->ptr)
			variable_child->ptr = tensor_parent->ptr + offset;
		variable_child->dev_handle = tensor_parent->dev_handle;
		variable_child->offset = tensor_parent->offset + offset;
	}
}

struct starpu_data_interface_ops *starpu_tensor_filter_pick_variable_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_variable_ops;
}
