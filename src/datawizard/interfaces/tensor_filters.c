/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static void _starpu_tensor_filter_block(int dim, void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
		   unsigned id, unsigned nparts, uintptr_t shadow_size)
{
	struct starpu_tensor_interface *tensor_father = (struct starpu_tensor_interface *) father_interface;
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
		nx = tensor_father->nx - 2 * shadow_size;
		ny = tensor_father->ny;
		nz = tensor_father->nz;
		nt = tensor_father->nt;
		nn = nx;
		blocksize = 1;
		break;

	case 2: /* vertical*/
		nx = tensor_father->nx;
		/* actual number of elements */
		ny = tensor_father->ny - 2 * shadow_size;
		nz = tensor_father->nz;
		nt = tensor_father->nt;
		nn = ny;
		blocksize = tensor_father->ldy;
		break;

	case 3: /* depth*/
		nx = tensor_father->nx;
		ny = tensor_father->ny;
		/* actual number of elements */
		nz = tensor_father->nz - 2 * shadow_size;
		nt = tensor_father->nt;
		nn = nz;
		blocksize = tensor_father->ldz;
		break;

	case 4: /* time*/
		nx = tensor_father->nx;
		ny = tensor_father->ny;
		nz = tensor_father->nz;
		/* actual number of elements */
		nt = tensor_father->nt - 2 * shadow_size;
		nn = nt;
		blocksize = tensor_father->ldt;
		break;
	default:
		STARPU_ASSERT_MSG(0, "Unknown value for dim");
	}

	size_t elemsize = tensor_father->elemsize;

	STARPU_ASSERT_MSG(nparts <= nn, "cannot split %u elements in %u parts", nn, nparts);

	uint32_t child_nn;
	size_t offset;
	starpu_filter_nparts_compute_chunk_size_and_offset(nn, nparts, elemsize, id, blocksize, &child_nn, &offset);

	child_nn += 2 * shadow_size;

	STARPU_ASSERT_MSG(tensor_father->id == STARPU_TENSOR_INTERFACE_ID, "%s can only be applied on a tensor data", __func__);
	tensor_child->id = tensor_father->id;

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

	if (tensor_father->dev_handle)
	{
		if (tensor_father->ptr)
			tensor_child->ptr = tensor_father->ptr + offset;
		tensor_child->ldy = tensor_father->ldy;
		tensor_child->ldz = tensor_father->ldz;
		tensor_child->ldt = tensor_father->ldt;
		tensor_child->dev_handle = tensor_father->dev_handle;
		tensor_child->offset = tensor_father->offset + offset;
	}
}

void starpu_tensor_filter_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				unsigned id, unsigned nparts)
{
	_starpu_tensor_filter_block(1, father_interface, child_interface, f, id, nparts, 0);
}

void starpu_tensor_filter_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				       unsigned id, unsigned nparts)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_tensor_filter_block(1, father_interface, child_interface, f, id, nparts, shadow_size);
}

void starpu_tensor_filter_vertical_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					 unsigned id, unsigned nparts)
{
	_starpu_tensor_filter_block(2, father_interface, child_interface, f, id, nparts, 0);
}

void starpu_tensor_filter_vertical_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
						unsigned id, unsigned nparts)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_tensor_filter_block(2, father_interface, child_interface, f, id, nparts, shadow_size);
}

void starpu_tensor_filter_depth_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				      unsigned id, unsigned nparts)
{
	_starpu_tensor_filter_block(3, father_interface, child_interface, f, id, nparts, 0);
}

void starpu_tensor_filter_depth_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					     unsigned id, unsigned nparts)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_tensor_filter_block(3, father_interface, child_interface, f, id, nparts, shadow_size);
}

void starpu_tensor_filter_time_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				     unsigned id, unsigned nparts)
{
	_starpu_tensor_filter_block(4, father_interface, child_interface, f, id, nparts, 0);
}

void starpu_tensor_filter_time_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					    unsigned id, unsigned nparts)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_tensor_filter_block(4, father_interface, child_interface, f, id, nparts, shadow_size);
}

static void _starpu_tensor_filter_pick_block(int dim, void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
					     unsigned id, unsigned nparts)
{
	struct starpu_tensor_interface *tensor_father = (struct starpu_tensor_interface *) father_interface;
	struct starpu_block_interface *block_child = (struct starpu_block_interface *) child_interface;

	unsigned blocksize;
	uint32_t nn;
	uint32_t nx = tensor_father->nx;
	uint32_t ny = tensor_father->ny;
	uint32_t nz = tensor_father->nz;
	uint32_t nt = tensor_father->nt;

	switch(dim)
	{
		/* along y-axis */
		case 1:
			nn = ny;
			blocksize = tensor_father->ldy;
			break;
		/* along z-axis */
		case 2:
			nn = nz;
			blocksize = tensor_father->ldz;
			break;
		/* along t-axis */
		case 3:
			nn = nt;
			blocksize = tensor_father->ldt;
			break;
		default:
			STARPU_ASSERT_MSG(0, "Unknown value for dim");
	}

	size_t elemsize = tensor_father->elemsize;

	size_t chunk_pos = (size_t)f->filter_arg_ptr;

	STARPU_ASSERT_MSG(nparts <= nn, "cannot get %u blocks", nparts);
	STARPU_ASSERT_MSG((chunk_pos + id) < nn, "the chosen block should be in the tensor");

	size_t offset = (chunk_pos + id) * blocksize * elemsize;

	STARPU_ASSERT_MSG(tensor_father->id == STARPU_TENSOR_INTERFACE_ID, "%s can only be applied on a tensor data", __func__);
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

	if (tensor_father->dev_handle)
	{
		if (tensor_father->ptr)
			block_child->ptr = tensor_father->ptr + offset;
		switch(dim)
		{
			/* along y-axis */
			case 1:
				block_child->ldy = tensor_father->ldz;
				block_child->ldz = tensor_father->ldt;
				break;
			/* along z-axis */
			case 2:
				block_child->ldy = tensor_father->ldy;
				block_child->ldz = tensor_father->ldt;
				break;
			/* along t-axis */
			case 3:
				block_child->ldy = tensor_father->ldy;
				block_child->ldz = tensor_father->ldz;
				break;
			default:
				STARPU_ASSERT_MSG(0, "Unknown value for dim");
		}
		block_child->dev_handle = tensor_father->dev_handle;
		block_child->offset = tensor_father->offset + offset;
	}
}
void starpu_tensor_filter_pick_block_t(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				       unsigned id, unsigned nparts)
{
	_starpu_tensor_filter_pick_block(3, father_interface, child_interface, f, id, nparts);
}

void starpu_tensor_filter_pick_block_z(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				       unsigned id, unsigned nparts)
{
	_starpu_tensor_filter_pick_block(2, father_interface, child_interface, f, id, nparts);
}

void starpu_tensor_filter_pick_block_y(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
				       unsigned id, unsigned nparts)
{
	_starpu_tensor_filter_pick_block(1, father_interface, child_interface, f, id, nparts);
}

struct starpu_data_interface_ops *starpu_tensor_filter_pick_block_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_block_ops;
}

void starpu_tensor_filter_pick_variable(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nchunks)
{
	struct starpu_tensor_interface *tensor_father = (struct starpu_tensor_interface *) father_interface;
	/* each chunk becomes a variable */
	struct starpu_variable_interface *variable_child = (struct starpu_variable_interface *) child_interface;

	uint32_t nx = tensor_father->nx;
	uint32_t ny = tensor_father->ny;
	uint32_t nz = tensor_father->nz;
	uint32_t nt = tensor_father->nt;

	unsigned ldy = tensor_father->ldy;
	unsigned ldz = tensor_father->ldz;
	unsigned ldt = tensor_father->ldt;

	size_t elemsize = tensor_father->elemsize;

	uint32_t* chunk_pos = (uint32_t*)f->filter_arg_ptr;
	// int i;
	// for(i=0; i<4; i++)
	// {
	// 	printf("pos is %d\n", chunk_pos[i]);
	// }

	STARPU_ASSERT_MSG((chunk_pos[0] < nx)&&(chunk_pos[1] < ny)&&(chunk_pos[2] < nz)&&(chunk_pos[3] < nt), "the chosen variable should be in the tensor");

	size_t offset = (chunk_pos[3] * ldt + chunk_pos[2] * ldz + chunk_pos[1] * ldy + chunk_pos[0]) * elemsize;

	STARPU_ASSERT_MSG(tensor_father->id == STARPU_TENSOR_INTERFACE_ID, "%s can only be applied on a tensor data", __func__);

	/* update the child's interface */
	variable_child->id = STARPU_VARIABLE_INTERFACE_ID;
	variable_child->elemsize = elemsize;

	/* is the information on this node valid ? */
	if (tensor_father->dev_handle)
	{
		if (tensor_father->ptr)
			variable_child->ptr = tensor_father->ptr + offset;
		variable_child->dev_handle = tensor_father->dev_handle;
		variable_child->offset = tensor_father->offset + offset;
	}
}

struct starpu_data_interface_ops *starpu_tensor_filter_pick_variable_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_variable_ops;
}
