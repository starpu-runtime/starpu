/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

void starpu_block_filter_func_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                                    unsigned id, unsigned nparts)
{
        struct starpu_block_interface *block_father = (struct starpu_block_interface *) father_interface;
        struct starpu_block_interface *block_child = (struct starpu_block_interface *) child_interface;

	uint32_t nx = block_father->nx;
        uint32_t ny = block_father->ny;
        uint32_t nz = block_father->nz;
	size_t elemsize = block_father->elemsize;

	STARPU_ASSERT(nparts <= nx);

	uint32_t chunk_size = (nx + nparts - 1)/nparts;
	size_t offset = id*chunk_size*elemsize;

        uint32_t child_nx = STARPU_MIN(chunk_size, nx - id*chunk_size);

	block_child->nx = child_nx;
	block_child->ny = ny;
	block_child->nz = nz;
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

void starpu_block_shadow_filter_func_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                                    unsigned id, unsigned nparts)
{
        struct starpu_block_interface *block_father = (struct starpu_block_interface *) father_interface;
        struct starpu_block_interface *block_child = (struct starpu_block_interface *) child_interface;

        uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	/* actual number of elements */
	uint32_t nx = block_father->nx - 2 * shadow_size;
        uint32_t ny = block_father->ny;
        uint32_t nz = block_father->nz;
	size_t elemsize = block_father->elemsize;

	STARPU_ASSERT(nparts <= nx);

	uint32_t chunk_size = (nx + nparts - 1)/nparts;
	size_t offset = id*chunk_size*elemsize;

        uint32_t child_nx = STARPU_MIN(chunk_size, nx - id*chunk_size) + 2 * shadow_size;

	block_child->nx = child_nx;
	block_child->ny = ny;
	block_child->nz = nz;
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

void starpu_vertical_block_filter_func_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                                    unsigned id, unsigned nparts)
{
        struct starpu_block_interface *block_father = (struct starpu_block_interface *) father_interface;
        struct starpu_block_interface *block_child = (struct starpu_block_interface *) child_interface;

	uint32_t nx = block_father->nx;
        uint32_t ny = block_father->ny;
        uint32_t nz = block_father->nz;
	size_t elemsize = block_father->elemsize;

	STARPU_ASSERT(nparts <= ny);

	uint32_t chunk_size = (ny + nparts - 1)/nparts;
	size_t offset = id*chunk_size*block_father->ldy*elemsize;

        uint32_t child_ny = STARPU_MIN(chunk_size, ny - id*chunk_size);

	block_child->nx = nx;
	block_child->ny = child_ny;
	block_child->nz = nz;
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

void starpu_vertical_block_shadow_filter_func_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                                    unsigned id, unsigned nparts)
{
        struct starpu_block_interface *block_father = (struct starpu_block_interface *) father_interface;
        struct starpu_block_interface *block_child = (struct starpu_block_interface *) child_interface;

        uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	uint32_t nx = block_father->nx;
	/* actual number of elements */
        uint32_t ny = block_father->ny - 2 * shadow_size;
        uint32_t nz = block_father->nz;
	size_t elemsize = block_father->elemsize;

	STARPU_ASSERT(nparts <= ny);

	uint32_t chunk_size = (ny + nparts - 1)/nparts;
	size_t offset = id*chunk_size*block_father->ldy*elemsize;

        uint32_t child_ny = STARPU_MIN(chunk_size, ny - id*chunk_size) + 2 * shadow_size;

	block_child->nx = nx;
	block_child->ny = child_ny;
	block_child->nz = nz;
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

void starpu_depth_block_filter_func_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                                    unsigned id, unsigned nparts)
{
        struct starpu_block_interface *block_father = (struct starpu_block_interface *) father_interface;
        struct starpu_block_interface *block_child = (struct starpu_block_interface *) child_interface;

	uint32_t nx = block_father->nx;
        uint32_t ny = block_father->ny;
        uint32_t nz = block_father->nz;
	size_t elemsize = block_father->elemsize;

	STARPU_ASSERT(nparts <= nz);

	uint32_t chunk_size = (nz + nparts - 1)/nparts;
	size_t offset = id*chunk_size*block_father->ldz*elemsize;

        uint32_t child_nz = STARPU_MIN(chunk_size, nz - id*chunk_size);

	block_child->nx = nx;
	block_child->ny = ny;
	block_child->nz = child_nz;
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

void starpu_depth_block_shadow_filter_func_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f,
                                    unsigned id, unsigned nparts)
{
        struct starpu_block_interface *block_father = (struct starpu_block_interface *) father_interface;
        struct starpu_block_interface *block_child = (struct starpu_block_interface *) child_interface;

        uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	uint32_t nx = block_father->nx;
        uint32_t ny = block_father->ny;
	/* actual number of elements */
        uint32_t nz = block_father->nz - 2 * shadow_size;
	size_t elemsize = block_father->elemsize;

	STARPU_ASSERT(nparts <= nz);

	uint32_t chunk_size = (nz + nparts - 1)/nparts;
	size_t offset = id*chunk_size*block_father->ldz*elemsize;

        uint32_t child_nz = STARPU_MIN(chunk_size, nz - id*chunk_size) + 2 * shadow_size;

	block_child->nx = nx;
	block_child->ny = ny;
	block_child->nz = child_nz;
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
