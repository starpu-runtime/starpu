/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void starpu_vector_filter_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
        struct starpu_vector_interface *vector_father = (struct starpu_vector_interface *) father_interface;
        struct starpu_vector_interface *vector_child = (struct starpu_vector_interface *) child_interface;

	uint32_t nx = vector_father->nx;
	size_t elemsize = vector_father->elemsize;

	STARPU_ASSERT_MSG(nchunks <= nx, "cannot split %u elements in %u parts", nx, nchunks);

	uint32_t child_nx;
	size_t offset;
	starpu_filter_nparts_compute_chunk_size_and_offset(nx, nchunks, elemsize, id, 1, &child_nx, &offset);

	STARPU_ASSERT_MSG(vector_father->id == STARPU_VECTOR_INTERFACE_ID, "%s can only be applied on a vector data", __func__);
	vector_child->id = vector_father->id;
	vector_child->nx = child_nx;
	vector_child->elemsize = elemsize;
	STARPU_ASSERT_MSG(vector_father->allocsize == vector_father->nx * vector_father->elemsize, "partitioning vector with non-trival allocsize not supported yet, patch welcome");
	vector_child->allocsize = vector_child->nx * elemsize;

	if (vector_father->dev_handle)
	{
		if (vector_father->ptr)
			vector_child->ptr = vector_father->ptr + offset;
		vector_child->dev_handle = vector_father->dev_handle;
		vector_child->offset = vector_father->offset + offset;
	}
}


void starpu_vector_filter_block_shadow(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
        struct starpu_vector_interface *vector_father = (struct starpu_vector_interface *) father_interface;
        struct starpu_vector_interface *vector_child = (struct starpu_vector_interface *) child_interface;

        uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	/* actual number of elements */
	uint32_t nx = vector_father->nx - 2 * shadow_size;
	size_t elemsize = vector_father->elemsize;

	STARPU_ASSERT_MSG(nchunks <= nx, "cannot split %u elements in %u parts", nx, nchunks);

	uint32_t child_nx;
	size_t offset;
	starpu_filter_nparts_compute_chunk_size_and_offset(nx, nchunks, elemsize, id, 1, &child_nx, &offset);
	child_nx += 2*shadow_size;

	STARPU_ASSERT_MSG(vector_father->id == STARPU_VECTOR_INTERFACE_ID, "%s can only be applied on a vector data", __func__);
	vector_child->id = vector_father->id;
	vector_child->nx = child_nx;
	vector_child->elemsize = elemsize;
	STARPU_ASSERT_MSG(vector_father->allocsize == vector_father->nx * vector_father->elemsize, "partitioning vector with non-trival allocsize not supported yet, patch welcome");
	vector_child->allocsize = vector_child->nx * elemsize;

	if (vector_father->dev_handle)
	{
		if (vector_father->ptr)
			vector_child->ptr = vector_father->ptr + offset;
		vector_child->dev_handle = vector_father->dev_handle;
		vector_child->offset = vector_father->offset + offset;
	}
}


void starpu_vector_filter_divide_in_2(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nchunks)
{
        /* there cannot be more than 2 chunks */
	STARPU_ASSERT_MSG(id < 2, "Only %u parts", id);

	struct starpu_vector_interface *vector_father = (struct starpu_vector_interface *) father_interface;
	struct starpu_vector_interface *vector_child = (struct starpu_vector_interface *) child_interface;

	uint32_t length_first = f->filter_arg;

	uint32_t nx = vector_father->nx;
	size_t elemsize = vector_father->elemsize;

	STARPU_ASSERT_MSG(length_first < nx, "First part is too long: %u vs %u", length_first, nx);

	STARPU_ASSERT_MSG(vector_father->id == STARPU_VECTOR_INTERFACE_ID, "%s can only be applied on a vector data", __func__);
	vector_child->id = vector_father->id;
	STARPU_ASSERT_MSG(vector_father->allocsize == vector_father->nx * vector_father->elemsize, "partitioning vector with non-trival allocsize not supported yet, patch welcome");

	/* this is the first child */
	if (id == 0)
	{
		vector_child->nx = length_first;
		vector_child->elemsize = elemsize;
		vector_child->allocsize = vector_child->nx * elemsize;

		if (vector_father->dev_handle)
		{
			if (vector_father->ptr)
				vector_child->ptr = vector_father->ptr;
			vector_child->offset = vector_father->offset;
			vector_child->dev_handle = vector_father->dev_handle;
		}
	}
	else /* the second child */
	{
		vector_child->nx = nx - length_first;
		vector_child->elemsize = elemsize;
		vector_child->allocsize = vector_child->nx * elemsize;

		if (vector_father->dev_handle)
		{
			if (vector_father->ptr)
				vector_child->ptr = vector_father->ptr + length_first*elemsize;
			vector_child->offset = vector_father->offset + length_first*elemsize;
			vector_child->dev_handle = vector_father->dev_handle;
		}
	}
}


void starpu_vector_filter_list_long(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nchunks)
{
        struct starpu_vector_interface *vector_father = (struct starpu_vector_interface *) father_interface;
        struct starpu_vector_interface *vector_child = (struct starpu_vector_interface *) child_interface;

        long *length_tab = (long *) f->filter_arg_ptr;

	size_t elemsize = vector_father->elemsize;

	long chunk_size = length_tab[id];

	STARPU_ASSERT_MSG(vector_father->id == STARPU_VECTOR_INTERFACE_ID, "%s can only be applied on a vector data", __func__);
	vector_child->id = vector_father->id;
	vector_child->nx = chunk_size;
	vector_child->elemsize = elemsize;
	STARPU_ASSERT_MSG(vector_father->allocsize == vector_father->nx * vector_father->elemsize, "partitioning vector with non-trival allocsize not supported yet, patch welcomed");
	vector_child->allocsize = vector_child->nx * elemsize;

	if (vector_father->dev_handle)
	{
		/* compute the current position */
		unsigned current_pos = 0;
		unsigned i;
		for (i = 0; i < id; i++)
			current_pos += length_tab[i];

		if (vector_father->ptr)
			vector_child->ptr = vector_father->ptr + current_pos*elemsize;
		vector_child->offset = vector_father->offset + current_pos*elemsize;
		vector_child->dev_handle = vector_father->dev_handle;
	}
}

void starpu_vector_filter_list(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nchunks)
{
        struct starpu_vector_interface *vector_father = (struct starpu_vector_interface *) father_interface;
        struct starpu_vector_interface *vector_child = (struct starpu_vector_interface *) child_interface;

        uint32_t *length_tab = (uint32_t *) f->filter_arg_ptr;

	size_t elemsize = vector_father->elemsize;

	uint32_t chunk_size = length_tab[id];

	STARPU_ASSERT_MSG(vector_father->id == STARPU_VECTOR_INTERFACE_ID, "%s can only be applied on a vector data", __func__);
	vector_child->id = vector_father->id;
	vector_child->nx = chunk_size;
	vector_child->elemsize = elemsize;
	STARPU_ASSERT_MSG(vector_father->allocsize == vector_father->nx * vector_father->elemsize, "partitioning vector with non-trival allocsize not supported yet, patch welcomed");
	vector_child->allocsize = vector_child->nx * elemsize;

	if (vector_father->dev_handle)
	{
		/* compute the current position */
		unsigned current_pos = 0;
		unsigned i;
		for (i = 0; i < id; i++)
			current_pos += length_tab[i];

		if (vector_father->ptr)
			vector_child->ptr = vector_father->ptr + current_pos*elemsize;
		vector_child->offset = vector_father->offset + current_pos*elemsize;
		vector_child->dev_handle = vector_father->dev_handle;
	}
}
