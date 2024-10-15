/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static void _starpu_vector_filter_block(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks, uintptr_t shadow_size)
{
	struct starpu_vector_interface *vector_parent = (struct starpu_vector_interface *) parent_interface;
	struct starpu_vector_interface *vector_child = (struct starpu_vector_interface *) child_interface;

	/* actual number of elements */
	size_t nx = vector_parent->nx - 2 * shadow_size;
	size_t elemsize = vector_parent->elemsize;

	STARPU_ASSERT_MSG(nchunks <= nx, "cannot split %zu elements in %u parts", nx, nchunks);

	size_t child_nx;
	size_t offset;
	starpu_filter_nparts_compute_chunk_size_and_offset(nx, nchunks, elemsize, id, 1, &child_nx, &offset);
	child_nx += 2*shadow_size;

	STARPU_ASSERT_MSG(vector_parent->id == STARPU_VECTOR_INTERFACE_ID, "%s can only be applied on a vector data", __func__);
	vector_child->id = vector_parent->id;
	vector_child->nx = child_nx;
	vector_child->elemsize = elemsize;
	vector_child->allocsize = vector_child->nx * elemsize;

	if (vector_parent->dev_handle)
	{
		if (vector_parent->ptr)
			vector_child->ptr = vector_parent->ptr + offset;
		vector_child->dev_handle = vector_parent->dev_handle;
		vector_child->offset = vector_parent->offset + offset;
	}
}

void starpu_vector_filter_block(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	_starpu_vector_filter_block(parent_interface, child_interface, f, id, nchunks, 0);
}


void starpu_vector_filter_block_shadow(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	uintptr_t shadow_size = (uintptr_t) f->filter_arg_ptr;

	_starpu_vector_filter_block(parent_interface, child_interface, f, id, nchunks, shadow_size);
}


void starpu_vector_filter_divide_in_2(void *parent_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nchunks)
{
	/* there cannot be more than 2 chunks */
	STARPU_ASSERT_MSG(id < 2, "Only %u parts", id);

	struct starpu_vector_interface *vector_parent = (struct starpu_vector_interface *) parent_interface;
	struct starpu_vector_interface *vector_child = (struct starpu_vector_interface *) child_interface;

	size_t length_first = f->filter_arg;

	size_t nx = vector_parent->nx;
	size_t elemsize = vector_parent->elemsize;

	STARPU_ASSERT_MSG(length_first < nx, "First part is too long: %zu vs %zu", length_first, nx);

	STARPU_ASSERT_MSG(vector_parent->id == STARPU_VECTOR_INTERFACE_ID, "%s can only be applied on a vector data", __func__);
	vector_child->id = vector_parent->id;

	/* this is the first child */
	if (id == 0)
	{
		vector_child->nx = length_first;
		vector_child->elemsize = elemsize;
		vector_child->allocsize = vector_child->nx * elemsize;

		if (vector_parent->dev_handle)
		{
			if (vector_parent->ptr)
				vector_child->ptr = vector_parent->ptr;
			vector_child->offset = vector_parent->offset;
			vector_child->dev_handle = vector_parent->dev_handle;
		}
	}
	else /* the second child */
	{
		vector_child->nx = nx - length_first;
		vector_child->elemsize = elemsize;
		vector_child->allocsize = vector_child->nx * elemsize;

		if (vector_parent->dev_handle)
		{
			if (vector_parent->ptr)
				vector_child->ptr = vector_parent->ptr + length_first*elemsize;
			vector_child->offset = vector_parent->offset + length_first*elemsize;
			vector_child->dev_handle = vector_parent->dev_handle;
		}
	}
}


void starpu_vector_filter_list_long(void *parent_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nchunks)
{
	struct starpu_vector_interface *vector_parent = (struct starpu_vector_interface *) parent_interface;
	struct starpu_vector_interface *vector_child = (struct starpu_vector_interface *) child_interface;

	long *length_tab = (long *) f->filter_arg_ptr;

	size_t elemsize = vector_parent->elemsize;

	long chunk_size = length_tab[id];

	STARPU_ASSERT_MSG(vector_parent->id == STARPU_VECTOR_INTERFACE_ID, "%s can only be applied on a vector data", __func__);
	vector_child->id = vector_parent->id;
	vector_child->nx = chunk_size;
	vector_child->elemsize = elemsize;
	vector_child->allocsize = vector_child->nx * elemsize;

	if (vector_parent->dev_handle)
	{
		/* compute the current position */
		unsigned current_pos = 0;
		unsigned i;
		for (i = 0; i < id; i++)
			current_pos += length_tab[i];

		if (vector_parent->ptr)
			vector_child->ptr = vector_parent->ptr + current_pos*elemsize;
		vector_child->offset = vector_parent->offset + current_pos*elemsize;
		vector_child->dev_handle = vector_parent->dev_handle;
	}
}

void starpu_vector_filter_list(void *parent_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nchunks)
{
	struct starpu_vector_interface *vector_parent = (struct starpu_vector_interface *) parent_interface;
	struct starpu_vector_interface *vector_child = (struct starpu_vector_interface *) child_interface;

	uint32_t *length_tab = (uint32_t *) f->filter_arg_ptr;

	size_t elemsize = vector_parent->elemsize;

	uint32_t chunk_size = length_tab[id];

	STARPU_ASSERT_MSG(vector_parent->id == STARPU_VECTOR_INTERFACE_ID, "%s can only be applied on a vector data", __func__);
	vector_child->id = vector_parent->id;
	vector_child->nx = chunk_size;
	vector_child->elemsize = elemsize;
	vector_child->allocsize = vector_child->nx * elemsize;

	if (vector_parent->dev_handle)
	{
		/* compute the current position */
		unsigned current_pos = 0;
		unsigned i;
		for (i = 0; i < id; i++)
			current_pos += length_tab[i];

		if (vector_parent->ptr)
			vector_child->ptr = vector_parent->ptr + current_pos*elemsize;
		vector_child->offset = vector_parent->offset + current_pos*elemsize;
		vector_child->dev_handle = vector_parent->dev_handle;
	}
}

void starpu_vector_filter_pick_variable(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	struct starpu_vector_interface *vector_parent = (struct starpu_vector_interface *) parent_interface;
	/* each chunk becomes a variable */
	struct starpu_variable_interface *variable_child = (struct starpu_variable_interface *) child_interface;

	/* actual number of elements */
	size_t nx = vector_parent->nx;
	size_t elemsize = vector_parent->elemsize;

	size_t chunk_pos = (size_t)f->filter_arg_ptr;

	STARPU_ASSERT_MSG(nchunks <= nx, "cannot get %u variables", nchunks);
	STARPU_ASSERT_MSG((chunk_pos + id) < nx, "the chosen variable should be in the vector");

	size_t offset = (chunk_pos + id) * elemsize;

	STARPU_ASSERT_MSG(vector_parent->id == STARPU_VECTOR_INTERFACE_ID, "%s can only be applied on a vector data", __func__);

	variable_child->id = STARPU_VARIABLE_INTERFACE_ID;
	variable_child->elemsize = elemsize;

	if (vector_parent->dev_handle)
	{
		if (vector_parent->ptr)
			variable_child->ptr = vector_parent->ptr + offset;
		variable_child->dev_handle = vector_parent->dev_handle;
		variable_child->offset = vector_parent->offset + offset;
	}
}

struct starpu_data_interface_ops *starpu_vector_filter_pick_variable_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_variable_ops;
}
