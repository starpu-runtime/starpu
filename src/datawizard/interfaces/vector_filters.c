/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <common/config.h>
#include <datawizard/hierarchy.h>

void starpu_block_filter_func_vector(starpu_filter *f, starpu_data_handle root_handle)
{
	unsigned nchunks;
	uint32_t arg = f->filter_arg;

	starpu_vector_interface_t *vector_root =
		starpu_data_get_interface_on_node(root_handle, 0);

	uint32_t nx = vector_root->nx;
	size_t elemsize = vector_root->elemsize;

	/* we will have arg chunks */
	nchunks = STARPU_MIN(nx, arg);

	/* first allocate the children data_state */
	starpu_data_create_children(root_handle, nchunks, root_handle->ops);

	/* actually create all the chunks */
	unsigned chunk;
	for (chunk = 0; chunk < nchunks; chunk++)
	{
		uint32_t chunk_size = (nx + nchunks - 1)/nchunks;
		size_t offset = chunk*chunk_size*elemsize;

		uint32_t child_nx = 
			STARPU_MIN(chunk_size, nx - chunk*chunk_size);

		starpu_data_handle chunk_handle =
			starpu_data_get_child(root_handle, chunk);

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			starpu_vector_interface_t *local =
				starpu_data_get_interface_on_node(chunk_handle, node);

			local->nx = child_nx;
			local->elemsize = elemsize;

			if (root_handle->per_node[node].allocated) {
				starpu_vector_interface_t *local_root =
					starpu_data_get_interface_on_node(root_handle, node);

				local->ptr = local_root->ptr + offset;
			}
		}
	}
}


void starpu_divide_in_2_filter_func_vector(starpu_filter *f, starpu_data_handle root_handle)
{
	uint32_t length_first = f->filter_arg;

	starpu_vector_interface_t *vector_root =
		starpu_data_get_interface_on_node(root_handle, 0);

	uint32_t nx = vector_root->nx;
	size_t elemsize = vector_root->elemsize;

	/* first allocate the children data_state */
	starpu_data_create_children(root_handle, 2, root_handle->ops);

	STARPU_ASSERT(length_first < nx);

	starpu_data_handle chunk0_handle =
		starpu_data_get_child(root_handle, 0);

	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		starpu_vector_interface_t *local =
			starpu_data_get_interface_on_node(chunk0_handle, node);

		local->nx = length_first;
		local->elemsize = elemsize;

		if (root_handle->per_node[node].allocated) {
			starpu_vector_interface_t *local_root =
				starpu_data_get_interface_on_node(root_handle, node);

			local->ptr = local_root->ptr;
		}
	}

	starpu_data_handle chunk1_handle =
		starpu_data_get_child(root_handle, 1);

	for (node = 0; node < MAXNODES; node++)
	{
		starpu_vector_interface_t *local =
			starpu_data_get_interface_on_node(chunk1_handle, node);

		local->nx = nx - length_first;
		local->elemsize = elemsize;

		if (root_handle->per_node[node].allocated) {
			starpu_vector_interface_t *local_root =
				starpu_data_get_interface_on_node(root_handle, node);

			local->ptr = local_root->ptr + length_first*elemsize;
		}
	}
}

void starpu_list_filter_func_vector(starpu_filter *f, starpu_data_handle root_handle)
{
	uint32_t nchunks = f->filter_arg;
	uint32_t *length_tab = f->filter_arg_ptr;

	starpu_vector_interface_t *vector_root =
		starpu_data_get_interface_on_node(root_handle, 0);

	uint32_t nx = vector_root->nx;
	size_t elemsize = vector_root->elemsize;

	/* first allocate the children data_state */
	starpu_data_create_children(root_handle, nchunks, root_handle->ops);

	unsigned current_pos = 0;

	/* actually create all the chunks */
	unsigned chunk;
	for (chunk = 0; chunk < nchunks; chunk++)
	{
		starpu_data_handle chunk_handle =
			starpu_data_get_child(root_handle, chunk);

		uint32_t chunk_size = length_tab[chunk];

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			starpu_vector_interface_t *local =
				starpu_data_get_interface_on_node(chunk_handle, node);

			local->nx = chunk_size;
			local->elemsize = elemsize;

			if (root_handle->per_node[node].allocated) {
				starpu_vector_interface_t *local_root =
					starpu_data_get_interface_on_node(root_handle, node);

				local->ptr = local_root->ptr + current_pos*elemsize;
			}
		}

		current_pos += chunk_size;
	}

	STARPU_ASSERT(current_pos == nx);
}
