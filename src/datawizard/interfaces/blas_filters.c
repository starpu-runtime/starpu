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

/*
 * an example of a dummy partition function : blocks ...
 */
unsigned starpu_block_filter_func(starpu_filter *f, data_state *root_data)
{
	unsigned nchunks;
	uint32_t arg = f->filter_arg;

	starpu_blas_interface_t *blas_root =
		starpu_data_get_interface_on_node(root_data, 0);

	uint32_t nx = blas_root->nx;
	uint32_t ny = blas_root->ny;
	size_t elemsize = blas_root->elemsize;

	/* we will have arg chunks */
	nchunks = STARPU_MIN(nx, arg);

	/* first allocate the children data_state */
	starpu_data_create_children(root_data, nchunks);

	/* actually create all the chunks */
	unsigned chunk;
	for (chunk = 0; chunk < nchunks; chunk++)
	{
		size_t chunk_size = ((size_t)nx + nchunks - 1)/nchunks;
		size_t offset = (size_t)chunk*chunk_size*elemsize;

		uint32_t child_nx = 
			STARPU_MIN(chunk_size, (size_t)nx - (size_t)chunk*chunk_size);

		starpu_data_handle chunk_handle =
			starpu_data_get_child(root_data, chunk);

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			starpu_blas_interface_t *local = 
				starpu_data_get_interface_on_node(chunk_handle, node);

			local->nx = child_nx;
			local->ny = ny;
			local->elemsize = elemsize;

			if (root_data->per_node[node].allocated) {
				starpu_blas_interface_t *local_root =
					starpu_data_get_interface_on_node(root_data, node);

				local->ptr = local_root->ptr + offset;
				local->ld = local_root->ld;
			}
		}
	}

	return nchunks;
}

unsigned starpu_vertical_block_filter_func(starpu_filter *f, data_state *root_data)
{
	unsigned nchunks;
	uint32_t arg = f->filter_arg;

	uint32_t nx = root_data->interface[0].blas.nx;
	uint32_t ny = root_data->interface[0].blas.ny;
	size_t elemsize = root_data->interface[0].blas.elemsize;

	/* we will have arg chunks */
	nchunks = STARPU_MIN(ny, arg);
	
	/* first allocate the children data_state */
	root_data->children = calloc(nchunks, sizeof(data_state));
	STARPU_ASSERT(root_data->children);

	/* actually create all the chunks */
	unsigned chunk;
	for (chunk = 0; chunk < nchunks; chunk++)
	{
		size_t chunk_size = ((size_t)ny + nchunks - 1)/nchunks;

		size_t child_ny = 
			STARPU_MIN(chunk_size, (size_t)ny - (size_t)chunk*chunk_size);

		starpu_data_handle chunk_handle =
			starpu_data_get_child(root_data, chunk);

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			starpu_blas_interface_t *local =
				starpu_data_get_interface_on_node(chunk_handle, node);

			local->nx = nx;
			local->ny = child_ny;
			local->elemsize = elemsize;

			if (root_data->per_node[node].allocated) {
				starpu_blas_interface_t *local_root =
					starpu_data_get_interface_on_node(root_data, node);

				size_t offset = 
					(size_t)chunk*chunk_size*local_root->ld*elemsize;
				local->ptr = local_root->ptr + offset;
				local->ld = local_root->ld;
			}
		}
	}

	return nchunks;
}
