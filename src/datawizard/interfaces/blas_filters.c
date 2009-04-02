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

#include "blas_filters.h"
#include "blas_interface.h"

/*
 * an example of a dummy partition function : blocks ...
 */
unsigned starpu_block_filter_func(starpu_filter *f, data_state *root_data)
{
	unsigned nchunks;
	uint32_t arg = f->filter_arg;

	starpu_blas_interface_t *blas_root = &root_data->interface[0].blas;
	uint32_t nx = blas_root->nx;
	uint32_t ny = blas_root->ny;
	size_t elemsize = blas_root->elemsize;

	/* we will have arg chunks */
	nchunks = STARPU_MIN(nx, arg);

	/* first allocate the children data_state */
	root_data->children = calloc(nchunks, sizeof(data_state));
	STARPU_ASSERT(root_data->children);

	/* actually create all the chunks */
	unsigned chunk;
	for (chunk = 0; chunk < nchunks; chunk++)
	{
		uint32_t chunk_size = (nx + nchunks - 1)/nchunks;
		size_t offset = chunk*chunk_size*elemsize;

		uint32_t child_nx = 
			STARPU_MIN(chunk_size, nx - chunk*chunk_size);

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			starpu_blas_interface_t *local = &root_data->children[chunk].interface[node].blas;

			local->nx = child_nx;
			local->ny = ny;
			local->elemsize = elemsize;

			if (root_data->per_node[node].allocated) {
				local->ptr = root_data->interface[node].blas.ptr + offset;
				local->ld = root_data->interface[node].blas.ld;
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
		uint32_t chunk_size = (ny + nchunks - 1)/nchunks;

		uint32_t child_ny = 
			STARPU_MIN(chunk_size, ny - chunk*chunk_size);

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			starpu_blas_interface_t *local = &root_data->children[chunk].interface[node].blas;

			local->nx = nx;
			local->ny = child_ny;
			local->elemsize = elemsize;

			if (root_data->per_node[node].allocated) {
				size_t offset = 
					chunk*chunk_size*root_data->interface[node].blas.ld*elemsize;
				local->ptr = root_data->interface[node].blas.ptr + offset;
				local->ld = root_data->interface[node].blas.ld;
			}
		}
	}

	return nchunks;
}
