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

#include "csr_filters.h"
#include "csr_interface.h"

unsigned starpu_vertical_block_filter_func_csr(starpu_filter *f, data_state *root_data)
{
	unsigned nchunks;
	uint32_t arg = f->filter_arg;

	uint32_t nrow = root_data->interface[0].csr.nrow;
	size_t elemsize = root_data->interface[0].csr.elemsize;
	uint32_t firstentry = root_data->interface[0].csr.firstentry;

	/* we will have arg chunks */
	nchunks = STARPU_MIN(nrow, arg);
	
	/* first allocate the children data_state */
	root_data->children = calloc(nchunks, sizeof(data_state));
	STARPU_ASSERT(root_data->children);

	/* actually create all the chunks */
	uint32_t chunk_size = (nrow + nchunks - 1)/nchunks;

	/* XXX */
	STARPU_ASSERT(root_data->per_node[0].allocated);
	uint32_t *rowptr = root_data->interface[0].csr.rowptr;

	unsigned chunk;
	for (chunk = 0; chunk < nchunks; chunk++)
	{
		uint32_t first_index = chunk*chunk_size - firstentry;
		uint32_t local_firstentry = rowptr[first_index];

		uint32_t child_nrow = 
			STARPU_MIN(chunk_size, nrow - chunk*chunk_size);

		uint32_t local_nnz = rowptr[first_index + child_nrow] - rowptr[first_index]; 

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			starpu_csr_interface_t *local = &root_data->children[chunk].interface[node].csr;

			local->nnz = local_nnz;
			local->nrow = child_nrow;
			local->firstentry = local_firstentry;
			local->elemsize = elemsize;

			if (root_data->per_node[node].allocated) {
				local->rowptr = &root_data->interface[node].csr.rowptr[first_index];
				local->colind = &root_data->interface[node].csr.colind[local_firstentry];
				float *nzval = (float *)(root_data->interface[node].csr.nzval);
				local->nzval = (uintptr_t)&nzval[local_firstentry];
			}
		}
	}

	return nchunks;
}
