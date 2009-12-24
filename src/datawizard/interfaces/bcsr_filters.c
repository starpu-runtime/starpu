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

extern struct data_interface_ops_t interface_blas_ops;

unsigned starpu_canonical_block_filter_bcsr(starpu_filter *f __attribute__((unused)), starpu_data_handle root_handle)
{
	unsigned nchunks;

	struct starpu_bcsr_interface_s *interface =
		starpu_data_get_interface_on_node(root_handle, 0);

	uint32_t nnz = interface->nnz;

	size_t elemsize = interface->elemsize;
	uint32_t firstentry = interface->firstentry;

	/* size of the tiles */
	uint32_t r = interface->r;
	uint32_t c = interface->c;

	/* we create as many subdata as there are blocks ... */
	nchunks = nnz;
	
	/* first allocate the children data_state */
	starpu_data_create_children(root_handle, nchunks, sizeof(starpu_blas_interface_t));

	/* actually create all the chunks */

	/* XXX */
	STARPU_ASSERT(root_handle->per_node[0].allocated);

	/* each chunk becomes a small dense matrix */
	unsigned chunk;
	for (chunk = 0; chunk < nchunks; chunk++)
	{
		starpu_data_handle sub_handle = starpu_data_get_child(root_handle, chunk);
		uint32_t ptr_offset = c*r*chunk*elemsize;

		unsigned node;
		for (node = 0; node < MAXNODES; node++)
		{
			starpu_blas_interface_t *local =
				starpu_data_get_interface_on_node(root_handle, node);

			local->nx = c;
			local->ny = r;
			local->ld = c;
			local->elemsize = elemsize;

			if (root_handle->per_node[node].allocated) {
				struct starpu_bcsr_interface_s *node_interface =
					starpu_data_get_interface_on_node(root_handle, node);
				uint8_t *nzval = (uint8_t *)(node_interface->nzval);
				local->ptr = (uintptr_t)&nzval[firstentry + ptr_offset];
			}
		}

		sub_handle->ops = &interface_blas_ops;
	}

	return nchunks;

}
