/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void starpu_csr_filter_vertical_block(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	struct starpu_csr_interface *csr_parent = (struct starpu_csr_interface *) parent_interface;
	struct starpu_csr_interface *csr_child = (struct starpu_csr_interface *) child_interface;

	uint32_t nrow = csr_parent->nrow;
	size_t elemsize = csr_parent->elemsize;
	uint32_t firstentry = csr_parent->firstentry;

	uint32_t *ram_rowptr = csr_parent->ram_rowptr;

	size_t first_index;
	size_t child_nrow;

	starpu_filter_nparts_compute_chunk_size_and_offset(nrow, nchunks, 1, id, 1, &child_nrow, &first_index);

	uint32_t local_firstentry = ram_rowptr[first_index] - firstentry;
	uint32_t local_lastentry = ram_rowptr[first_index + child_nrow] - firstentry;

	uint32_t local_nnz = local_lastentry - local_firstentry;

	STARPU_ASSERT_MSG(csr_parent->id == STARPU_CSR_INTERFACE_ID, "%s can only be applied on a csr data", __func__);
	csr_child->id = csr_parent->id;
	csr_child->nnz = local_nnz;
	csr_child->nrow = child_nrow;
	csr_child->firstentry = local_firstentry;
	csr_child->elemsize = elemsize;
	csr_child->ram_colind = &csr_parent->ram_colind[local_firstentry];
	csr_child->ram_rowptr = &ram_rowptr[first_index];

	if (csr_parent->nzval)
	{
		csr_child->rowptr = &csr_parent->rowptr[first_index];
		csr_child->colind = &csr_parent->colind[local_firstentry];
		csr_child->nzval = csr_parent->nzval + local_firstentry * elemsize;
	}
}
