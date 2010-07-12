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
#include <datawizard/filters.h>

void starpu_vertical_block_filter_func_csr(void *father_interface, void *child_interface, __attribute__((unused)) struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	starpu_csr_interface_t *csr_father = father_interface;
	starpu_csr_interface_t *csr_child = child_interface;

	uint32_t nrow = csr_father->nrow;
	size_t elemsize = csr_father->elemsize;
	uint32_t firstentry = csr_father->firstentry;

	uint32_t chunk_size = (nrow + nchunks - 1)/nchunks;

	uint32_t *rowptr = csr_father->rowptr;

	uint32_t first_index = id*chunk_size - firstentry;
	uint32_t local_firstentry = rowptr[first_index];
	
	uint32_t child_nrow = 
	  STARPU_MIN(chunk_size, nrow - id*chunk_size);
	
	uint32_t local_nnz = rowptr[first_index + child_nrow] - rowptr[first_index]; 
	
	csr_child->nnz = local_nnz;
	csr_child->nrow = child_nrow;
	csr_child->firstentry = local_firstentry;
	csr_child->elemsize = elemsize;
	
	if (csr_father->nzval) {
	  csr_child->rowptr = &csr_father->rowptr[first_index];
	  csr_child->colind = &csr_father->colind[local_firstentry];
	  csr_child->nzval = csr_father->nzval + local_firstentry * elemsize;
	}
}
