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

void starpu_canonical_block_filter_bcsr(void *father_interface, void *child_interface, __attribute__((unused)) struct starpu_data_filter *f, unsigned id, __attribute__((unused)) unsigned nparts)
{
	struct starpu_bcsr_interface_s *bcsr_father = father_interface;
	/* each chunk becomes a small dense matrix */
	starpu_matrix_interface_t *matrix_child = child_interface;
	
	uint32_t nnz = bcsr_father->nnz;

	size_t elemsize = bcsr_father->elemsize;
	uint32_t firstentry = bcsr_father->firstentry;

	/* size of the tiles */
	uint32_t r = bcsr_father->r;
	uint32_t c = bcsr_father->c;
	
	uint32_t ptr_offset = c*r*id*elemsize;

	matrix_child->nx = c;
	matrix_child->ny = r;
	matrix_child->ld = c;
	matrix_child->elemsize = elemsize;

	if (bcsr_father->nzval) {
	  uint8_t *nzval = (uint8_t *)(bcsr_father->nzval);
	  matrix_child->ptr = (uintptr_t)&nzval[firstentry + ptr_offset];
	}
}
