/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2021, 2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "complex_interface.h"

void starpu_complex_filter_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	struct starpu_complex_interface *complex_father = father_interface;
	struct starpu_complex_interface *complex_child = child_interface;

	uint32_t nx = complex_father->nx;
	size_t elemsize = 2*sizeof(double);

	STARPU_ASSERT_MSG(nchunks <= nx, "%u parts for %u elements", nchunks, nx);

	uint32_t child_nx;
	size_t offset;
	/* Compute the split */
	starpu_filter_nparts_compute_chunk_size_and_offset(nx, nchunks, elemsize, id, 1, &child_nx, &offset);

	complex_child->nx = child_nx;

	if (complex_father->dev_handle)
	{
		if (complex_father->ptr)
		{
			complex_child->ptr = complex_father->ptr + offset;
		}
		complex_child->dev_handle = complex_father->dev_handle;
		complex_child->offset = complex_father->offset + offset;
	}
}

