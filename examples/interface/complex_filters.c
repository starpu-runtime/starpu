/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void starpu_complex_filter_block(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	struct starpu_complex_interface *complex_parent = parent_interface;
	struct starpu_complex_interface *complex_child = child_interface;

	size_t nx = complex_parent->nx;
	size_t elemsize = sizeof(double);

	STARPU_ASSERT_MSG(nchunks <= nx, "%u parts for %zu elements", nchunks, nx);

	size_t child_nx;
	size_t offset;
	/* Compute the split */
	starpu_filter_nparts_compute_chunk_size_and_offset(nx, nchunks, elemsize, id, 1,  &child_nx, &offset);

	complex_child->nx = child_nx;

	if (complex_parent->real)
	{
		complex_child->real = (void*) ((uintptr_t) complex_parent->real + offset);
		complex_child->imaginary = (void*) ((uintptr_t) complex_parent->imaginary + offset);
	}
}

void starpu_complex_filter_canonical(void *parent_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	struct starpu_complex_interface *complex_parent = parent_interface;
	struct starpu_vector_interface *vector_child = child_interface;

	STARPU_ASSERT_MSG(nchunks == 2, "complex can only be split into two pieces");
	STARPU_ASSERT_MSG(id < 2, "complex has only two pieces");

	vector_child->id = STARPU_VECTOR_INTERFACE_ID;
	if (id == 0)
		vector_child->ptr = (uintptr_t) complex_parent->real;
	else
		vector_child->ptr = (uintptr_t) complex_parent->imaginary;

	/* the complex interface doesn't support dev_handle/offset */
	vector_child->dev_handle = vector_child->ptr;
	vector_child->offset = 0;

	vector_child->nx = complex_parent->nx;
	vector_child->elemsize = sizeof(double);
	vector_child->slice_base = 0;
	vector_child->allocsize = vector_child->nx * vector_child->elemsize;
}

struct starpu_data_interface_ops *starpu_complex_filter_canonical_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned child)
{
	return &starpu_interface_vector_ops;
}
