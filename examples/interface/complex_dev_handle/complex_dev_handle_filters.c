/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "complex_dev_handle_interface.h"

void starpu_complex_dev_handle_filter_block(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	struct starpu_complex_dev_handle_interface *complex_dev_handle_father = father_interface;
	struct starpu_complex_dev_handle_interface *complex_dev_handle_child = child_interface;

	uint32_t nx = complex_dev_handle_father->nx;
	size_t elemsize = sizeof(double);

	STARPU_ASSERT_MSG(nchunks <= nx, "%u parts for %u elements", nchunks, nx);

	uint32_t child_nx;
	size_t offset;
	/* Compute the split */
	starpu_filter_nparts_compute_chunk_size_and_offset(nx, nchunks, elemsize, id, 1,
						     &child_nx, &offset);

	complex_dev_handle_child->nx = child_nx;

	if (complex_dev_handle_father->dev_handle_real)
	{
		if (complex_dev_handle_father->ptr_real)
		{
			complex_dev_handle_child->ptr_real = complex_dev_handle_father->ptr_real + offset;
			complex_dev_handle_child->ptr_imaginary = complex_dev_handle_father->ptr_imaginary + offset;
		}
		complex_dev_handle_child->dev_handle_real = complex_dev_handle_father->dev_handle_real;
		complex_dev_handle_child->offset_real = complex_dev_handle_father->offset_real + offset;
		complex_dev_handle_child->dev_handle_imaginary = complex_dev_handle_father->dev_handle_imaginary;
		complex_dev_handle_child->offset_imaginary = complex_dev_handle_father->offset_imaginary + offset;
	}
}

void starpu_complex_dev_handle_filter_canonical(void *father_interface, void *child_interface, STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned id, unsigned nchunks)
{
	struct starpu_complex_dev_handle_interface *complex_dev_handle_father = father_interface;
	struct starpu_vector_interface *vector_child = child_interface;

	STARPU_ASSERT_MSG(nchunks == 2, "complex_dev_handle can only be split into two pieces");
	STARPU_ASSERT_MSG(id < 2, "complex_dev_handle has only two pieces");

	vector_child->id = STARPU_VECTOR_INTERFACE_ID;

	vector_child->nx = complex_dev_handle_father->nx;
	vector_child->elemsize = sizeof(double);
	vector_child->slice_base = 0;
	vector_child->allocsize = vector_child->nx * vector_child->elemsize;

	if (complex_dev_handle_father->dev_handle_real)
	{
		if (complex_dev_handle_father->ptr_real)
		{
			if (id == 0)
				vector_child->ptr = complex_dev_handle_father->ptr_real;
			else
				vector_child->ptr = complex_dev_handle_father->ptr_imaginary;
		}
		if (id == 0)
		{
			vector_child->dev_handle = complex_dev_handle_father->dev_handle_real;
			vector_child->offset = complex_dev_handle_father->offset_real;
		}
		else
		{
			vector_child->dev_handle = complex_dev_handle_father->dev_handle_imaginary;
			vector_child->offset = complex_dev_handle_father->offset_imaginary;
		}
		
	}
}

struct starpu_data_interface_ops *starpu_complex_dev_handle_filter_canonical_child_ops(STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, unsigned child)
{
	return &starpu_interface_vector_ops;
}
