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

uintptr_t starpu_complex_dev_handle_get_ptr_real(starpu_data_handle_t handle)
{
	struct starpu_complex_dev_handle_interface *complex_dev_handle_interface =
		(struct starpu_complex_dev_handle_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return complex_dev_handle_interface->ptr_real;
}

uintptr_t starpu_complex_dev_handle_get_ptr_imaginary(starpu_data_handle_t handle)
{
	struct starpu_complex_dev_handle_interface *complex_dev_handle_interface =
		(struct starpu_complex_dev_handle_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return complex_dev_handle_interface->ptr_imaginary;
}

int starpu_complex_dev_handle_get_nx(starpu_data_handle_t handle)
{
	struct starpu_complex_dev_handle_interface *complex_dev_handle_interface =
		(struct starpu_complex_dev_handle_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return complex_dev_handle_interface->nx;
}

uintptr_t starpu_complex_dev_handle_get_dev_handle_real(starpu_data_handle_t handle)
{
	struct starpu_complex_dev_handle_interface *complex_dev_handle_interface =
		(struct starpu_complex_dev_handle_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return complex_dev_handle_interface->dev_handle_real;
}

uintptr_t starpu_complex_dev_handle_get_dev_handle_imaginary(starpu_data_handle_t handle)
{
	struct starpu_complex_dev_handle_interface *complex_dev_handle_interface =
		(struct starpu_complex_dev_handle_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return complex_dev_handle_interface->dev_handle_imaginary;
}

size_t starpu_complex_dev_handle_get_offset_real(starpu_data_handle_t handle)
{
	struct starpu_complex_dev_handle_interface *complex_dev_handle_interface =
		(struct starpu_complex_dev_handle_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return complex_dev_handle_interface->offset_real;
}

size_t starpu_complex_dev_handle_get_offset_imaginary(starpu_data_handle_t handle)
{
	struct starpu_complex_dev_handle_interface *complex_dev_handle_interface =
		(struct starpu_complex_dev_handle_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return complex_dev_handle_interface->offset_imaginary;
}

static void complex_dev_handle_register_data_handle(starpu_data_handle_t handle, int home_node, void *data_interface)
{
	struct starpu_complex_dev_handle_interface *complex_dev_handle_interface = (struct starpu_complex_dev_handle_interface *) data_interface;

	int node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_complex_dev_handle_interface *local_interface = (struct starpu_complex_dev_handle_interface *)
			starpu_data_get_interface_on_node(handle, node);

		local_interface->nx = complex_dev_handle_interface->nx;
		if (node == home_node)
		{
			local_interface->ptr_real = complex_dev_handle_interface->ptr_real;
			local_interface->dev_handle_real = complex_dev_handle_interface->dev_handle_real;
			local_interface->offset_real = complex_dev_handle_interface->offset_real;
			local_interface->ptr_imaginary = complex_dev_handle_interface->ptr_imaginary;
			local_interface->dev_handle_imaginary = complex_dev_handle_interface->dev_handle_imaginary;
			local_interface->offset_imaginary = complex_dev_handle_interface->offset_imaginary;
		}
		else
		{
			local_interface->ptr_real = 0;
			local_interface->dev_handle_real = 0;
			local_interface->offset_real = 0;
			local_interface->ptr_imaginary = 0;
			local_interface->dev_handle_imaginary = 0;
			local_interface->offset_imaginary = 0;
		}
	}
}

static starpu_ssize_t complex_dev_handle_allocate_data_on_node(void *data_interface, unsigned node)
{
	struct starpu_complex_dev_handle_interface *complex_dev_handle_interface = (struct starpu_complex_dev_handle_interface *) data_interface;

	uintptr_t addr_real = 0, addr_imaginary = 0, dev_handle_real, dev_handle_imaginary;

	starpu_ssize_t requested_memory = complex_dev_handle_interface->nx * sizeof(double);

	dev_handle_real = starpu_malloc_on_node(node, requested_memory);
	if (!dev_handle_real)
		goto fail_real;
	dev_handle_imaginary = starpu_malloc_on_node(node, requested_memory);
	if (!dev_handle_imaginary)
		goto fail_imaginary;

	if (!starpu_node_needs_offset(node))
	{
		addr_real = dev_handle_real;
		addr_imaginary = dev_handle_imaginary;
	}

	/* update the data properly in consequence */
	complex_dev_handle_interface->ptr_real = addr_real;
	complex_dev_handle_interface->dev_handle_real = dev_handle_real;
	complex_dev_handle_interface->offset_real = 0;
	complex_dev_handle_interface->ptr_imaginary = addr_imaginary;
	complex_dev_handle_interface->dev_handle_imaginary = dev_handle_imaginary;
	complex_dev_handle_interface->offset_imaginary = 0;

	return 2*requested_memory;

fail_imaginary:
	starpu_free_on_node(node, dev_handle_real, requested_memory);
fail_real:
	return -ENOMEM;
}

static void complex_dev_handle_free_data_on_node(void *data_interface, unsigned node)
{
	struct starpu_complex_dev_handle_interface *complex_dev_handle_interface = (struct starpu_complex_dev_handle_interface *) data_interface;
	starpu_ssize_t requested_memory = complex_dev_handle_interface->nx * sizeof(double);

	starpu_free_on_node(node, (uintptr_t) complex_dev_handle_interface->dev_handle_real, requested_memory);
	complex_dev_handle_interface->ptr_real = 0;
	complex_dev_handle_interface->dev_handle_real = 0;
	starpu_free_on_node(node, (uintptr_t) complex_dev_handle_interface->dev_handle_imaginary, requested_memory);
	complex_dev_handle_interface->ptr_imaginary = 0;
	complex_dev_handle_interface->dev_handle_imaginary = 0;
}

static size_t complex_dev_handle_get_size(starpu_data_handle_t handle)
{
	size_t size;
	struct starpu_complex_dev_handle_interface *complex_dev_handle_interface = (struct starpu_complex_dev_handle_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	size = complex_dev_handle_interface->nx * 2 * sizeof(double);
	return size;
}

static uint32_t complex_dev_handle_footprint(starpu_data_handle_t handle)
{
	return starpu_hash_crc32c_be(starpu_complex_dev_handle_get_nx(handle), 0);
}

static int complex_dev_handle_pack_data(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_complex_dev_handle_interface *complex_dev_handle_interface = (struct starpu_complex_dev_handle_interface *)
		starpu_data_get_interface_on_node(handle, node);

	*count = complex_dev_handle_get_size(handle);
	if (ptr != NULL)
	{
		char *real = (void *)complex_dev_handle_interface->ptr_real;
		char *imaginary = (void *)complex_dev_handle_interface->ptr_imaginary;

		*ptr = (void*) starpu_malloc_on_node_flags(node, *count, 0);
		char *data = (char*) *ptr;
		memcpy(data, real, complex_dev_handle_interface->nx*sizeof(double));
		memcpy(data+complex_dev_handle_interface->nx*sizeof(double), imaginary, complex_dev_handle_interface->nx*sizeof(double));
	}

	return 0;
}

static int complex_dev_handle_peek_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	char *data = ptr;
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_complex_dev_handle_interface *complex_dev_handle_interface = (struct starpu_complex_dev_handle_interface *)
		starpu_data_get_interface_on_node(handle, node);

	STARPU_ASSERT(count == 2 * complex_dev_handle_interface->nx * sizeof(double));

	char *real = (void *)complex_dev_handle_interface->ptr_real;
	char *imaginary = (void *)complex_dev_handle_interface->ptr_imaginary;

	memcpy(real, data, complex_dev_handle_interface->nx*sizeof(double));
	memcpy(imaginary, data+complex_dev_handle_interface->nx*sizeof(double), complex_dev_handle_interface->nx*sizeof(double));

	return 0;
}

static int complex_dev_handle_unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	complex_dev_handle_peek_data(handle, node, ptr, count);

	starpu_free_on_node_flags(node, (uintptr_t) ptr, count, 0);

	return 0;
}

static starpu_ssize_t complex_dev_handle_describe(void *data_interface, char *buf, size_t size)
{
	struct starpu_complex_dev_handle_interface *complex_dev_handle_interface = (struct starpu_complex_dev_handle_interface *) data_interface;
	return snprintf(buf, size, "Complex_dev_handle%d", complex_dev_handle_interface->nx);
}

static int complex_dev_handle_compare(void *data_interface_a, void *data_interface_b)
{
	struct starpu_complex_dev_handle_interface *complex_dev_handle_a = (struct starpu_complex_dev_handle_interface *) data_interface_a;
	struct starpu_complex_dev_handle_interface *complex_dev_handle_b = (struct starpu_complex_dev_handle_interface *) data_interface_b;

	return (complex_dev_handle_a->nx == complex_dev_handle_b->nx);
}

int copy_any_to_any(void *src_interface, unsigned src_node,
		    void *dst_interface, unsigned dst_node,
		    void *async_data)
{
	struct starpu_complex_dev_handle_interface *src_complex_dev_handle = src_interface;
	struct starpu_complex_dev_handle_interface *dst_complex_dev_handle = dst_interface;
	int ret = 0;

	if (starpu_interface_copy(src_complex_dev_handle->dev_handle_real, src_complex_dev_handle->offset_real, src_node,
				  dst_complex_dev_handle->dev_handle_real, dst_complex_dev_handle->offset_real, dst_node,
				  src_complex_dev_handle->nx*sizeof(double),
				  async_data))
		ret = -EAGAIN;
	if (starpu_interface_copy(src_complex_dev_handle->dev_handle_imaginary, src_complex_dev_handle->offset_imaginary, src_node,
				  dst_complex_dev_handle->dev_handle_imaginary, dst_complex_dev_handle->offset_imaginary, dst_node,
				  src_complex_dev_handle->nx*sizeof(double),
				  async_data))
		ret = -EAGAIN;
	return ret;
}

const struct starpu_data_copy_methods complex_dev_handle_copy_methods =
{
	.any_to_any = copy_any_to_any
};

struct starpu_data_interface_ops interface_complex_dev_handle_ops =
{
	.register_data_handle = complex_dev_handle_register_data_handle,
	.allocate_data_on_node = complex_dev_handle_allocate_data_on_node,
	.free_data_on_node = complex_dev_handle_free_data_on_node,
	.copy_methods = &complex_dev_handle_copy_methods,
	.get_size = complex_dev_handle_get_size,
	.footprint = complex_dev_handle_footprint,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct starpu_complex_dev_handle_interface),
	.to_pointer = NULL,
	.pack_data = complex_dev_handle_pack_data,
	.peek_data = complex_dev_handle_peek_data,
	.unpack_data = complex_dev_handle_unpack_data,
	.describe = complex_dev_handle_describe,
	.compare = complex_dev_handle_compare
};

void starpu_complex_dev_handle_data_register(starpu_data_handle_t *handleptr, int home_node, uintptr_t ptr_real, uintptr_t ptr_imaginary, int nx)
{
	struct starpu_complex_dev_handle_interface complex_dev_handle =
	{
		.ptr_real = ptr_real,
		.dev_handle_real = ptr_real,
		.ptr_imaginary = ptr_imaginary,
		.dev_handle_imaginary = ptr_imaginary,
		.nx = nx
	};

	starpu_data_register(handleptr, home_node, &complex_dev_handle, &interface_complex_dev_handle_ops);
}

void starpu_complex_dev_handle_ptr_register(starpu_data_handle_t handle, int node, uintptr_t ptr_real, uintptr_t ptr_imaginary, uintptr_t dev_handle_real, uintptr_t dev_handle_imaginary, size_t offset_real, size_t offset_imaginary)
{
	struct starpu_complex_dev_handle_interface *complex_dev_handle_interface = starpu_data_get_interface_on_node(handle, node);
	starpu_data_ptr_register(handle, node);
	complex_dev_handle_interface->ptr_real = ptr_real;
	complex_dev_handle_interface->dev_handle_real = dev_handle_real;
	complex_dev_handle_interface->offset_real = offset_real;
	complex_dev_handle_interface->ptr_imaginary = ptr_imaginary;
	complex_dev_handle_interface->dev_handle_imaginary = dev_handle_imaginary;
	complex_dev_handle_interface->offset_imaginary = offset_imaginary;
}
