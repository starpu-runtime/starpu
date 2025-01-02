/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

double *starpu_complex_get_real(starpu_data_handle_t handle)
{
	struct starpu_complex_interface *complex_interface =
		(struct starpu_complex_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return complex_interface->real;
}

double *starpu_complex_get_imaginary(starpu_data_handle_t handle)
{
	struct starpu_complex_interface *complex_interface =
		(struct starpu_complex_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return complex_interface->imaginary;
}

int starpu_complex_get_nx(starpu_data_handle_t handle)
{
	struct starpu_complex_interface *complex_interface =
		(struct starpu_complex_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	return complex_interface->nx;
}

static void complex_register_data_handle(starpu_data_handle_t handle, int home_node, void *data_interface)
{
	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *) data_interface;

	int node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_complex_interface *local_interface = (struct starpu_complex_interface *)
			starpu_data_get_interface_on_node(handle, node);

		local_interface->nx = complex_interface->nx;
		if (node == home_node)
		{
			local_interface->real = complex_interface->real;
			local_interface->imaginary = complex_interface->imaginary;
		}
		else
		{
			local_interface->real = NULL;
			local_interface->imaginary = NULL;
		}
	}
}

static starpu_ssize_t complex_allocate_data_on_node(void *data_interface, unsigned node)
{
	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *) data_interface;

	double *addr_real = NULL;
	double *addr_imaginary = NULL;
	starpu_ssize_t requested_memory = complex_interface->nx * sizeof(complex_interface->real[0]);

	addr_real = (double*) starpu_malloc_on_node(node, requested_memory);
	if (!addr_real)
		goto fail_real;
	addr_imaginary = (double*) starpu_malloc_on_node(node, requested_memory);
	if (!addr_imaginary)
		goto fail_imaginary;

	/* update the data properly in consequence */
	complex_interface->real = addr_real;
	complex_interface->imaginary = addr_imaginary;

	return 2*requested_memory;

fail_imaginary:
	starpu_free_on_node(node, (uintptr_t) addr_real, requested_memory);
fail_real:
	return -ENOMEM;
}

static void complex_free_data_on_node(void *data_interface, unsigned node)
{
	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *) data_interface;
	starpu_ssize_t requested_memory = complex_interface->nx * sizeof(complex_interface->real[0]);

	starpu_free_on_node(node, (uintptr_t) complex_interface->real, requested_memory);
	complex_interface->real = NULL;
	starpu_free_on_node(node, (uintptr_t) complex_interface->imaginary, requested_memory);
	complex_interface->imaginary = NULL;
}

static size_t complex_get_size(starpu_data_handle_t handle)
{
	size_t size;
	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	size = complex_interface->nx * 2 * sizeof(double);
	return size;
}

static uint32_t complex_footprint(starpu_data_handle_t handle)
{
	return starpu_hash_crc32c_be(starpu_complex_get_nx(handle), 0);
}

static int complex_pack_data(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *)
		starpu_data_get_interface_on_node(handle, node);

	*count = complex_get_size(handle);
	if (ptr != NULL)
	{
		char *data;
		data = (void*) starpu_malloc_on_node_flags(node, *count, 0);
		*ptr = data;
		memcpy(data, complex_interface->real, complex_interface->nx*sizeof(double));
		memcpy(data+complex_interface->nx*sizeof(double), complex_interface->imaginary, complex_interface->nx*sizeof(double));
	}

	return 0;
}

static int complex_peek_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	char *data = ptr;
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *)
		starpu_data_get_interface_on_node(handle, node);

	STARPU_ASSERT(count == 2 * complex_interface->nx * sizeof(double));
	memcpy(complex_interface->real, data, complex_interface->nx*sizeof(double));
	memcpy(complex_interface->imaginary, data+complex_interface->nx*sizeof(double), complex_interface->nx*sizeof(double));

	return 0;
}

static int complex_unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	complex_peek_data(handle, node, ptr, count);

	starpu_free_on_node_flags(node, (uintptr_t) ptr, count, 0);

	return 0;
}

static starpu_ssize_t complex_describe(void *data_interface, char *buf, size_t size)
{
	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *) data_interface;
	return snprintf(buf, size, "Complex%d", complex_interface->nx);
}

static int complex_compare(void *data_interface_a, void *data_interface_b)
{
	struct starpu_complex_interface *complex_a = (struct starpu_complex_interface *) data_interface_a;
	struct starpu_complex_interface *complex_b = (struct starpu_complex_interface *) data_interface_b;

	return (complex_a->nx == complex_b->nx);
}

#define _pack(dst, src)   do { memcpy(dst, &src, sizeof(src)); dst += sizeof(src); } while(0)
#define _unpack(dst, src) do { memcpy(&dst, src, sizeof(dst)); src += sizeof(dst); } while(0)

static starpu_ssize_t complex_size_meta(struct starpu_complex_interface *complex_interface)
{
	return sizeof(complex_interface->real) + sizeof(complex_interface->imaginary) + sizeof(complex_interface->nx);
}

static int complex_pack_meta(void *data_interface, void **ptr, starpu_ssize_t *count)
{
 	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *) data_interface;

	*count = complex_size_meta(complex_interface);
	*ptr = calloc(1, *count);

	char *cur = *ptr;
	_pack(cur, complex_interface->real);
	_pack(cur, complex_interface->imaginary);
	_pack(cur, complex_interface->nx);

	return 0;
}

static int complex_unpack_meta(void **data_interface, void *ptr, starpu_ssize_t *count)
{
	*data_interface = calloc(1, sizeof(struct starpu_complex_interface));
	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *) (*data_interface);
	char *cur = ptr;

	_unpack(complex_interface->real, cur);
	_unpack(complex_interface->imaginary, cur);
	_unpack(complex_interface->nx, cur);

	*count = complex_size_meta(complex_interface);

	return 0;
}

int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data)
{
	struct starpu_complex_interface *src_complex = src_interface;
	struct starpu_complex_interface *dst_complex = dst_interface;
	int ret = 0;

	if (starpu_interface_copy((uintptr_t) src_complex->real, 0, src_node,
				  (uintptr_t) dst_complex->real, 0, dst_node,
				  src_complex->nx*sizeof(src_complex->real[0]),
				  async_data))
		ret = -EAGAIN;
	if (starpu_interface_copy((uintptr_t) src_complex->imaginary, 0, src_node,
				  (uintptr_t) dst_complex->imaginary, 0, dst_node,
				  src_complex->nx*sizeof(src_complex->imaginary[0]),
				  async_data))
		ret = -EAGAIN;
	return ret;
}

const struct starpu_data_copy_methods complex_copy_methods =
{
	.any_to_any = copy_any_to_any
};

struct starpu_data_interface_ops interface_complex_ops =
{
	.register_data_handle = complex_register_data_handle,
	.allocate_data_on_node = complex_allocate_data_on_node,
	.free_data_on_node = complex_free_data_on_node,
	.copy_methods = &complex_copy_methods,
	.get_size = complex_get_size,
	.footprint = complex_footprint,
	.interfaceid = STARPU_UNKNOWN_INTERFACE_ID,
	.interface_size = sizeof(struct starpu_complex_interface),
	.to_pointer = NULL,
	.pack_data = complex_pack_data,
	.peek_data = complex_peek_data,
	.unpack_data = complex_unpack_data,
	.describe = complex_describe,
	.compare = complex_compare,
	.pack_meta = complex_pack_meta,
	.unpack_meta = complex_unpack_meta,
	.free_meta = NULL
};

void starpu_complex_data_register_ops()
{
	starpu_data_register_ops(&interface_complex_ops);
}

void starpu_complex_data_register(starpu_data_handle_t *handleptr, int home_node, double *real, double *imaginary, int nx)
{
	struct starpu_complex_interface complex =
	{
		.real = real,
		.imaginary = imaginary,
		.nx = nx
	};

	starpu_data_register(handleptr, home_node, &complex, &interface_complex_ops);
}
