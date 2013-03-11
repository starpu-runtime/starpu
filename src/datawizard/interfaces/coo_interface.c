/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012 inria
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
#include <common/fxt.h>
#include <datawizard/memalloc.h>

static int
copy_any_to_any(void *src_interface, unsigned src_node,
		void *dst_interface, unsigned dst_node, void *async_data)
{
	size_t size = 0;
	struct starpu_coo_interface *src_coo, *dst_coo;
	int ret = 0;

	src_coo = (struct starpu_coo_interface *) src_interface;
	dst_coo = (struct starpu_coo_interface *) dst_interface;

	size = src_coo->n_values * sizeof(src_coo->columns[0]);
	if (starpu_interface_copy(
		(uintptr_t) src_coo->columns, 0, src_node,
		(uintptr_t) dst_coo->columns, 0, dst_node,
		size, async_data))
		ret = -EAGAIN;

	/* sizeof(src_coo->columns[0]) == sizeof(src_coo->rows[0]) */
	if (starpu_interface_copy(
		(uintptr_t) src_coo->rows, 0, src_node,
		(uintptr_t) dst_coo->rows, 0, dst_node,
		size, async_data))
		ret = -EAGAIN;

	size = src_coo->n_values * src_coo->elemsize;
	if (starpu_interface_copy(
		src_coo->values, 0, src_node,
		dst_coo->values, 0, dst_node,
		size, async_data))
		ret = -EAGAIN;

	_STARPU_TRACE_DATA_COPY(src_node, dst_node,
		src_coo->n_values *
		(2 * sizeof(src_coo->rows[0]) + src_coo->elemsize));

	return ret;
}

static const struct starpu_data_copy_methods coo_copy_data_methods =
{
	.any_to_any          = copy_any_to_any,
};

static void
register_coo_handle(starpu_data_handle_t handle, unsigned home_node,
		    void *data_interface)
{
	struct starpu_coo_interface *coo_interface =
		(struct starpu_coo_interface *) data_interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_coo_interface *local_interface;
		local_interface = (struct starpu_coo_interface *)
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
		{
			local_interface->values = coo_interface->values;
			local_interface->columns = coo_interface->columns;
			local_interface->rows = coo_interface->rows;
		}
		else
		{
			local_interface->values = 0;
			local_interface->columns = 0;
			local_interface->rows = 0;
		}

		local_interface->nx = coo_interface->nx;
		local_interface->ny = coo_interface->ny;
		local_interface->n_values = coo_interface->n_values;
		local_interface->elemsize = coo_interface->elemsize;
	}
}

static ssize_t
allocate_coo_buffer_on_node(void *data_interface, unsigned dst_node)
{
	uint32_t *addr_columns = NULL;
	uint32_t *addr_rows = NULL;
	uintptr_t addr_values = 0;

	struct starpu_coo_interface *coo_interface =
		(struct starpu_coo_interface *) data_interface;

	uint32_t n_values = coo_interface->n_values;
	size_t elemsize = coo_interface->elemsize;

	addr_columns = (void*) starpu_malloc_on_node(dst_node, n_values * sizeof(coo_interface->columns[0]));
	if (STARPU_UNLIKELY(addr_columns == NULL))
		goto fail_columns;
	addr_rows = (void*) starpu_malloc_on_node(dst_node, n_values * sizeof(coo_interface->rows[0]));
	if (STARPU_UNLIKELY(addr_rows == NULL))
		goto fail_rows;
	addr_values = starpu_malloc_on_node(dst_node, n_values * elemsize);
	if (STARPU_UNLIKELY(addr_values == (uintptr_t) NULL))
		goto fail_values;

	coo_interface->columns = addr_columns;
	coo_interface->rows = addr_rows;
	coo_interface->values = addr_values;

	return n_values * (sizeof(coo_interface->columns[0]) + sizeof(coo_interface->rows[0]) + elemsize);

fail_values:
	starpu_free_on_node(dst_node, (uintptr_t) addr_rows, n_values * sizeof(coo_interface->rows[0]));
fail_rows:
	starpu_free_on_node(dst_node, (uintptr_t) addr_columns, n_values * sizeof(coo_interface->columns[0]));
fail_columns:
	return -ENOMEM;
}

static void
free_coo_buffer_on_node(void *data_interface, unsigned node)
{
	struct starpu_coo_interface *coo_interface = (struct starpu_coo_interface *) data_interface;
	uint32_t n_values = coo_interface->n_values;
	size_t elemsize = coo_interface->elemsize;

	starpu_free_on_node(node, (uintptr_t) coo_interface->columns, n_values * sizeof(coo_interface->columns[0]));
	starpu_free_on_node(node, (uintptr_t) coo_interface->rows, n_values * sizeof(coo_interface->rows[0]));
	starpu_free_on_node(node, coo_interface->values, n_values * elemsize);
}

static size_t
coo_interface_get_size(starpu_data_handle_t handle)
{
	struct starpu_coo_interface *coo_interface;
	coo_interface = (struct starpu_coo_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	return coo_interface->nx * coo_interface->ny * coo_interface->elemsize;
}

static uint32_t
coo_interface_footprint(starpu_data_handle_t handle)
{
	struct starpu_coo_interface *coo_interface;
	coo_interface = (struct starpu_coo_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	return starpu_crc32_be(coo_interface->nx * coo_interface->ny, 0);
}

static int
coo_compare(void *a, void *b)
{
	struct starpu_coo_interface *coo_a, *coo_b;

	coo_a = (struct starpu_coo_interface *) a;
	coo_b = (struct starpu_coo_interface *) b;

	return (coo_a->nx == coo_b->nx &&
		coo_a->ny == coo_b->ny &&
		coo_a->n_values == coo_b->n_values &&
		coo_a->elemsize == coo_b->elemsize);
}

static void
display_coo_interface(starpu_data_handle_t handle, FILE *f)
{
	struct starpu_coo_interface *coo_interface =
	coo_interface = (struct starpu_coo_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	fprintf(f, "%u\t%u", coo_interface->nx, coo_interface->ny);
}

struct starpu_data_interface_ops _starpu_interface_coo_ops =
{
	.register_data_handle  = register_coo_handle,
	.allocate_data_on_node = allocate_coo_buffer_on_node,
	.handle_to_pointer     = NULL,
	.free_data_on_node     = free_coo_buffer_on_node,
	.copy_methods          = &coo_copy_data_methods,
	.get_size              = coo_interface_get_size,
	.footprint             = coo_interface_footprint,
	.compare               = coo_compare,
	.interfaceid           = STARPU_COO_INTERFACE_ID,
	.interface_size        = sizeof(struct starpu_coo_interface),
	.display               = display_coo_interface
};

void
starpu_coo_data_register(starpu_data_handle_t *handleptr, unsigned home_node,
			 uint32_t nx, uint32_t ny, uint32_t n_values,
			 uint32_t *columns, uint32_t *rows,
			 uintptr_t values, size_t elemsize)
{
	struct starpu_coo_interface coo_interface =
	{
		.values = values,
		.columns = columns,
		.rows = rows,
		.nx = nx,
		.ny = ny,
		.n_values = n_values,
		.elemsize = elemsize,
	};

	starpu_data_register(handleptr, home_node, &coo_interface,
			     &_starpu_interface_coo_ops);
}
