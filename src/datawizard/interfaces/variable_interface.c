/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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
#include <datawizard/coherency.h>
#include <datawizard/copy_driver.h>
#include <datawizard/filters.h>
#include <starpu_hash.h>
#include <starpu_cuda.h>
#include <starpu_opencl.h>
#include <drivers/opencl/driver_opencl.h>
#include <drivers/scc/driver_scc_source.h>
#include <drivers/mic/driver_mic_source.h>

static int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data);

static const struct starpu_data_copy_methods variable_copy_data_methods_s =
{
	.any_to_any = copy_any_to_any,
};

static void register_variable_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface);
static starpu_ssize_t allocate_variable_buffer_on_node(void *data_interface_, unsigned dst_node);
static void *variable_handle_to_pointer(starpu_data_handle_t data_handle, unsigned node);
static void free_variable_buffer_on_node(void *data_interface, unsigned node);
static size_t variable_interface_get_size(starpu_data_handle_t handle);
static uint32_t footprint_variable_interface_crc32(starpu_data_handle_t handle);
static int variable_compare(void *data_interface_a, void *data_interface_b);
static void display_variable_interface(starpu_data_handle_t handle, FILE *f);
static int pack_variable_handle(starpu_data_handle_t handle, unsigned node, void **ptr, ssize_t *count);
static int unpack_variable_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);

struct starpu_data_interface_ops starpu_interface_variable_ops =
{
	.register_data_handle = register_variable_handle,
	.allocate_data_on_node = allocate_variable_buffer_on_node,
	.handle_to_pointer = variable_handle_to_pointer,
	.free_data_on_node = free_variable_buffer_on_node,
	.copy_methods = &variable_copy_data_methods_s,
	.get_size = variable_interface_get_size,
	.footprint = footprint_variable_interface_crc32,
	.compare = variable_compare,
	.interfaceid = STARPU_VARIABLE_INTERFACE_ID,
	.interface_size = sizeof(struct starpu_variable_interface),
	.display = display_variable_interface,
	.pack_data = pack_variable_handle,
	.unpack_data = unpack_variable_handle
};

static void *variable_handle_to_pointer(starpu_data_handle_t handle, unsigned node)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	return (void*) STARPU_VARIABLE_GET_PTR(starpu_data_get_interface_on_node(handle, node));
}

static void register_variable_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface)
{
	struct starpu_variable_interface *variable_interface = (struct starpu_variable_interface *)data_interface;
	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_variable_interface *local_interface = (struct starpu_variable_interface *)
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
		{
			local_interface->ptr = variable_interface->ptr;
			local_interface->dev_handle = variable_interface->dev_handle;
			local_interface->offset = variable_interface->offset;
		}
		else
		{
			local_interface->ptr = 0;
			local_interface->dev_handle = 0;
			local_interface->offset = 0;
		}

		local_interface->id = variable_interface->id;
		local_interface->elemsize = variable_interface->elemsize;
	}
}

/* declare a new data with the variable interface */
void starpu_variable_data_register(starpu_data_handle_t *handleptr, unsigned home_node,
                        uintptr_t ptr, size_t elemsize)
{
	struct starpu_variable_interface variable =
	{
		.id = STARPU_VARIABLE_INTERFACE_ID,
		.ptr = ptr,
		.dev_handle = ptr,
		.offset = 0,
		.elemsize = elemsize
	};

#ifdef STARPU_USE_SCC
	_starpu_scc_set_offset_in_shared_memory((void*)variable.ptr, (void**)&(variable.dev_handle),
			&(variable.offset));
#endif

	starpu_data_register(handleptr, home_node, &variable, &starpu_interface_variable_ops);
}


static uint32_t footprint_variable_interface_crc32(starpu_data_handle_t handle)
{
	return starpu_hash_crc32c_be(starpu_variable_get_elemsize(handle), 0);
}

static int variable_compare(void *data_interface_a, void *data_interface_b)
{
	struct starpu_variable_interface *variable_a = (struct starpu_variable_interface *) data_interface_a;
	struct starpu_variable_interface *variable_b = (struct starpu_variable_interface *) data_interface_b;

	/* Two variables are considered compatible if they have the same size */
	return (variable_a->elemsize == variable_b->elemsize);
}

static void display_variable_interface(starpu_data_handle_t handle, FILE *f)
{
	struct starpu_variable_interface *variable_interface = (struct starpu_variable_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	fprintf(f, "%ld\t", (long)variable_interface->elemsize);
}

static int pack_variable_handle(starpu_data_handle_t handle, unsigned node, void **ptr, ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_variable_interface *variable_interface = (struct starpu_variable_interface *)
		starpu_data_get_interface_on_node(handle, node);

	*count = variable_interface->elemsize;

	if (ptr != NULL)
	{
		*ptr = malloc(*count);
		memcpy(*ptr, (void*)variable_interface->ptr, variable_interface->elemsize);
	}

	return 0;
}

static int unpack_variable_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_variable_interface *variable_interface = (struct starpu_variable_interface *)
		starpu_data_get_interface_on_node(handle, node);

	STARPU_ASSERT(count == variable_interface->elemsize);

	memcpy((void*)variable_interface->ptr, ptr, variable_interface->elemsize);
	return 0;
}

static size_t variable_interface_get_size(starpu_data_handle_t handle)
{
	struct starpu_variable_interface *variable_interface = (struct starpu_variable_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	return variable_interface->elemsize;
}

uintptr_t starpu_variable_get_local_ptr(starpu_data_handle_t handle)
{
	unsigned node;
	node = _starpu_memory_node_get_local_key();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	return STARPU_VARIABLE_GET_PTR(starpu_data_get_interface_on_node(handle, node));
}

size_t starpu_variable_get_elemsize(starpu_data_handle_t handle)
{
	return STARPU_VARIABLE_GET_ELEMSIZE(starpu_data_get_interface_on_node(handle, 0));
}

/* memory allocation/deallocation primitives for the variable interface */

/* returns the size of the allocated area */
static starpu_ssize_t allocate_variable_buffer_on_node(void *data_interface_, unsigned dst_node)
{
	struct starpu_variable_interface *variable_interface = (struct starpu_variable_interface *) data_interface_;
	size_t elemsize = variable_interface->elemsize;
	uintptr_t addr = starpu_malloc_on_node(dst_node, elemsize);

	if (!addr)
		return -ENOMEM;

	/* update the data properly in consequence */
	variable_interface->ptr = addr;

	return elemsize;
}

static void free_variable_buffer_on_node(void *data_interface, unsigned node)
{
	struct starpu_variable_interface *variable_interface = (struct starpu_variable_interface *) data_interface;
	starpu_free_on_node(node, variable_interface->ptr, variable_interface->elemsize);
}

static int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data)
{
	struct starpu_variable_interface *src_variable = (struct starpu_variable_interface *) src_interface;
	struct starpu_variable_interface *dst_variable = (struct starpu_variable_interface *) dst_interface;

	size_t elemsize = dst_variable->elemsize;

	uintptr_t ptr_src = src_variable->ptr;
	uintptr_t ptr_dst = dst_variable->ptr;
	int ret;

	ret = starpu_interface_copy(ptr_src, 0, src_node, ptr_dst, 0, dst_node, elemsize, async_data);

	_STARPU_TRACE_DATA_COPY(src_node, dst_node, elemsize);

	return ret;
}
