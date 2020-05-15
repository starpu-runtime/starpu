/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data);

static const struct starpu_data_copy_methods variable_copy_data_methods_s =
{
	.any_to_any = copy_any_to_any,
};

static void register_variable_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface);
static starpu_ssize_t allocate_variable_buffer_on_node(void *data_interface_, unsigned dst_node);
static void *variable_to_pointer(void *data_interface, unsigned node);
static int variable_pointer_is_inside(void *data_interface, unsigned node, void *ptr);
static void free_variable_buffer_on_node(void *data_interface, unsigned node);
static size_t variable_interface_get_size(starpu_data_handle_t handle);
static uint32_t footprint_variable_interface_crc32(starpu_data_handle_t handle);
static int variable_compare(void *data_interface_a, void *data_interface_b);
static void display_variable_interface(starpu_data_handle_t handle, FILE *f);
static int pack_variable_handle(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count);
static int unpack_variable_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);
static starpu_ssize_t describe(void *data_interface, char *buf, size_t size);

struct starpu_data_interface_ops starpu_interface_variable_ops =
{
	.register_data_handle = register_variable_handle,
	.allocate_data_on_node = allocate_variable_buffer_on_node,
	.to_pointer = variable_to_pointer,
	.pointer_is_inside = variable_pointer_is_inside,
	.free_data_on_node = free_variable_buffer_on_node,
	.copy_methods = &variable_copy_data_methods_s,
	.get_size = variable_interface_get_size,
	.footprint = footprint_variable_interface_crc32,
	.compare = variable_compare,
	.interfaceid = STARPU_VARIABLE_INTERFACE_ID,
	.interface_size = sizeof(struct starpu_variable_interface),
	.display = display_variable_interface,
	.pack_data = pack_variable_handle,
	.unpack_data = unpack_variable_handle,
	.describe = describe,
	.name = "STARPU_VARIABLE_INTERFACE"
};

static void *variable_to_pointer(void *data_interface, unsigned node)
{
	(void) node;
	return (void*) STARPU_VARIABLE_GET_PTR(data_interface);
}

static int variable_pointer_is_inside(void *data_interface, unsigned node, void *ptr)
{
	(void) node;
	struct starpu_variable_interface *variable_interface = data_interface;
	return (char*) ptr >= (char*) variable_interface->ptr &&
		(char*) ptr < (char*) variable_interface->ptr + variable_interface->elemsize;
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
void starpu_variable_data_register(starpu_data_handle_t *handleptr, int home_node,
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
#ifndef STARPU_SIMGRID
	if (home_node >= 0 && starpu_node_get_kind(home_node) == STARPU_CPU_RAM)
	{
		if (elemsize)
		{
			STARPU_ASSERT_ACCESSIBLE(ptr);
			STARPU_ASSERT_ACCESSIBLE(ptr + elemsize - 1);
		}
	}
#endif

	starpu_data_register(handleptr, home_node, &variable, &starpu_interface_variable_ops);
}

void starpu_variable_ptr_register(starpu_data_handle_t handle, unsigned node,
				  uintptr_t ptr, uintptr_t dev_handle, size_t offset)
{
	struct starpu_variable_interface *variable_interface = starpu_data_get_interface_on_node(handle, node);
	starpu_data_ptr_register(handle, node);
	variable_interface->ptr = ptr;
	variable_interface->dev_handle = dev_handle;
	variable_interface->offset = offset;
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
	return variable_a->elemsize == variable_b->elemsize;
}

static void display_variable_interface(starpu_data_handle_t handle, FILE *f)
{
	struct starpu_variable_interface *variable_interface = (struct starpu_variable_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	fprintf(f, "%ld\t", (long)variable_interface->elemsize);
}

static int pack_variable_handle(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_variable_interface *variable_interface = (struct starpu_variable_interface *)
		starpu_data_get_interface_on_node(handle, node);

	*count = variable_interface->elemsize;

	if (ptr != NULL)
	{
		*ptr = (void *)starpu_malloc_on_node_flags(node, *count, 0);
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

	starpu_free_on_node_flags(node, (uintptr_t)ptr, count, 0);

	return 0;
}

static size_t variable_interface_get_size(starpu_data_handle_t handle)
{
	struct starpu_variable_interface *variable_interface = (struct starpu_variable_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(variable_interface->id == STARPU_VARIABLE_INTERFACE_ID, "Error. The given data is not a variable.");
#endif

	return variable_interface->elemsize;
}

uintptr_t starpu_variable_get_local_ptr(starpu_data_handle_t handle)
{
	unsigned node;
	node = starpu_worker_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	return STARPU_VARIABLE_GET_PTR(starpu_data_get_interface_on_node(handle, node));
}

size_t starpu_variable_get_elemsize(starpu_data_handle_t handle)
{
	return STARPU_VARIABLE_GET_ELEMSIZE(starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM));
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

	starpu_interface_data_copy(src_node, dst_node, elemsize);

	return ret;
}
static starpu_ssize_t describe(void *data_interface, char *buf, size_t size)
{
	struct starpu_variable_interface *variable = (struct starpu_variable_interface *) data_interface;
	return snprintf(buf, size, "v%u",
			(unsigned) variable->elemsize);
}
