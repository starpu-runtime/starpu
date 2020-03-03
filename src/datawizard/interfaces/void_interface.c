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

static int dummy_copy(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data);

static const struct starpu_data_copy_methods void_copy_data_methods_s =
{
	.any_to_any = dummy_copy,
};

static void register_void_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface);
static starpu_ssize_t allocate_void_buffer_on_node(void *data_interface_, unsigned dst_node);
static int void_pointer_is_inside(void *data_interface, unsigned node, void *ptr);
static void free_void_buffer_on_node(void *data_interface, unsigned node);
static size_t void_interface_get_size(starpu_data_handle_t handle);
static uint32_t footprint_void_interface_crc32(starpu_data_handle_t handle);
static int void_compare(void *data_interface_a, void *data_interface_b);
static void display_void_interface(starpu_data_handle_t handle, FILE *f);
static int pack_void_handle(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count);
static int unpack_void_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);
static starpu_ssize_t describe(void *data_interface, char *buf, size_t size);

struct starpu_data_interface_ops starpu_interface_void_ops =
{
	.register_data_handle = register_void_handle,
	.allocate_data_on_node = allocate_void_buffer_on_node,
	.free_data_on_node = free_void_buffer_on_node,
	.copy_methods = &void_copy_data_methods_s,
	.get_size = void_interface_get_size,
	.footprint = footprint_void_interface_crc32,
	.compare = void_compare,
	.interfaceid = STARPU_VOID_INTERFACE_ID,
	.interface_size = 0,
	.display = display_void_interface,
	.pack_data = pack_void_handle,
	.unpack_data = unpack_void_handle,
	.describe = describe,
	.pointer_is_inside = void_pointer_is_inside,
	.name = "STARPU_VOID_INTERFACE"
};

static int void_pointer_is_inside(void *data_interface STARPU_ATTRIBUTE_UNUSED, unsigned node STARPU_ATTRIBUTE_UNUSED, void *ptr STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

static void register_void_handle(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED,
				unsigned home_node STARPU_ATTRIBUTE_UNUSED,
				void *data_interface STARPU_ATTRIBUTE_UNUSED)
{
	/* Since there is no real data to register, we don't do anything */
}

/* declare a new data with the void interface */
void starpu_void_data_register(starpu_data_handle_t *handleptr)
{
	starpu_data_register(handleptr, STARPU_MAIN_RAM, NULL, &starpu_interface_void_ops);
}


static uint32_t footprint_void_interface_crc32(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

static int void_compare(void *data_interface_a STARPU_ATTRIBUTE_UNUSED,
			void *data_interface_b STARPU_ATTRIBUTE_UNUSED)
{
	/* There is no allocation required, and therefore nothing to cache
	 * anyway. */
	return 1;
}

static void display_void_interface(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED, FILE *f)
{
	fprintf(f, "void\t");
}

static int pack_void_handle(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED,
			    unsigned node STARPU_ATTRIBUTE_UNUSED,
			    void **ptr,
			    starpu_ssize_t *count)
{
	*count = 0;
	*ptr = NULL;
	return 0;
}

static int unpack_void_handle(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED,
			      unsigned node STARPU_ATTRIBUTE_UNUSED,
			      void *ptr STARPU_ATTRIBUTE_UNUSED,
			      size_t count STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

static size_t void_interface_get_size(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

/* memory allocation/deallocation primitives for the void interface */

/* returns the size of the allocated area */
static starpu_ssize_t allocate_void_buffer_on_node(void *data_interface STARPU_ATTRIBUTE_UNUSED,
						   unsigned dst_node STARPU_ATTRIBUTE_UNUSED)
{
	/* Successfuly allocated 0 bytes */
	return 0;
}

static void free_void_buffer_on_node(void *data_interface STARPU_ATTRIBUTE_UNUSED ,
					unsigned node STARPU_ATTRIBUTE_UNUSED)
{
	/* There is no buffer actually */
}

static int dummy_copy(void *src_interface STARPU_ATTRIBUTE_UNUSED,
			unsigned src_node STARPU_ATTRIBUTE_UNUSED,
			void *dst_interface STARPU_ATTRIBUTE_UNUSED,
			unsigned dst_node STARPU_ATTRIBUTE_UNUSED,
			void *async_data STARPU_ATTRIBUTE_UNUSED)
{
	return 0;
}

static starpu_ssize_t describe(void *data_interface STARPU_ATTRIBUTE_UNUSED, char *buf, size_t size)
{
	return snprintf(buf, size, "0");
}
