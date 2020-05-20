/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static const struct starpu_data_copy_methods vector_copy_data_methods_s =
{
	.any_to_any = copy_any_to_any,
};

static void vector_init(void *data_interface);
static void register_vector_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface);
static starpu_ssize_t allocate_vector_buffer_on_node(void *data_interface_, unsigned dst_node);
static void *vector_to_pointer(void *data_interface, unsigned node);
static int vector_pointer_is_inside(void *data_interface, unsigned node, void *ptr);
static void free_vector_buffer_on_node(void *data_interface, unsigned node);
static size_t vector_interface_get_size(starpu_data_handle_t handle);
static size_t vector_interface_get_alloc_size(starpu_data_handle_t handle);
static uint32_t footprint_vector_interface_crc32(starpu_data_handle_t handle);
static uint32_t alloc_footprint_vector_interface_crc32(starpu_data_handle_t handle);
static int vector_compare(void *data_interface_a, void *data_interface_b);
static int vector_alloc_compare(void *data_interface_a, void *data_interface_b);
static void display_vector_interface(starpu_data_handle_t handle, FILE *f);
static int pack_vector_handle(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count);
static int unpack_vector_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);
static starpu_ssize_t describe(void *data_interface, char *buf, size_t size);

struct starpu_data_interface_ops starpu_interface_vector_ops =
{
	.init = vector_init,
	.register_data_handle = register_vector_handle,
	.allocate_data_on_node = allocate_vector_buffer_on_node,
	.to_pointer = vector_to_pointer,
	.pointer_is_inside = vector_pointer_is_inside,
	.free_data_on_node = free_vector_buffer_on_node,
	.copy_methods = &vector_copy_data_methods_s,
	.get_size = vector_interface_get_size,
	.get_alloc_size = vector_interface_get_alloc_size,
	.footprint = footprint_vector_interface_crc32,
	.alloc_footprint = alloc_footprint_vector_interface_crc32,
	.compare = vector_compare,
	.alloc_compare = vector_alloc_compare,
	.interfaceid = STARPU_VECTOR_INTERFACE_ID,
	.interface_size = sizeof(struct starpu_vector_interface),
	.display = display_vector_interface,
	.pack_data = pack_vector_handle,
	.unpack_data = unpack_vector_handle,
	.describe = describe,
	.name = "STARPU_VECTOR_INTERFACE"
};

static void vector_init(void *data_interface)
{
	struct starpu_vector_interface *vector_interface = data_interface;
	vector_interface->allocsize = -1;
}

static void *vector_to_pointer(void *data_interface, unsigned node)
{
	(void) node;
	struct starpu_vector_interface *vector_interface = data_interface;

	return (void*) vector_interface->ptr;
}

static int vector_pointer_is_inside(void *data_interface, unsigned node, void *ptr)
{
	(void) node;
	struct starpu_vector_interface *vector_interface = data_interface;

	return (char*) ptr >= (char*) vector_interface->ptr &&
		(char*) ptr < (char*) vector_interface->ptr + vector_interface->nx*vector_interface->elemsize;
}

static void register_vector_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface)
{
	struct starpu_vector_interface *vector_interface = (struct starpu_vector_interface *) data_interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_vector_interface *local_interface = (struct starpu_vector_interface *)
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
		{
			local_interface->ptr = vector_interface->ptr;
                        local_interface->dev_handle = vector_interface->dev_handle;
                        local_interface->offset = vector_interface->offset;
		}
		else
		{
			local_interface->ptr = 0;
                        local_interface->dev_handle = 0;
                        local_interface->offset = 0;
		}

		local_interface->id = vector_interface->id;
		local_interface->nx = vector_interface->nx;
		local_interface->elemsize = vector_interface->elemsize;
		local_interface->allocsize = vector_interface->allocsize;
		local_interface->slice_base = vector_interface->slice_base;
	}
}

/* declare a new data with the vector interface */
void starpu_vector_data_register_allocsize(starpu_data_handle_t *handleptr, int home_node,
					   uintptr_t ptr, uint32_t nx, size_t elemsize, size_t allocsize)
{
	struct starpu_vector_interface vector =
	{
		.id = STARPU_VECTOR_INTERFACE_ID,
		.ptr = ptr,
		.nx = nx,
		.elemsize = elemsize,
                .dev_handle = ptr,
		.slice_base = 0,
                .offset = 0,
		.allocsize = allocsize,
	};
#if (!defined(STARPU_SIMGRID) && !defined(STARPU_OPENMP))
	if (home_node >= 0 && starpu_node_get_kind(home_node) == STARPU_CPU_RAM)
	{
		if (nx && elemsize)
		{
			STARPU_ASSERT_ACCESSIBLE(ptr);
			STARPU_ASSERT_ACCESSIBLE(ptr + nx*elemsize - 1);
		}
	}
#endif

	starpu_data_register(handleptr, home_node, &vector, &starpu_interface_vector_ops);
}

void starpu_vector_data_register(starpu_data_handle_t *handleptr, int home_node,
				 uintptr_t ptr, uint32_t nx, size_t elemsize)
{
	starpu_vector_data_register_allocsize(handleptr, home_node, ptr, nx, elemsize, nx * elemsize);
}

void starpu_vector_ptr_register(starpu_data_handle_t handle, unsigned node,
			uintptr_t ptr, uintptr_t dev_handle, size_t offset)
{
	struct starpu_vector_interface *vector_interface = starpu_data_get_interface_on_node(handle, node);
	starpu_data_ptr_register(handle, node);
	vector_interface->ptr = ptr;
	vector_interface->dev_handle = dev_handle;
	vector_interface->offset = offset;
}


static uint32_t footprint_vector_interface_crc32(starpu_data_handle_t handle)
{
	return starpu_hash_crc32c_be(starpu_vector_get_nx(handle), 0);
}

static uint32_t alloc_footprint_vector_interface_crc32(starpu_data_handle_t handle)
{
	return starpu_hash_crc32c_be(starpu_vector_get_allocsize(handle), 0);
}

static int vector_compare(void *data_interface_a, void *data_interface_b)
{
	struct starpu_vector_interface *vector_a = (struct starpu_vector_interface *) data_interface_a;
	struct starpu_vector_interface *vector_b = (struct starpu_vector_interface *) data_interface_b;

	/* Two vectors are considered compatible if they have the same size */
	return (vector_a->nx == vector_b->nx)
		&& (vector_a->elemsize == vector_b->elemsize);
}

static int vector_alloc_compare(void *data_interface_a, void *data_interface_b)
{
	struct starpu_vector_interface *vector_a = (struct starpu_vector_interface *) data_interface_a;
	struct starpu_vector_interface *vector_b = (struct starpu_vector_interface *) data_interface_b;

	/* Two vectors are considered allocation-compatible if they have the same size */
	return (vector_a->allocsize == vector_b->allocsize);
}

static void display_vector_interface(starpu_data_handle_t handle, FILE *f)
{
	struct starpu_vector_interface *vector_interface = (struct starpu_vector_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	fprintf(f, "%u\t", vector_interface->nx);
}

static int pack_vector_handle(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_vector_interface *vector_interface = (struct starpu_vector_interface *)
		starpu_data_get_interface_on_node(handle, node);

	*count = vector_interface->nx*vector_interface->elemsize;

	if (ptr != NULL)
	{
		*ptr = (void *)starpu_malloc_on_node_flags(node, *count, 0);
		memcpy(*ptr, (void*)vector_interface->ptr, vector_interface->elemsize*vector_interface->nx);
	}

	return 0;
}

static int unpack_vector_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_vector_interface *vector_interface = (struct starpu_vector_interface *)
		starpu_data_get_interface_on_node(handle, node);

	STARPU_ASSERT(count == vector_interface->elemsize * vector_interface->nx);
	memcpy((void*)vector_interface->ptr, ptr, count);

	starpu_free_on_node_flags(node, (uintptr_t)ptr, count, 0);

	return 0;
}

static size_t vector_interface_get_size(starpu_data_handle_t handle)
{
	size_t size;
	struct starpu_vector_interface *vector_interface = (struct starpu_vector_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(vector_interface->id == STARPU_VECTOR_INTERFACE_ID, "Error. The given data is not a vector.");
#endif

	size = vector_interface->nx * vector_interface->elemsize;

	return size;
}

static size_t vector_interface_get_alloc_size(starpu_data_handle_t handle)
{
	size_t size;
	struct starpu_vector_interface *vector_interface = (struct starpu_vector_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(vector_interface->id == STARPU_VECTOR_INTERFACE_ID, "Error. The given data is not a vector.");
#endif

	size = vector_interface->allocsize;
	STARPU_ASSERT_MSG(size != (size_t)-1, "The vector allocation size needs to be defined");

	return size;
}

/* offer an access to the data parameters */
uint32_t starpu_vector_get_nx(starpu_data_handle_t handle)
{
	struct starpu_vector_interface *vector_interface = (struct starpu_vector_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(vector_interface->id == STARPU_VECTOR_INTERFACE_ID, "Error. The given data is not a vector.");
#endif

	return vector_interface->nx;
}

uintptr_t starpu_vector_get_local_ptr(starpu_data_handle_t handle)
{
	unsigned node;
	node = starpu_worker_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_vector_interface *vector_interface = (struct starpu_vector_interface *)
		starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(vector_interface->id == STARPU_VECTOR_INTERFACE_ID, "Error. The given data is not a vector.");
#endif

	return vector_interface->ptr;
}

size_t starpu_vector_get_elemsize(starpu_data_handle_t handle)
{
	struct starpu_vector_interface *vector_interface = (struct starpu_vector_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(vector_interface->id == STARPU_VECTOR_INTERFACE_ID, "Error. The given data is not a vector.");
#endif

	return vector_interface->elemsize;
}

size_t starpu_vector_get_allocsize(starpu_data_handle_t handle)
{
	struct starpu_vector_interface *vector_interface = (struct starpu_vector_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(vector_interface->id == STARPU_VECTOR_INTERFACE_ID, "Error. The given data is not a vector.");
#endif

	return vector_interface->allocsize;
}

/* memory allocation/deallocation primitives for the vector interface */

/* returns the size of the allocated area */
static starpu_ssize_t allocate_vector_buffer_on_node(void *data_interface_, unsigned dst_node)
{
	uintptr_t addr = 0, handle;

	struct starpu_vector_interface *vector_interface = (struct starpu_vector_interface *) data_interface_;

	starpu_ssize_t allocated_memory = vector_interface->allocsize;
	handle = starpu_malloc_on_node(dst_node, allocated_memory);
	if (!handle)
		return -ENOMEM;

	if (starpu_node_get_kind(dst_node) != STARPU_OPENCL_RAM)
		addr = handle;

	/* update the data properly in consequence */
	vector_interface->ptr = addr;
	vector_interface->dev_handle = handle;
        vector_interface->offset = 0;

	return allocated_memory;
}

static void free_vector_buffer_on_node(void *data_interface, unsigned node)
{
	struct starpu_vector_interface *vector_interface = (struct starpu_vector_interface *) data_interface;

	starpu_free_on_node(node, vector_interface->dev_handle, vector_interface->allocsize);
}

static int copy_any_to_any(void *src_interface, unsigned src_node,
                           void *dst_interface, unsigned dst_node, void *async_data)
{
	struct starpu_vector_interface *src_vector = src_interface;
	struct starpu_vector_interface *dst_vector = dst_interface;
	int ret;

	ret = starpu_interface_copy(src_vector->dev_handle, src_vector->offset, src_node,
				    dst_vector->dev_handle, dst_vector->offset, dst_node,
				    src_vector->nx*src_vector->elemsize, async_data);

	starpu_interface_data_copy(src_node, dst_node, src_vector->nx*src_vector->elemsize);
	return ret;
}

static starpu_ssize_t describe(void *data_interface, char *buf, size_t size)
{
	struct starpu_vector_interface *vector = (struct starpu_vector_interface *) data_interface;
	return snprintf(buf, size, "V%ux%u",
			(unsigned) vector->nx,
			(unsigned) vector->elemsize);
}
