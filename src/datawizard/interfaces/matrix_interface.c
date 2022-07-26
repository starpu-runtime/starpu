/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#ifdef BUILDING_STARPU
#include <datawizard/memory_nodes.h>
#endif

static int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data);
static int map_matrix(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
static int unmap_matrix(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
static int update_map_matrix(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

static const struct starpu_data_copy_methods matrix_copy_data_methods_s =
{
	.any_to_any = copy_any_to_any,
};

static void matrix_init(void *data_interface);
static void register_matrix_handle(starpu_data_handle_t handle, int home_node, void *data_interface);
static void *matrix_to_pointer(void *data_interface, unsigned node);
static int matrix_pointer_is_inside(void *data_interface, unsigned node, void *ptr);
static starpu_ssize_t allocate_matrix_buffer_on_node(void *data_interface_, unsigned dst_node);
static void free_matrix_buffer_on_node(void *data_interface, unsigned node);
static void reuse_matrix_buffer_on_node(void *data_interface, const void *new_data_interface, unsigned node);
static size_t matrix_interface_get_size(starpu_data_handle_t handle);
static size_t matrix_interface_get_alloc_size(starpu_data_handle_t handle);
static uint32_t footprint_matrix_interface_crc32(starpu_data_handle_t handle);
static uint32_t alloc_footprint_matrix_interface_crc32(starpu_data_handle_t handle);
static int matrix_compare(void *data_interface_a, void *data_interface_b);
static int matrix_alloc_compare(void *data_interface_a, void *data_interface_b);
static void display_matrix_interface(starpu_data_handle_t handle, FILE *f);
static int pack_matrix_handle(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count);
static int peek_matrix_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);
static int unpack_matrix_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);
static starpu_ssize_t describe(void *data_interface, char *buf, size_t size);

struct starpu_data_interface_ops starpu_interface_matrix_ops =
{
	.init = matrix_init,
	.register_data_handle = register_matrix_handle,
	.allocate_data_on_node = allocate_matrix_buffer_on_node,
	.to_pointer = matrix_to_pointer,
	.pointer_is_inside = matrix_pointer_is_inside,
	.free_data_on_node = free_matrix_buffer_on_node,
	.reuse_data_on_node = reuse_matrix_buffer_on_node,
	.map_data = map_matrix,
	.unmap_data = unmap_matrix,
	.update_map = update_map_matrix,
	.copy_methods = &matrix_copy_data_methods_s,
	.get_size = matrix_interface_get_size,
	.get_alloc_size = matrix_interface_get_alloc_size,
	.footprint = footprint_matrix_interface_crc32,
	.alloc_footprint = alloc_footprint_matrix_interface_crc32,
	.compare = matrix_compare,
	.alloc_compare = matrix_alloc_compare,
	.interfaceid = STARPU_MATRIX_INTERFACE_ID,
	.interface_size = sizeof(struct starpu_matrix_interface),
	.display = display_matrix_interface,
	.pack_data = pack_matrix_handle,
	.peek_data = peek_matrix_handle,
	.unpack_data = unpack_matrix_handle,
	.describe = describe,
	.name = "STARPU_MATRIX_INTERFACE"
};

static void matrix_init(void *data_interface)
{
	struct starpu_matrix_interface *matrix_interface = data_interface;
	matrix_interface->allocsize = -1;
}

static void register_matrix_handle(starpu_data_handle_t handle, int home_node, void *data_interface)
{
	struct starpu_matrix_interface *matrix_interface = (struct starpu_matrix_interface *) data_interface;

	int node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_matrix_interface *local_interface = (struct starpu_matrix_interface *)
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
		{
			local_interface->ptr = matrix_interface->ptr;
			local_interface->dev_handle = matrix_interface->dev_handle;
			local_interface->offset = matrix_interface->offset;
			local_interface->ld  = matrix_interface->ld;
		}
		else
		{
			local_interface->ptr = 0;
			local_interface->dev_handle = 0;
			local_interface->offset = 0;
			local_interface->ld  = 0;
		}

		local_interface->id = matrix_interface->id;
		local_interface->nx = matrix_interface->nx;
		local_interface->ny = matrix_interface->ny;
		local_interface->elemsize = matrix_interface->elemsize;
		local_interface->allocsize  = matrix_interface->allocsize;
	}
}

static void *matrix_to_pointer(void *data_interface, unsigned node)
{
	(void) node;
	struct starpu_matrix_interface *matrix_interface = data_interface;

	return (void*) matrix_interface->ptr;
}

static int matrix_pointer_is_inside(void *data_interface, unsigned node, void *ptr)
{
	(void) node;
	struct starpu_matrix_interface *matrix_interface = data_interface;
	uint32_t ld = matrix_interface->ld;
	uint32_t nx = matrix_interface->nx;
	uint32_t ny = matrix_interface->ny;
	size_t elemsize = matrix_interface->elemsize;

	if ((char*) ptr < (char*) matrix_interface->ptr)
		return 0;

	size_t offset = ((char*)ptr - (char*)matrix_interface->ptr)/elemsize;

	if(offset/ld >= ny)
		return 0;
	if(offset%ld >= nx)
		return 0;

	return 1;
}


/* declare a new data with the matrix interface */
void starpu_matrix_data_register_allocsize(starpu_data_handle_t *handleptr, int home_node,
					   uintptr_t ptr, uint32_t ld, uint32_t nx,
					   uint32_t ny, size_t elemsize, size_t allocsize)
{
	STARPU_ASSERT_MSG(ld >= nx, "ld = %u should not be less than nx = %u.", ld, nx);
	struct starpu_matrix_interface matrix_interface =
	{
		.id = STARPU_MATRIX_INTERFACE_ID,
		.ptr = ptr,
		.ld = ld,
		.nx = nx,
		.ny = ny,
		.elemsize = elemsize,
		.dev_handle = ptr,
		.offset = 0,
		.allocsize = allocsize,
	};
#ifndef STARPU_SIMGRID
	if (home_node >= 0 && starpu_node_get_kind(home_node) == STARPU_CPU_RAM)
	{
		if (nx && ny && elemsize)
		{
			STARPU_ASSERT_ACCESSIBLE(ptr);
			STARPU_ASSERT_ACCESSIBLE(ptr + (ny-1)*ld*elemsize + nx*elemsize - 1);
		}
	}
#endif

	starpu_data_register(handleptr, home_node, &matrix_interface, &starpu_interface_matrix_ops);
}

void starpu_matrix_data_register(starpu_data_handle_t *handleptr, int home_node,
				 uintptr_t ptr, uint32_t ld, uint32_t nx,
				 uint32_t ny, size_t elemsize)
{
	starpu_matrix_data_register_allocsize(handleptr, home_node, ptr, ld, nx, ny, elemsize, nx * ny * elemsize);
}

void starpu_matrix_ptr_register(starpu_data_handle_t handle, unsigned node,
				uintptr_t ptr, uintptr_t dev_handle, size_t offset, uint32_t ld)
{
	struct starpu_matrix_interface *matrix_interface = starpu_data_get_interface_on_node(handle, node);
	starpu_data_ptr_register(handle, node);
	matrix_interface->ptr = ptr;
	matrix_interface->dev_handle = dev_handle;
	matrix_interface->offset = offset;
	matrix_interface->ld = ld;
}

static uint32_t footprint_matrix_interface_crc32(starpu_data_handle_t handle)
{
	return starpu_hash_crc32c_be(starpu_matrix_get_nx(handle), starpu_matrix_get_ny(handle));
}

static uint32_t alloc_footprint_matrix_interface_crc32(starpu_data_handle_t handle)
{
	return starpu_hash_crc32c_be(starpu_matrix_get_allocsize(handle), 0);
}

static int matrix_compare(void *data_interface_a, void *data_interface_b)
{
	struct starpu_matrix_interface *matrix_a = (struct starpu_matrix_interface *) data_interface_a;
	struct starpu_matrix_interface *matrix_b = (struct starpu_matrix_interface *) data_interface_b;

	/* Two matricess are considered compatible if they have the same size */
	return (matrix_a->nx == matrix_b->nx)
		&& (matrix_a->ny == matrix_b->ny)
		&& (matrix_a->elemsize == matrix_b->elemsize);
}

static int matrix_alloc_compare(void *data_interface_a, void *data_interface_b)
{
	struct starpu_matrix_interface *matrix_a = (struct starpu_matrix_interface *) data_interface_a;
	struct starpu_matrix_interface *matrix_b = (struct starpu_matrix_interface *) data_interface_b;

	/* Two matricess are considered allocation-compatible if they have the same size */
	return (matrix_a->allocsize == matrix_b->allocsize);
}

static void display_matrix_interface(starpu_data_handle_t handle, FILE *f)
{
	struct starpu_matrix_interface *matrix_interface = (struct starpu_matrix_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	fprintf(f, "%u\t%u\t", matrix_interface->nx, matrix_interface->ny);
}

#define IS_CONTIGUOUS_MATRIX(nx, ny, ld) ((nx) == (ld))

//#define DYNAMIC_MATRICES

struct pack_matrix_header
{
#ifdef DYNAMIC_MATRICES
	/* Receiving matrices with different sizes from MPI */
	/* FIXME: that would break alignment for O_DIRECT disk access...
	 * while in the disk case, we do know the matrix size anyway */
	uint32_t nx;
	uint32_t ny;
	size_t elemsize;
#endif
};

static int pack_matrix_handle(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_matrix_interface *matrix_interface = (struct starpu_matrix_interface *)
		starpu_data_get_interface_on_node(handle, node);

	uint32_t ld = matrix_interface->ld;
	uint32_t nx = matrix_interface->nx;
	uint32_t ny = matrix_interface->ny;
	size_t elemsize = matrix_interface->elemsize;

	*count = nx*ny*elemsize + sizeof(struct pack_matrix_header);

	if (ptr != NULL)
	{
		char *matrix = (void *)matrix_interface->ptr;

		*ptr = (void *)starpu_malloc_on_node_flags(node, *count, 0);

		struct pack_matrix_header *header = *ptr;
#ifdef DYNAMIC_MATRICES
		header->nx = nx;
		header->ny = ny;
		header->elemsize = elemsize;
#endif

		char *cur = (char*) *ptr + sizeof(*header);

		if (IS_CONTIGUOUS_MATRIX(nx, ny, ld))
			memcpy(cur, matrix, nx*ny*elemsize);
		else
		{
			uint32_t y;
			for(y=0 ; y<ny ; y++)
			{
				memcpy(cur, matrix, nx*elemsize);
				cur += nx*elemsize;
				matrix += ld * elemsize;
			}
		}
	}

	return 0;
}

static int peek_matrix_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_matrix_interface *matrix_interface = (struct starpu_matrix_interface *)
		starpu_data_get_interface_on_node(handle, node);

	uint32_t ld = matrix_interface->ld;
	uint32_t nx = matrix_interface->nx;
	uint32_t ny = matrix_interface->ny;
	size_t elemsize = matrix_interface->elemsize;

	struct pack_matrix_header *header = ptr;

#ifdef DYNAMIC_MATRICES
	STARPU_ASSERT(count >= sizeof(*header));

	if (IS_CONTIGUOUS_MATRIX(nx, ny, ld))
	{
		/* We can store whatever can fit */

		STARPU_ASSERT_MSG(header->elemsize == elemsize,
				"Data element size %u needs to be same as the received data element size %u",
				(unsigned) elemsize, (unsigned) header->elemsize);

		STARPU_ASSERT_MSG(header->nx * header->ny * header->elemsize <= matrix_interface->allocsize,
				"Initial size of data %lu needs to be big enough for received data %ux%ux%u",
				(unsigned long) matrix_interface->allocsize,
				(unsigned) header->nx, (unsigned) header->ny,
				(unsigned) header->elemsize);

		/* Better keep it contiguous */
		matrix_interface->ld = ld = header->nx;
	}
	else
	{
		STARPU_ASSERT_MSG(header->nx <= nx,
				"Initial nx %u of data needs to be big enough for received data nx %u\n",
				nx, header->nx);
		STARPU_ASSERT_MSG(header->ny <= ny,
				"Initial ny %u of data needs to be big enough for received data ny %u\n",
				ny, header->ny);
	}

	matrix_interface->nx = nx = header->nx;
	matrix_interface->ny = ny = header->ny;
#endif

	char *cur = (char*) ptr + sizeof(*header);

	STARPU_ASSERT(count == sizeof(*header) + elemsize * nx * ny);

	char *matrix = (void *)matrix_interface->ptr;

	if (IS_CONTIGUOUS_MATRIX(nx, ny, ld))
		memcpy(matrix, ptr, nx*ny*elemsize);
	else
	{
		uint32_t y;
		for(y=0 ; y<ny ; y++)
		{
			memcpy(matrix, cur, nx*elemsize);
			cur += nx*elemsize;
			matrix += ld * elemsize;
		}
	}

	return 0;
}

static int unpack_matrix_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	peek_matrix_handle(handle, node, ptr, count);
	starpu_free_on_node_flags(node, (uintptr_t)ptr, count, 0);

	return 0;
}

static size_t matrix_interface_get_size(starpu_data_handle_t handle)
{
	struct starpu_matrix_interface *matrix_interface = (struct starpu_matrix_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(matrix_interface->id == STARPU_MATRIX_INTERFACE_ID, "Error. The given data is not a matrix.");
#endif

	return matrix_interface->nx * matrix_interface->ny * matrix_interface->elemsize;
}

static size_t matrix_interface_get_alloc_size(starpu_data_handle_t handle)
{
	struct starpu_matrix_interface *matrix_interface = (struct starpu_matrix_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(matrix_interface->id == STARPU_MATRIX_INTERFACE_ID, "Error. The given data is not a matrix.");
#endif

	STARPU_ASSERT_MSG(matrix_interface->allocsize != (size_t)-1, "The matrix allocation size needs to be defined");

	return matrix_interface->allocsize;
}

/* offer an access to the data parameters */
uint32_t starpu_matrix_get_nx(starpu_data_handle_t handle)
{
	struct starpu_matrix_interface *matrix_interface = (struct starpu_matrix_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(matrix_interface->id == STARPU_MATRIX_INTERFACE_ID, "Error. The given data is not a matrix.");
#endif

	return matrix_interface->nx;
}

uint32_t starpu_matrix_get_ny(starpu_data_handle_t handle)
{
	struct starpu_matrix_interface *matrix_interface = (struct starpu_matrix_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(matrix_interface->id == STARPU_MATRIX_INTERFACE_ID, "Error. The given data is not a matrix.");
#endif

	return matrix_interface->ny;
}

uint32_t starpu_matrix_get_local_ld(starpu_data_handle_t handle)
{
	unsigned node;
	node = starpu_worker_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_matrix_interface *matrix_interface = (struct starpu_matrix_interface *)
		starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(matrix_interface->id == STARPU_MATRIX_INTERFACE_ID, "Error. The given data is not a matrix.");
#endif

	return matrix_interface->ld;
}

uintptr_t starpu_matrix_get_local_ptr(starpu_data_handle_t handle)
{
	unsigned node;
	node = starpu_worker_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_matrix_interface *matrix_interface = (struct starpu_matrix_interface *)
		starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(matrix_interface->id == STARPU_MATRIX_INTERFACE_ID, "Error. The given data is not a matrix.");
#endif

	return matrix_interface->ptr;
}

size_t starpu_matrix_get_elemsize(starpu_data_handle_t handle)
{
	struct starpu_matrix_interface *matrix_interface = (struct starpu_matrix_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(matrix_interface->id == STARPU_MATRIX_INTERFACE_ID, "Error. The given data is not a matrix.");
#endif

	return matrix_interface->elemsize;
}

size_t starpu_matrix_get_allocsize(starpu_data_handle_t handle)
{
	struct starpu_matrix_interface *matrix_interface = (struct starpu_matrix_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(matrix_interface->id == STARPU_MATRIX_INTERFACE_ID, "Error. The given data is not a matrix.");
#endif

	return matrix_interface->allocsize;
}

/* memory allocation/deallocation primitives for the matrix interface */

/* returns the size of the allocated area */
static starpu_ssize_t allocate_matrix_buffer_on_node(void *data_interface_, unsigned dst_node)
{
	uintptr_t addr = 0, handle;

	struct starpu_matrix_interface *matrix_interface = (struct starpu_matrix_interface *) data_interface_;

	uint32_t ld = matrix_interface->nx; // by default

	starpu_ssize_t allocated_memory = matrix_interface->allocsize;
	handle = starpu_malloc_on_node(dst_node, allocated_memory);

	if (!handle)
		return -ENOMEM;

	if (starpu_node_get_kind(dst_node) != STARPU_OPENCL_RAM)
		addr = handle;

	/* update the data properly in consequence */
	matrix_interface->ptr = addr;
	matrix_interface->dev_handle = handle;
	matrix_interface->offset = 0;
	matrix_interface->ld = ld;

	return allocated_memory;
}

static void free_matrix_buffer_on_node(void *data_interface, unsigned node)
{
	struct starpu_matrix_interface *matrix_interface = (struct starpu_matrix_interface *) data_interface;

	starpu_free_on_node(node, matrix_interface->dev_handle, matrix_interface->allocsize);
}

static void reuse_matrix_buffer_on_node(void *dst_data_interface, const void *cached_interface, unsigned node STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_matrix_interface *dst_matrix_interface = dst_data_interface;
	const struct starpu_matrix_interface *cached_matrix_interface = cached_interface;

	dst_matrix_interface->ptr = cached_matrix_interface->ptr;
	dst_matrix_interface->dev_handle = cached_matrix_interface->dev_handle;
	dst_matrix_interface->offset = cached_matrix_interface->offset;
	dst_matrix_interface->ld = dst_matrix_interface->nx; // by default
}

static int map_matrix(void *src_interface, unsigned src_node,
		      void *dst_interface, unsigned dst_node)
{
	struct starpu_matrix_interface *src_matrix = src_interface;
	struct starpu_matrix_interface *dst_matrix = dst_interface;
	int ret;
	uintptr_t mapped;

	/* map area ld*(ny-1)+nx */
	mapped = starpu_interface_map(src_matrix->dev_handle, src_matrix->offset, src_node, dst_node, (src_matrix->ld*(src_matrix->ny-1)+src_matrix->nx)*src_matrix->elemsize, &ret);
	if (mapped)
	{
		dst_matrix->dev_handle = mapped;
		dst_matrix->offset = 0;
		if (starpu_node_get_kind(dst_node) != STARPU_OPENCL_RAM)
			dst_matrix->ptr = mapped;
		dst_matrix->ld = src_matrix->ld;
		return 0;
	}
	return ret;
}

static int unmap_matrix(void *src_interface, unsigned src_node,
			void *dst_interface, unsigned dst_node)
{
	struct starpu_matrix_interface *src_matrix = src_interface;
	struct starpu_matrix_interface *dst_matrix = dst_interface;

	int ret = starpu_interface_unmap(src_matrix->dev_handle, src_matrix->offset, src_node, dst_matrix->dev_handle, dst_node, (src_matrix->ld*(src_matrix->ny-1)+src_matrix->nx)*src_matrix->elemsize);
	dst_matrix->dev_handle = 0;

	return ret;
}

static int update_map_matrix(void *src_interface, unsigned src_node,
			     void *dst_interface, unsigned dst_node)
{
	struct starpu_matrix_interface *src_matrix = src_interface;
	struct starpu_matrix_interface *dst_matrix = dst_interface;

	return starpu_interface_update_map(src_matrix->dev_handle, src_matrix->offset, src_node, dst_matrix->dev_handle, dst_matrix->offset, dst_node, (src_matrix->ld*(src_matrix->ny-1)+src_matrix->nx)*src_matrix->elemsize);
}

static int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data)
{
	struct starpu_matrix_interface *src_matrix = (struct starpu_matrix_interface *) src_interface;
	struct starpu_matrix_interface *dst_matrix = (struct starpu_matrix_interface *) dst_interface;
	int ret = 0;

	uint32_t nx = dst_matrix->nx;
	uint32_t ny = dst_matrix->ny;
	size_t elemsize = dst_matrix->elemsize;

	uint32_t ld_src = src_matrix->ld;
	uint32_t ld_dst = dst_matrix->ld;

	if (starpu_interface_copy2d(src_matrix->dev_handle, src_matrix->offset, src_node,
				    dst_matrix->dev_handle, dst_matrix->offset, dst_node,
				    nx * elemsize,
				    ny, ld_src * elemsize, ld_dst * elemsize,
				    async_data))
		ret = -EAGAIN;

	starpu_interface_data_copy(src_node, dst_node, (size_t)nx*ny*elemsize);

	return ret;
}

static starpu_ssize_t describe(void *data_interface, char *buf, size_t size)
{
	struct starpu_matrix_interface *matrix = (struct starpu_matrix_interface *) data_interface;
	return snprintf(buf, size, "M%ux%ux%u",
			(unsigned) matrix->nx,
			(unsigned) matrix->ny,
			(unsigned) matrix->elemsize);
}
