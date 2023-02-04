/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <common/utils.h>

static int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data);
static int map_ndim(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
static int unmap_ndim(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
static int update_map_ndim(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
static size_t _get_size(uint32_t* nn, size_t ndim, size_t elemsize);

static const struct starpu_data_copy_methods ndim_copy_data_methods_s =
{
	.any_to_any = copy_any_to_any,
};

static void register_ndim_handle(starpu_data_handle_t handle, int home_node, void *data_interface);
static void unregister_ndim_handle(starpu_data_handle_t handle);
static void *ndim_to_pointer(void *data_interface, unsigned node);
static int ndim_pointer_is_inside(void *data_interface, unsigned node, void *ptr);
static starpu_ssize_t allocate_ndim_buffer_on_node(void *data_interface_, unsigned dst_node);
static void free_ndim_buffer_on_node(void *data_interface, unsigned node);
static void reuse_ndim_buffer_on_node(void *dst_data_interface, const void *cached_interface, unsigned node);
static size_t ndim_interface_get_size(starpu_data_handle_t handle);
static uint32_t footprint_ndim_interface_crc32(starpu_data_handle_t handle);
static int ndim_compare(void *data_interface_a, void *data_interface_b);
static void display_ndim_interface(starpu_data_handle_t handle, FILE *f);
static int pack_ndim_handle(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count);
static int peek_ndim_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);
static int unpack_ndim_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);
static starpu_ssize_t describe(void *data_interface, char *buf, size_t size);
static int pack_meta_ndim_handle(void *data_interface, void **ptr, starpu_ssize_t *count);
static int unpack_meta_ndim_handle(void **data_interface, void *ptr, starpu_ssize_t *count);

struct starpu_data_interface_ops starpu_interface_ndim_ops =
{
	.register_data_handle = register_ndim_handle,
	.unregister_data_handle = unregister_ndim_handle,
	.allocate_data_on_node = allocate_ndim_buffer_on_node,
	.to_pointer = ndim_to_pointer,
	.pointer_is_inside = ndim_pointer_is_inside,
	.free_data_on_node = free_ndim_buffer_on_node,
	.reuse_data_on_node = reuse_ndim_buffer_on_node,
	.map_data = map_ndim,
	.unmap_data = unmap_ndim,
	.update_map = update_map_ndim,
	.copy_methods = &ndim_copy_data_methods_s,
	.get_size = ndim_interface_get_size,
	.footprint = footprint_ndim_interface_crc32,
	.compare = ndim_compare,
	.interfaceid = STARPU_NDIM_INTERFACE_ID,
	.interface_size = sizeof(struct starpu_ndim_interface),
	.display = display_ndim_interface,
	.pack_data = pack_ndim_handle,
	.peek_data = peek_ndim_handle,
	.unpack_data = unpack_ndim_handle,
	.pack_meta = pack_meta_ndim_handle,
	.unpack_meta = unpack_meta_ndim_handle,
	.describe = describe,
	.name = "STARPU_NDIM_INTERFACE",
	.dontcache = 0
};

static void *ndim_to_pointer(void *data_interface, unsigned node)
{
	(void) node;
	struct starpu_ndim_interface *ndim_interface = data_interface;

	return (void*) ndim_interface->ptr;
}

static int ndim_pointer_is_inside(void *data_interface, unsigned node, void *ptr)
{
	(void) node;
	struct starpu_ndim_interface *ndim_interface = data_interface;
	size_t ndim = ndim_interface->ndim;
	uint32_t* ldn = ndim_interface->ldn;
	uint32_t* nn = ndim_interface->nn;
	size_t elemsize = ndim_interface->elemsize;

	if ((char*) ptr < (char*) ndim_interface->ptr)
		return 0;

	size_t offset = ((char*)ptr - (char*)ndim_interface->ptr)/elemsize;

	if(ndim == 0 && offset >= 1)
		return 0;

	int i;
	uint32_t d = offset;
	for (i=ndim-1; i>=0; i--)
	{
		if(d/ldn[i] >= nn[i])
			return 0;
		d = d % ldn[i];
	}

	return 1;
}

static void register_ndim_handle(starpu_data_handle_t handle, int home_node, void *data_interface)
{
	struct starpu_ndim_interface *ndim_interface = (struct starpu_ndim_interface *) data_interface;

	size_t ndim = ndim_interface->ndim;

	int node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_ndim_interface *local_interface = (struct starpu_ndim_interface *)
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
		{
			local_interface->ptr = ndim_interface->ptr;
			local_interface->dev_handle = ndim_interface->dev_handle;
			local_interface->offset = ndim_interface->offset;
			uint32_t* ldn_org = ndim_interface->ldn;
			uint32_t* ldn_cpy;
			_STARPU_MALLOC(ldn_cpy, ndim*sizeof(uint32_t));
			if (ndim)
				memcpy(ldn_cpy, ldn_org, ndim*sizeof(uint32_t));
			local_interface->ldn = ldn_cpy;
		}
		else
		{
			local_interface->ptr = 0;
			local_interface->dev_handle = 0;
			local_interface->offset = 0;
			uint32_t* ldn_zero;
			_STARPU_CALLOC(ldn_zero, ndim, sizeof(uint32_t));
			local_interface->ldn = ldn_zero;
		}

		local_interface->id = ndim_interface->id;
		uint32_t* nn_org = ndim_interface->nn;
		uint32_t* nn_cpy;
		_STARPU_MALLOC(nn_cpy, ndim*sizeof(uint32_t));
		if (ndim)
			memcpy(nn_cpy, nn_org, ndim*sizeof(uint32_t));
		local_interface->nn = nn_cpy;
		local_interface->ndim = ndim_interface->ndim;
		local_interface->elemsize = ndim_interface->elemsize;
		local_interface->allocsize = ndim_interface->allocsize;
	}
}

static void unregister_ndim_handle(starpu_data_handle_t handle)
{
	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_ndim_interface *local_interface = (struct starpu_ndim_interface *) starpu_data_get_interface_on_node(handle, node);

		free(local_interface->nn);
		free(local_interface->ldn);
	}
}

/* declare a new data with the BLAS interface */
void starpu_ndim_data_register(starpu_data_handle_t *handleptr, int home_node,
			       uintptr_t ptr, uint32_t* ldn, uint32_t* nn, size_t ndim, size_t elemsize)
{
	unsigned i;
	size_t allocsize = _get_size(nn, ndim, elemsize);

	for (i=1; i<ndim; i++)
	{
		STARPU_ASSERT_MSG(ldn[i]/ldn[i-1] >= nn[i-1], "ldn[%u]/ldn[%u] = %u/%u = %u should not be less than nn[%u] = %u.", i, i-1, ldn[i], ldn[i-1], ldn[i]/ldn[i-1], i-1, nn[i-1]);
	}

	struct starpu_ndim_interface ndim_interface =
	{
		.id = STARPU_NDIM_INTERFACE_ID,
		.ptr = ptr,
		.dev_handle = ptr,
		.offset = 0,
		.ldn = ldn,
		.nn = nn,
		.ndim = ndim,
		.elemsize = elemsize,
		.allocsize = allocsize,
	};
#ifndef STARPU_SIMGRID
	if (home_node >= 0 && starpu_node_get_kind(home_node) == STARPU_CPU_RAM)
	{
		uint32_t nn0 = ndim?nn[0]:1;
		int b = 1;
		size_t buffersize = 0;
		for (i = 1; i < ndim; i++)
		{
			if (nn[i])
			{
				buffersize += (nn[i]-1)*ldn[i]*elemsize;
			}
			else
			{
				b = 0;
				break;
			}
		}
		buffersize += nn0*elemsize;

		if (b && elemsize)
		{
			STARPU_ASSERT_ACCESSIBLE(ptr);
			STARPU_ASSERT_ACCESSIBLE(ptr + buffersize - 1);
		}
	}
#endif

	starpu_data_register(handleptr, home_node, &ndim_interface, &starpu_interface_ndim_ops);
}

void starpu_ndim_ptr_register(starpu_data_handle_t handle, unsigned node,
			      uintptr_t ptr, uintptr_t dev_handle, size_t offset, uint32_t* ldn)
{
	struct starpu_ndim_interface *ndim_interface = starpu_data_get_interface_on_node(handle, node);
	starpu_data_ptr_register(handle, node);
	ndim_interface->ptr = ptr;
	ndim_interface->dev_handle = dev_handle;
	ndim_interface->offset = offset;
	if (ndim_interface->ndim)
		memcpy(ndim_interface->ldn, ldn, ndim_interface->ndim*sizeof(uint32_t));
}

static uint32_t footprint_ndim_interface_crc32(starpu_data_handle_t handle)
{
	uint32_t hash;

	hash = starpu_hash_crc32c_be(starpu_ndim_get_elemsize(handle), 0);

	unsigned i;
	for (i=0; i<starpu_ndim_get_ndim(handle); i++)
		hash = starpu_hash_crc32c_be(starpu_ndim_get_ni(handle, i), hash);

	return hash;
}

static int ndim_compare(void *data_interface_a, void *data_interface_b)
{
	struct starpu_ndim_interface *ndim_a = (struct starpu_ndim_interface *) data_interface_a;
	struct starpu_ndim_interface *ndim_b = (struct starpu_ndim_interface *) data_interface_b;

	if (ndim_a->ndim != ndim_b->ndim)
		return 0;

	if (ndim_a->elemsize != ndim_b->elemsize)
		return 0;

	unsigned i;
	/* Two matricess are considered compatible if they have the same size */
	for (i=0; i<ndim_a->ndim; i++)
	{
		if (ndim_a->nn[i] != ndim_b->nn[i])
			return 0;
	}

	return 1;
}

static void display_ndim_interface(starpu_data_handle_t handle, FILE *f)
{
	struct starpu_ndim_interface *ndim_interface = (struct starpu_ndim_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	unsigned i;
	for (i=0; i<ndim_interface->ndim; i++)
		fprintf(f, "%u\t", ndim_interface->nn[i]);

	if (ndim_interface->ndim == 0)
	{
		fprintf(f, "%lu\t", (unsigned long)ndim_interface->elemsize);
	}
}

static int _is_contiguous_ndim(uint32_t* nn, uint32_t* ldn, size_t ndim)
{
	if (ndim == 0)
		return 1;

	unsigned i;
	uint32_t ldi = 1;
	for (i = 0; i<ndim-1; i++)
		ldi *= nn[i];

	if (ldi == ldn[ndim-1])
		return 1;
	else
		return 0;
}

static size_t _get_size(uint32_t* nn, size_t ndim, size_t elemsize)
{
	size_t size = elemsize;
	unsigned i;

	for (i=0; i<ndim; i++)
		size *= nn[i];

	return size;
}

static void _pack_cpy_ndim_ptr(char *cur, char* ndptr, uint32_t* nn, uint32_t* ldn, size_t dim, size_t elemsize)
{
	uint32_t i = dim - 1;
	uint32_t n;

	if(_is_contiguous_ndim(nn, ldn, dim))
	{
		memcpy(cur, ndptr, _get_size(nn, dim, elemsize));
	}
	else
	{
		char *ndptr_i = ndptr;
		size_t count = _get_size(nn, i, elemsize);
		for(n=0; n<nn[i]; n++)
		{
			_pack_cpy_ndim_ptr(cur, ndptr_i, nn, ldn, dim-1, elemsize);
			cur += count;
			ndptr_i += ldn[i] * elemsize;
		}
	}
}

static void _peek_cpy_ndim_ptr(char* ndptr, char *cur, uint32_t* nn, uint32_t* ldn, size_t dim, size_t elemsize)
{
	uint32_t i = dim - 1;
	uint32_t n;

	if(_is_contiguous_ndim(nn, ldn, dim))
	{
		memcpy(ndptr, cur, _get_size(nn, dim, elemsize));
	}
	else
	{
		char *ndptr_i = ndptr;
		size_t count = _get_size(nn, i, elemsize);
		for(n=0; n<nn[i]; n++)
		{
			_peek_cpy_ndim_ptr(ndptr_i, cur, nn, ldn, dim-1, elemsize);
			cur += count;
			ndptr_i += ldn[i] * elemsize;
		}
	}
}

static int pack_ndim_handle(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_ndim_interface *ndim_interface = (struct starpu_ndim_interface *)
		starpu_data_get_interface_on_node(handle, node);

	uint32_t* ldn = ndim_interface->ldn;
	uint32_t* nn = ndim_interface->nn;
	size_t ndim = ndim_interface->ndim;
	size_t elemsize = ndim_interface->elemsize;

	*count = _get_size(nn, ndim, elemsize);

	if (ptr != NULL)
	{
		char *ndptr = (void *)ndim_interface->ptr;

		*ptr = (void *)starpu_malloc_on_node_flags(node, *count, 0);

		char *cur = *ptr;

		_pack_cpy_ndim_ptr(cur, ndptr, nn, ldn, ndim, elemsize);
	}

	return 0;
}

static int peek_ndim_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_ndim_interface *ndim_interface = (struct starpu_ndim_interface *)
		starpu_data_get_interface_on_node(handle, node);

	uint32_t* ldn = ndim_interface->ldn;
	uint32_t* nn = ndim_interface->nn;
	size_t ndim = ndim_interface->ndim;
	size_t elemsize = ndim_interface->elemsize;

	STARPU_ASSERT(count == _get_size(nn, ndim, elemsize));

	char *cur = ptr;
	char *ndptr = (void *)ndim_interface->ptr;

	_peek_cpy_ndim_ptr(ndptr, cur, nn, ldn, ndim, elemsize);

	return 0;
}

static int unpack_ndim_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	peek_ndim_handle(handle, node, ptr, count);
	starpu_free_on_node_flags(node, (uintptr_t)ptr, count, 0);

	return 0;
}

static size_t ndim_interface_get_size(starpu_data_handle_t handle)
{
	struct starpu_ndim_interface *ndim_interface;

	ndim_interface = (struct starpu_ndim_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(ndim_interface->id == STARPU_NDIM_INTERFACE_ID, "Error. The given data is not a ndim array.");
#endif

	return _get_size(ndim_interface->nn, ndim_interface->ndim, ndim_interface->elemsize);
}

/* offer an access to the data parameters */
uint32_t* starpu_ndim_get_nn(starpu_data_handle_t handle)
{
	struct starpu_ndim_interface *ndim_interface = (struct starpu_ndim_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(ndim_interface->id == STARPU_NDIM_INTERFACE_ID, "Error. The given data is not a ndim array.");
#endif

	return ndim_interface->nn;
}

uint32_t starpu_ndim_get_ni(starpu_data_handle_t handle, size_t i)
{
	struct starpu_ndim_interface *ndim_interface = (struct starpu_ndim_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	STARPU_ASSERT_MSG(ndim_interface->ndim > 0, "The function can only be called when array dimension is greater than 0.");

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(ndim_interface->id == STARPU_NDIM_INTERFACE_ID, "Error. The given data is not a ndim array.");
#endif

	return ndim_interface->nn[i];
}

uint32_t* starpu_ndim_get_local_ldn(starpu_data_handle_t handle)
{
	unsigned node;
	node = starpu_worker_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_ndim_interface *ndim_interface = (struct starpu_ndim_interface *)
		starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(ndim_interface->id == STARPU_NDIM_INTERFACE_ID, "Error. The given data is not a ndim array.");
#endif

	return ndim_interface->ldn;
}

uint32_t starpu_ndim_get_local_ldi(starpu_data_handle_t handle, size_t i)
{
	unsigned node;
	node = starpu_worker_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_ndim_interface *ndim_interface = (struct starpu_ndim_interface *)
		starpu_data_get_interface_on_node(handle, node);

	STARPU_ASSERT_MSG(ndim_interface->ndim > 0, "The function can only be called when array dimension is greater than 0.");

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(ndim_interface->id == STARPU_NDIM_INTERFACE_ID, "Error. The given data is not a ndim array.");
#endif

	return ndim_interface->ldn[i];
}

uintptr_t starpu_ndim_get_local_ptr(starpu_data_handle_t handle)
{
	unsigned node;
	node = starpu_worker_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_ndim_interface *ndim_interface = (struct starpu_ndim_interface *)
		starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(ndim_interface->id == STARPU_NDIM_INTERFACE_ID, "Error. The given data is not a ndim array.");
#endif

	return ndim_interface->ptr;
}

size_t starpu_ndim_get_ndim(starpu_data_handle_t handle)
{
	struct starpu_ndim_interface *ndim_interface = (struct starpu_ndim_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(ndim_interface->id == STARPU_NDIM_INTERFACE_ID, "Error. The given data is not a ndim array.");
#endif

	return ndim_interface->ndim;
}

size_t starpu_ndim_get_elemsize(starpu_data_handle_t handle)
{
	struct starpu_ndim_interface *ndim_interface = (struct starpu_ndim_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(ndim_interface->id == STARPU_NDIM_INTERFACE_ID, "Error. The given data is not a ndim array.");
#endif

	return ndim_interface->elemsize;
}

/* memory allocation/deallocation primitives for the NDIM interface */

/* For a newly-allocated interface, the ld values are trivial */
static void set_trivial_ndim_ld(struct starpu_ndim_interface *dst_ndarr)
{
	size_t ndim = dst_ndarr->ndim;
	uint32_t* nn = dst_ndarr->nn;

	if (ndim > 0)
	{
		uint32_t ntmp = 1;
		dst_ndarr->ldn[0] = 1;
		size_t i;
		for (i=1; i<ndim; i++)
		{
			ntmp *= nn[i-1];
			dst_ndarr->ldn[i] = ntmp;
		}
	}
}

/* returns the size of the allocated area */
static starpu_ssize_t allocate_ndim_buffer_on_node(void *data_interface_, unsigned dst_node)
{
	uintptr_t addr = 0, handle;

	struct starpu_ndim_interface *dst_ndarr = (struct starpu_ndim_interface *) data_interface_;

	size_t arrsize = dst_ndarr->allocsize;

	handle = starpu_malloc_on_node(dst_node, arrsize);

	if (!handle)
		return -ENOMEM;

	if (starpu_node_get_kind(dst_node) != STARPU_OPENCL_RAM)
		addr = handle;

	/* update the data properly in consequence */
	dst_ndarr->ptr = addr;
	dst_ndarr->dev_handle = handle;
	dst_ndarr->offset = 0;

	set_trivial_ndim_ld(dst_ndarr);

	return arrsize;
}

static void free_ndim_buffer_on_node(void *data_interface, unsigned node)
{
	struct starpu_ndim_interface *ndim_interface = (struct starpu_ndim_interface *) data_interface;

	starpu_free_on_node(node, ndim_interface->dev_handle, ndim_interface->allocsize);
	ndim_interface->ptr = 0;
	ndim_interface->dev_handle = 0;
}

static void reuse_ndim_buffer_on_node(void *dst_data_interface, const void *cached_interface, unsigned node STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_ndim_interface *dst_ndarr = (struct starpu_ndim_interface *) dst_data_interface;
	const struct starpu_ndim_interface *cached_ndarr = (const struct starpu_ndim_interface *) cached_interface;

	dst_ndarr->ptr = cached_ndarr->ptr;
	dst_ndarr->dev_handle = cached_ndarr->dev_handle;
	dst_ndarr->offset = cached_ndarr->offset;

	set_trivial_ndim_ld(dst_ndarr);
}

static size_t _get_mapsize(uint32_t* nn, uint32_t* ldn, size_t ndim, size_t elemsize)
{
	uint32_t nn0 = ndim?nn[0]:1;
	size_t buffersize = 0;
	unsigned i;
	for (i = 1; i < ndim; i++)
	{
		buffersize += ldn[i]*(nn[i]-1)*elemsize;
	}
	buffersize += nn0*elemsize;
	return buffersize;
}

static int map_ndim(void *src_interface, unsigned src_node,
		    void *dst_interface, unsigned dst_node)
{
	struct starpu_ndim_interface *src_ndarr = src_interface;
	struct starpu_ndim_interface *dst_ndarr = dst_interface;
	int ret;
	uintptr_t mapped;

	size_t ndim = src_ndarr->ndim;

	/* map area ldn[ndim-1]*(nn[ndim-1]-1) + ldn[ndim-2]*(nn[ndim-2]-1) + ... + ldn[1]*(nn[1]-1) + nn0*/
	mapped = starpu_interface_map(src_ndarr->dev_handle, src_ndarr->offset, src_node, dst_node, _get_mapsize(src_ndarr->nn, src_ndarr->ldn, ndim, src_ndarr->elemsize), &ret);
	if (mapped)
	{
		dst_ndarr->dev_handle = mapped;
		dst_ndarr->offset = 0;
		if (starpu_node_get_kind(dst_node) != STARPU_OPENCL_RAM)
			dst_ndarr->ptr = mapped;
		size_t i;
		for (i=0; i<ndim; i++)
		{
			dst_ndarr->ldn[i] = src_ndarr->ldn[i];
		}
		return 0;
	}
	return ret;
}

static int unmap_ndim(void *src_interface, unsigned src_node,
		      void *dst_interface, unsigned dst_node)
{
	struct starpu_ndim_interface *src_ndarr = src_interface;
	struct starpu_ndim_interface *dst_ndarr = dst_interface;

	size_t ndim = src_ndarr->ndim;
	int ret = starpu_interface_unmap(src_ndarr->dev_handle, src_ndarr->offset, src_node, dst_ndarr->dev_handle, dst_node, _get_mapsize(src_ndarr->nn, src_ndarr->ldn, ndim, src_ndarr->elemsize));
	dst_ndarr->dev_handle = 0;

	return ret;
}

static int update_map_ndim(void *src_interface, unsigned src_node,
			   void *dst_interface, unsigned dst_node)
{
	struct starpu_ndim_interface *src_ndarr = src_interface;
	struct starpu_ndim_interface *dst_ndarr = dst_interface;

	size_t ndim = src_ndarr->ndim;
	return starpu_interface_update_map(src_ndarr->dev_handle, src_ndarr->offset, src_node, dst_ndarr->dev_handle, dst_ndarr->offset, dst_node, _get_mapsize(src_ndarr->nn, src_ndarr->ldn, ndim, src_ndarr->elemsize));
}

static int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data)
{
	struct starpu_ndim_interface *src_ndarr = (struct starpu_ndim_interface *) src_interface;
	struct starpu_ndim_interface *dst_ndarr = (struct starpu_ndim_interface *) dst_interface;
	int ret = 0;

	uint32_t* nn = dst_ndarr->nn;
	size_t ndim = dst_ndarr->ndim;
	size_t elemsize = dst_ndarr->elemsize;

	uint32_t* ldn_src = src_ndarr->ldn;
	uint32_t* ldn_dst = dst_ndarr->ldn;

	if (starpu_interface_copynd(src_ndarr->dev_handle, src_ndarr->offset, src_node,
				    dst_ndarr->dev_handle, dst_ndarr->offset, dst_node,
				    elemsize, ndim,
				    nn, ldn_src, ldn_dst,
				    async_data))
		ret = -EAGAIN;

	starpu_interface_data_copy(src_node, dst_node, _get_size(nn, ndim, elemsize));

	return ret;
}

static starpu_ssize_t describe(void *data_interface, char *buf, size_t size)
{
	struct starpu_ndim_interface *ndarr = (struct starpu_ndim_interface *) data_interface;

	size_t ndim = ndarr->ndim;
	int n = 0;
	size_t ret;
	unsigned i;
	for (i=0; i<ndim+1; i++)
	{
		ret = snprintf(buf + n, size, "%s%lu", i==0?"N":"x", (unsigned long) (i==ndim?ndarr->elemsize:ndarr->nn[i]));
		n += ret;
		if(size > ret)
			size -= ret;
		else
			size = 0;
	}

	return n;
}

static starpu_ssize_t size_meta_ndim_handle(struct starpu_ndim_interface *ndarr)
{
	starpu_ssize_t count;
	count = sizeof(ndarr->ndim) + sizeof(ndarr->offset) + sizeof(ndarr->allocsize) + sizeof(ndarr->elemsize);
	count += ndarr->ndim * (sizeof(ndarr->ldn[0]) + sizeof(ndarr->nn[0])) + sizeof(ndarr->ptr) + sizeof(ndarr->dev_handle);
	return count;
}


#define _pack(dst, src) do { memcpy(dst, &src, sizeof(src)); dst += sizeof(src); } while (0)

static int pack_meta_ndim_handle(void *data_interface, void **ptr, starpu_ssize_t *count)
{
	struct starpu_ndim_interface *ndarr = (struct starpu_ndim_interface *) data_interface;

	*count = size_meta_ndim_handle(ndarr);
	_STARPU_CALLOC(*ptr, *count, 1);
	char *cur = *ptr;

	_pack(cur, ndarr->ndim);
	_pack(cur, ndarr->offset);
	_pack(cur, ndarr->allocsize);
	_pack(cur, ndarr->elemsize);
	_pack(cur, ndarr->ptr);
	_pack(cur, ndarr->dev_handle);

	memcpy(cur, ndarr->ldn, ndarr->ndim*sizeof(ndarr->ldn[0]));
	cur += ndarr->ndim*sizeof(ndarr->ldn[0]);

	memcpy(cur, ndarr->nn, ndarr->ndim*sizeof(ndarr->nn[0]));
	return 0;
}

#define _unpack(dst, src) do {	memcpy(&dst, src, sizeof(dst)); src += sizeof(dst); } while(0)

static int unpack_meta_ndim_handle(void **data_interface, void *ptr, starpu_ssize_t *count)
{
	_STARPU_CALLOC(*data_interface, 1, sizeof(struct starpu_ndim_interface));
	struct starpu_ndim_interface *ndarr = (struct starpu_ndim_interface *)(*data_interface);
	char *cur = ptr;

	ndarr->id = STARPU_NDIM_INTERFACE_ID;

	_unpack(ndarr->ndim, cur);
	_unpack(ndarr->offset, cur);
	_unpack(ndarr->allocsize, cur);
	_unpack(ndarr->elemsize, cur);
	_unpack(ndarr->ptr, cur);
	_unpack(ndarr->dev_handle, cur);

	_STARPU_MALLOC(ndarr->ldn, ndarr->ndim*sizeof(ndarr->ldn[0]));
	memcpy(ndarr->ldn, cur, ndarr->ndim*sizeof(ndarr->ldn[0]));
	cur += ndarr->ndim*sizeof(ndarr->ldn[0]);

	_STARPU_MALLOC(ndarr->nn, ndarr->ndim*sizeof(ndarr->nn[0]));
	memcpy(ndarr->nn, cur, ndarr->ndim*sizeof(ndarr->nn[0]));

	*count = size_meta_ndim_handle(ndarr);

	return 0;
}
