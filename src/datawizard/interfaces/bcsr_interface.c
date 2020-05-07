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

/*
 * BCSR : blocked CSR, we use blocks of size (r x c)
 */

static int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data);

static const struct starpu_data_copy_methods bcsr_copy_data_methods_s =
{
	.any_to_any = copy_any_to_any,
};

static void register_bcsr_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface);
static void *bcsr_to_pointer(void *data_interface, unsigned node);
static int bcsr_pointer_is_inside(void *data_interface, unsigned node, void *ptr);
static starpu_ssize_t allocate_bcsr_buffer_on_node(void *data_interface, unsigned dst_node);
static void free_bcsr_buffer_on_node(void *data_interface, unsigned node);
static size_t bcsr_interface_get_size(starpu_data_handle_t handle);
static int bcsr_compare(void *data_interface_a, void *data_interface_b);
static uint32_t footprint_bcsr_interface_crc32(starpu_data_handle_t handle);
static starpu_ssize_t describe(void *data_interface, char *buf, size_t size);
static int pack_data(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count);
static int unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);

struct starpu_data_interface_ops starpu_interface_bcsr_ops =
{
	.register_data_handle = register_bcsr_handle,
	.allocate_data_on_node = allocate_bcsr_buffer_on_node,
	.free_data_on_node = free_bcsr_buffer_on_node,
	.copy_methods = &bcsr_copy_data_methods_s,
	.get_size = bcsr_interface_get_size,
	.interfaceid = STARPU_BCSR_INTERFACE_ID,
	.interface_size = sizeof(struct starpu_bcsr_interface),
	.footprint = footprint_bcsr_interface_crc32,
	.compare = bcsr_compare,
	.describe = describe,
	.to_pointer = bcsr_to_pointer,
	.pointer_is_inside = bcsr_pointer_is_inside,
	.name = "STARPU_BCSR_INTERFACE",
	.pack_data = pack_data,
	.unpack_data = unpack_data
};

static void *bcsr_to_pointer(void *data_interface, unsigned node)
{
	(void) node;
	struct starpu_bcsr_interface *bcsr_interface = data_interface;

	return (void*) bcsr_interface->nzval;
}

static int bcsr_pointer_is_inside(void *data_interface, unsigned node, void *ptr)
{
	(void) node;
	struct starpu_bcsr_interface *bcsr_interface = data_interface;

	return ((char*) ptr >= (char*) bcsr_interface->nzval &&
		(char*) ptr < (char*) bcsr_interface->nzval + bcsr_interface->nnz*bcsr_interface->r*bcsr_interface->c*bcsr_interface->elemsize)
	    || ((char*) ptr >= (char*) bcsr_interface->colind &&
		(char*) ptr < (char*) bcsr_interface->colind + bcsr_interface->nnz*sizeof(uint32_t))
	    || ((char*) ptr >= (char*) bcsr_interface->rowptr &&
		(char*) ptr < (char*) bcsr_interface->rowptr + (bcsr_interface->nrow+1)*sizeof(uint32_t));
}

static void register_bcsr_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface)
{
	struct starpu_bcsr_interface *bcsr_interface = (struct starpu_bcsr_interface *) data_interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_bcsr_interface *local_interface = (struct starpu_bcsr_interface *)
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
		{
			local_interface->nzval = bcsr_interface->nzval;
			local_interface->colind = bcsr_interface->colind;
			local_interface->rowptr = bcsr_interface->rowptr;
		}
		else
		{
			local_interface->nzval = 0;
			local_interface->colind = NULL;
			local_interface->rowptr = NULL;
		}

		local_interface->id = bcsr_interface->id;
		local_interface->nnz = bcsr_interface->nnz;
		local_interface->nrow = bcsr_interface->nrow;
		local_interface->firstentry = bcsr_interface->firstentry;
		local_interface->r = bcsr_interface->r;
		local_interface->c = bcsr_interface->c;
		local_interface->elemsize = bcsr_interface->elemsize;
	}
}

void starpu_bcsr_data_register(starpu_data_handle_t *handleptr, int home_node,
		uint32_t nnz, uint32_t nrow, uintptr_t nzval, uint32_t *colind,
		uint32_t *rowptr, uint32_t firstentry,
		uint32_t r, uint32_t c, size_t elemsize)
{
	struct starpu_bcsr_interface bcsr_interface =
	{
		.id = STARPU_BCSR_INTERFACE_ID,
		.nzval = nzval,
		.colind = colind,
		.rowptr = rowptr,
		.nnz = nnz,
		.nrow = nrow,
		.firstentry = firstentry,
		.r = r,
		.c = c,
		.elemsize = elemsize
	};
#ifndef STARPU_SIMGRID
	if (home_node >= 0 && starpu_node_get_kind(home_node) == STARPU_CPU_RAM)
	{
		if (nnz)
		{
			if (r && c && elemsize)
			{
				STARPU_ASSERT_ACCESSIBLE(nzval);
				STARPU_ASSERT_ACCESSIBLE(nzval + nnz*elemsize*r*c - 1);
			}
			STARPU_ASSERT_ACCESSIBLE(colind);
			STARPU_ASSERT_ACCESSIBLE((uintptr_t) colind + nnz*sizeof(uint32_t) - 1);
		}
		STARPU_ASSERT_ACCESSIBLE(rowptr);
		STARPU_ASSERT_ACCESSIBLE((uintptr_t) rowptr + (nrow+1)*sizeof(uint32_t) - 1);
	}
#endif

	starpu_data_register(handleptr, home_node, &bcsr_interface, &starpu_interface_bcsr_ops);
}

static uint32_t footprint_bcsr_interface_crc32(starpu_data_handle_t handle)
{
	uint32_t hash;

	hash = starpu_hash_crc32c_be(starpu_bcsr_get_nnz(handle), 0);
	hash = starpu_hash_crc32c_be(starpu_bcsr_get_c(handle), hash);
	hash = starpu_hash_crc32c_be(starpu_bcsr_get_r(handle), hash);

	return hash;
}

static int bcsr_compare(void *data_interface_a, void *data_interface_b)
{
	struct starpu_bcsr_interface *bcsr_a = (struct starpu_bcsr_interface *) data_interface_a;
	struct starpu_bcsr_interface *bcsr_b = (struct starpu_bcsr_interface *) data_interface_b;

	/* Two matricess are considered compatible if they have the same size */
	return (bcsr_a->nnz == bcsr_b->nnz)
		&& (bcsr_a->nrow == bcsr_b->nrow)
		&& (bcsr_a->r == bcsr_b->r)
		&& (bcsr_a->c == bcsr_b->c)
		&& (bcsr_a->elemsize == bcsr_b->elemsize);
}

/* offer an access to the data parameters */
uint32_t starpu_bcsr_get_nnz(starpu_data_handle_t handle)
{
	struct starpu_bcsr_interface *data_interface = (struct starpu_bcsr_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(data_interface->id == STARPU_BCSR_INTERFACE_ID, "Error. The given data is not a bcsr.");
#endif

	return data_interface->nnz;
}

uint32_t starpu_bcsr_get_nrow(starpu_data_handle_t handle)
{
	struct starpu_bcsr_interface *data_interface = (struct starpu_bcsr_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(data_interface->id == STARPU_BCSR_INTERFACE_ID, "Error. The given data is not a bcsr.");
#endif

	return data_interface->nrow;
}

uint32_t starpu_bcsr_get_firstentry(starpu_data_handle_t handle)
{
	struct starpu_bcsr_interface *data_interface = (struct starpu_bcsr_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(data_interface->id == STARPU_BCSR_INTERFACE_ID, "Error. The given data is not a bcsr.");
#endif

	return data_interface->firstentry;
}

uint32_t starpu_bcsr_get_r(starpu_data_handle_t handle)
{
	struct starpu_bcsr_interface *data_interface = (struct starpu_bcsr_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(data_interface->id == STARPU_BCSR_INTERFACE_ID, "Error. The given data is not a bcsr.");
#endif

	return data_interface->r;
}

uint32_t starpu_bcsr_get_c(starpu_data_handle_t handle)
{
	struct starpu_bcsr_interface *data_interface = (struct starpu_bcsr_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(data_interface->id == STARPU_BCSR_INTERFACE_ID, "Error. The given data is not a bcsr.");
#endif

	return data_interface->c;
}

size_t starpu_bcsr_get_elemsize(starpu_data_handle_t handle)
{
	struct starpu_bcsr_interface *data_interface = (struct starpu_bcsr_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(data_interface->id == STARPU_BCSR_INTERFACE_ID, "Error. The given data is not a bcsr.");
#endif

	return data_interface->elemsize;
}

uintptr_t starpu_bcsr_get_local_nzval(starpu_data_handle_t handle)
{
	unsigned node;
	node = starpu_worker_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_bcsr_interface *data_interface = (struct starpu_bcsr_interface *)
		starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(data_interface->id == STARPU_BCSR_INTERFACE_ID, "Error. The given data is not a bcsr.");
#endif

	return data_interface->nzval;
}

uint32_t *starpu_bcsr_get_local_colind(starpu_data_handle_t handle)
{
	int node;
	node = starpu_worker_get_local_memory_node();

	/* XXX 0 */
	struct starpu_bcsr_interface *data_interface = (struct starpu_bcsr_interface *)
		starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(data_interface->id == STARPU_BCSR_INTERFACE_ID, "Error. The given data is not a bcsr.");
#endif

	return data_interface->colind;
}

uint32_t *starpu_bcsr_get_local_rowptr(starpu_data_handle_t handle)
{
	int node;
	node = starpu_worker_get_local_memory_node();

	/* XXX 0 */
	struct starpu_bcsr_interface *data_interface = (struct starpu_bcsr_interface *)
		starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(data_interface->id == STARPU_BCSR_INTERFACE_ID, "Error. The given data is not a bcsr.");
#endif

	return data_interface->rowptr;
}

static size_t bcsr_interface_get_size(starpu_data_handle_t handle)
{
	size_t size;

	uint32_t nnz = starpu_bcsr_get_nnz(handle);
	uint32_t nrow = starpu_bcsr_get_nrow(handle);
	uint32_t r = starpu_bcsr_get_r(handle);
	uint32_t c = starpu_bcsr_get_c(handle);
	size_t elemsize = starpu_bcsr_get_elemsize(handle);

	size = nnz*r*c*elemsize + nnz*sizeof(uint32_t) + (nrow+1)*sizeof(uint32_t);

	return size;
}


/* memory allocation/deallocation primitives for the BLAS interface */

/* returns the size of the allocated area */
static starpu_ssize_t allocate_bcsr_buffer_on_node(void *data_interface_, unsigned dst_node)
{
	uintptr_t addr_nzval, addr_colind, addr_rowptr;
	starpu_ssize_t allocated_memory;

	/* we need the 3 arrays to be allocated */
	struct starpu_bcsr_interface *bcsr_interface = (struct starpu_bcsr_interface *) data_interface_;

	uint32_t nnz = bcsr_interface->nnz;
	uint32_t nrow = bcsr_interface->nrow;
	size_t elemsize = bcsr_interface->elemsize;

	uint32_t r = bcsr_interface->r;
	uint32_t c = bcsr_interface->c;

	STARPU_ASSERT_MSG(r && c, "partitioning bcsr with several memory nodes is not supported yet");

	if (nnz)
	{
		addr_nzval = starpu_malloc_on_node(dst_node, nnz*r*c*elemsize);
		if (!addr_nzval)
			goto fail_nzval;
		addr_colind = starpu_malloc_on_node(dst_node, nnz*sizeof(uint32_t));
		if (!addr_colind)
			goto fail_colind;
	}
	else
	{
		addr_nzval = addr_colind = 0;
	}
	addr_rowptr = starpu_malloc_on_node(dst_node, (nrow+1)*sizeof(uint32_t));
	if (!addr_rowptr)
		goto fail_rowptr;

	/* allocation succeeded */
	allocated_memory =
		nnz*r*c*elemsize + nnz*sizeof(uint32_t) + (nrow+1)*sizeof(uint32_t);

	/* update the data properly in consequence */
	bcsr_interface->nzval = addr_nzval;
	bcsr_interface->colind = (uint32_t*) addr_colind;
	bcsr_interface->rowptr = (uint32_t*) addr_rowptr;

	return allocated_memory;

fail_rowptr:
	if (nnz)
		starpu_free_on_node(dst_node, addr_colind, nnz*sizeof(uint32_t));
fail_colind:
	if (nnz)
		starpu_free_on_node(dst_node, addr_nzval, nnz*r*c*elemsize);
fail_nzval:
	/* allocation failed */
	return -ENOMEM;
}

static void free_bcsr_buffer_on_node(void *data_interface, unsigned node)
{
	struct starpu_bcsr_interface *bcsr_interface = (struct starpu_bcsr_interface *) data_interface;
	uint32_t nnz = bcsr_interface->nnz;
	uint32_t nrow = bcsr_interface->nrow;
	size_t elemsize = bcsr_interface->elemsize;
	uint32_t r = bcsr_interface->r;
	uint32_t c = bcsr_interface->c;

	if (nnz)
	{
		starpu_free_on_node(node, bcsr_interface->nzval, nnz*r*c*elemsize);
		starpu_free_on_node(node, (uintptr_t) bcsr_interface->colind, nnz*sizeof(uint32_t));
	}
	starpu_free_on_node(node, (uintptr_t) bcsr_interface->rowptr, (nrow+1)*sizeof(uint32_t));
}

static int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data)
{
	struct starpu_bcsr_interface *src_bcsr = (struct starpu_bcsr_interface *) src_interface;
	struct starpu_bcsr_interface *dst_bcsr = (struct starpu_bcsr_interface *) dst_interface;

	uint32_t nnz = src_bcsr->nnz;
	uint32_t nrow = src_bcsr->nrow;
	size_t elemsize = src_bcsr->elemsize;

	uint32_t r = src_bcsr->r;
	uint32_t c = src_bcsr->c;

	int ret = 0;

	if (nnz)
	{
		if (starpu_interface_copy(src_bcsr->nzval, 0, src_node, dst_bcsr->nzval, 0, dst_node, nnz*elemsize*r*c, async_data))
			ret = -EAGAIN;

		if (starpu_interface_copy((uintptr_t)src_bcsr->colind, 0, src_node, (uintptr_t)dst_bcsr->colind, 0, dst_node, nnz*sizeof(uint32_t), async_data))
			ret = -EAGAIN;
	}

	if (starpu_interface_copy((uintptr_t)src_bcsr->rowptr, 0, src_node, (uintptr_t)dst_bcsr->rowptr, 0, dst_node, (nrow+1)*sizeof(uint32_t), async_data))
		ret = -EAGAIN;

	starpu_interface_data_copy(src_node, dst_node, nnz*elemsize*r*c + (nnz+nrow+1)*sizeof(uint32_t));

	return ret;
}

static starpu_ssize_t describe(void *data_interface, char *buf, size_t size)
{
	struct starpu_bcsr_interface *bcsr = (struct starpu_bcsr_interface *) data_interface;
	return snprintf(buf, size, "b%ux%ux%ux%ux%u",
			(unsigned) bcsr->nnz,
			(unsigned) bcsr->nrow,
			(unsigned) bcsr->r,
			(unsigned) bcsr->c,
			(unsigned) bcsr->elemsize);
}

static int pack_data(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_bcsr_interface *bcsr = (struct starpu_bcsr_interface *) starpu_data_get_interface_on_node(handle, node);

	// We first pack colind
	*count = bcsr->nnz * sizeof(bcsr->colind[0]);
	// Then rowptr
	*count += (bcsr->nrow + 1) * sizeof(bcsr->rowptr[0]);
	// Then nnzval
	*count += bcsr->r * bcsr->c * bcsr->nnz * bcsr->elemsize;

	if (ptr != NULL)
	{
		*ptr = (void *)starpu_malloc_on_node_flags(node, *count, 0);
		char *tmp = *ptr;
		if (bcsr->nnz)
		{
			memcpy(tmp, (void*)bcsr->colind, bcsr->nnz * sizeof(bcsr->colind[0]));
			tmp += bcsr->nnz * sizeof(bcsr->colind[0]);
			memcpy(tmp, (void*)bcsr->rowptr, (bcsr->nrow + 1) * sizeof(bcsr->rowptr[0]));
			tmp += (bcsr->nrow + 1) * sizeof(bcsr->rowptr[0]);
		}
		memcpy(tmp, (void*)bcsr->nzval, bcsr->r * bcsr->c * bcsr->nnz * bcsr->elemsize);
	}

	return 0;
}

static int unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_bcsr_interface *bcsr = (struct starpu_bcsr_interface *) starpu_data_get_interface_on_node(handle, node);

	STARPU_ASSERT(count == (bcsr->nnz * sizeof(bcsr->colind[0]))+((bcsr->nrow + 1) * sizeof(bcsr->rowptr[0]))+(bcsr->r * bcsr->c * bcsr->nnz * bcsr->elemsize));

	char *tmp = ptr;
	if (bcsr->nnz)
	{
		memcpy((void*)bcsr->colind, tmp, bcsr->nnz * sizeof(bcsr->colind[0]));
		tmp += bcsr->nnz * sizeof(bcsr->colind[0]);
		memcpy((void*)bcsr->rowptr, tmp, (bcsr->nrow + 1) * sizeof(bcsr->rowptr[0]));
		tmp += (bcsr->nrow + 1) * sizeof(bcsr->rowptr[0]);
	}
	memcpy((void*)bcsr->nzval, tmp, bcsr->r * bcsr->c * bcsr->nnz * bcsr->elemsize);

	starpu_free_on_node_flags(node, (uintptr_t)ptr, count, 0);

	return 0;
}

