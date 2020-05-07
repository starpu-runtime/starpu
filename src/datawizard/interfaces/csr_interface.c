/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
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

static const struct starpu_data_copy_methods csr_copy_data_methods_s =
{
	.any_to_any = copy_any_to_any,
};

static void register_csr_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface);
static int csr_pointer_is_inside(void *data_interface, unsigned node, void *ptr);
static starpu_ssize_t allocate_csr_buffer_on_node(void *data_interface_, unsigned dst_node);
static void free_csr_buffer_on_node(void *data_interface, unsigned node);
static size_t csr_interface_get_size(starpu_data_handle_t handle);
static int csr_compare(void *data_interface_a, void *data_interface_b);
static uint32_t footprint_csr_interface_crc32(starpu_data_handle_t handle);
static starpu_ssize_t describe(void *data_interface, char *buf, size_t size);
static int pack_data(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count);
static int unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);

struct starpu_data_interface_ops starpu_interface_csr_ops =
{
	.register_data_handle = register_csr_handle,
	.allocate_data_on_node = allocate_csr_buffer_on_node,
	.free_data_on_node = free_csr_buffer_on_node,
	.copy_methods = &csr_copy_data_methods_s,
	.get_size = csr_interface_get_size,
	.interfaceid = STARPU_CSR_INTERFACE_ID,
	.interface_size = sizeof(struct starpu_csr_interface),
	.footprint = footprint_csr_interface_crc32,
	.compare = csr_compare,
	.describe = describe,
	.pointer_is_inside = csr_pointer_is_inside,
	.name = "STARPU_CSR_INTERFACE",
	.pack_data = pack_data,
	.unpack_data = unpack_data
};

static int csr_pointer_is_inside(void *data_interface, unsigned node, void *ptr)
{
	(void) node;
	struct starpu_csr_interface *csr_interface = data_interface;

	return ((char*) ptr >= (char*) csr_interface->nzval &&
		(char*) ptr < (char*) csr_interface->nzval + csr_interface->nnz*csr_interface->elemsize)
	    || ((char*) ptr >= (char*) csr_interface->colind &&
		(char*) ptr < (char*) csr_interface->colind + csr_interface->nnz*sizeof(uint32_t))
	    || ((char*) ptr >= (char*) csr_interface->rowptr &&
		(char*) ptr < (char*) csr_interface->rowptr + (csr_interface->nrow+1)*sizeof(uint32_t));
}

static void register_csr_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface)
{
	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *) data_interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_csr_interface *local_interface = (struct starpu_csr_interface *)
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
		{
			local_interface->nzval = csr_interface->nzval;
			local_interface->colind = csr_interface->colind;
		}
		else
		{
			local_interface->nzval = 0;
			local_interface->colind = NULL;
		}

		local_interface->id = csr_interface->id;
		local_interface->rowptr = csr_interface->rowptr;
		local_interface->nnz = csr_interface->nnz;
		local_interface->nrow = csr_interface->nrow;
		local_interface->firstentry = csr_interface->firstentry;
		local_interface->elemsize = csr_interface->elemsize;

	}
}

/* declare a new data with the BLAS interface */
void starpu_csr_data_register(starpu_data_handle_t *handleptr, int home_node,
		uint32_t nnz, uint32_t nrow, uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, size_t elemsize)
{
	struct starpu_csr_interface csr_interface =
	{
		.id = STARPU_CSR_INTERFACE_ID,
		.nnz = nnz,
		.nrow = nrow,
		.nzval = nzval,
		.colind = colind,
		.rowptr = rowptr,
		.firstentry = firstentry,
		.elemsize = elemsize
	};
#ifndef STARPU_SIMGRID
	if (home_node >= 0 && starpu_node_get_kind(home_node) == STARPU_CPU_RAM)
	{
		if (nnz)
		{
			if (elemsize)
			{
				STARPU_ASSERT_ACCESSIBLE(nzval);
				STARPU_ASSERT_ACCESSIBLE(nzval + nnz*elemsize - 1);
			}
			STARPU_ASSERT_ACCESSIBLE(colind);
			STARPU_ASSERT_ACCESSIBLE((uintptr_t) colind + nnz*sizeof(uint32_t) - 1);
		}
		STARPU_ASSERT_ACCESSIBLE(rowptr);
		STARPU_ASSERT_ACCESSIBLE((uintptr_t) rowptr + (nrow+1)*sizeof(uint32_t) - 1);
	}
#endif

	starpu_data_register(handleptr, home_node, &csr_interface, &starpu_interface_csr_ops);
}

static uint32_t footprint_csr_interface_crc32(starpu_data_handle_t handle)
{
	return starpu_hash_crc32c_be(starpu_csr_get_nnz(handle), 0);
}

static int csr_compare(void *data_interface_a, void *data_interface_b)
{
	struct starpu_csr_interface *csr_a = (struct starpu_csr_interface *) data_interface_a;
	struct starpu_csr_interface *csr_b = (struct starpu_csr_interface *) data_interface_b;

	/* Two matricess are considered compatible if they have the same size */
	return (csr_a->nnz == csr_b->nnz)
		&& (csr_a->nrow == csr_b->nrow)
		&& (csr_a->elemsize == csr_b->elemsize);
}

/* offer an access to the data parameters */
uint32_t starpu_csr_get_nnz(starpu_data_handle_t handle)
{
	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(csr_interface->id == STARPU_CSR_INTERFACE_ID, "Error. The given data is not a csr.");
#endif

	return csr_interface->nnz;
}

uint32_t starpu_csr_get_nrow(starpu_data_handle_t handle)
{
	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(csr_interface->id == STARPU_CSR_INTERFACE_ID, "Error. The given data is not a csr.");
#endif

	return csr_interface->nrow;
}

uint32_t starpu_csr_get_firstentry(starpu_data_handle_t handle)
{
	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(csr_interface->id == STARPU_CSR_INTERFACE_ID, "Error. The given data is not a csr.");
#endif

	return csr_interface->firstentry;
}

size_t starpu_csr_get_elemsize(starpu_data_handle_t handle)
{
	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(csr_interface->id == STARPU_CSR_INTERFACE_ID, "Error. The given data is not a csr.");
#endif

	return csr_interface->elemsize;
}

uintptr_t starpu_csr_get_local_nzval(starpu_data_handle_t handle)
{
	unsigned node;
	node = starpu_worker_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *)
		starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(csr_interface->id == STARPU_CSR_INTERFACE_ID, "Error. The given data is not a csr.");
#endif

	return csr_interface->nzval;
}

uint32_t *starpu_csr_get_local_colind(starpu_data_handle_t handle)
{
	unsigned node;
	node = starpu_worker_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *)
		starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(csr_interface->id == STARPU_CSR_INTERFACE_ID, "Error. The given data is not a csr.");
#endif

	return csr_interface->colind;
}

uint32_t *starpu_csr_get_local_rowptr(starpu_data_handle_t handle)
{
	unsigned node;
	node = starpu_worker_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *)
		starpu_data_get_interface_on_node(handle, node);

#ifdef STARPU_DEBUG
	STARPU_ASSERT_MSG(csr_interface->id == STARPU_CSR_INTERFACE_ID, "Error. The given data is not a csr.");
#endif

	return csr_interface->rowptr;
}

static size_t csr_interface_get_size(starpu_data_handle_t handle)
{
	size_t size;

	uint32_t nnz = starpu_csr_get_nnz(handle);
	uint32_t nrow = starpu_csr_get_nrow(handle);
	size_t elemsize = starpu_csr_get_elemsize(handle);

	size = nnz*elemsize + nnz*sizeof(uint32_t) + (nrow+1)*sizeof(uint32_t);

	return size;
}

/* memory allocation/deallocation primitives for the BLAS interface */

/* returns the size of the allocated area */
static starpu_ssize_t allocate_csr_buffer_on_node(void *data_interface_, unsigned dst_node)
{
	uintptr_t addr_nzval = 0;
	uint32_t *addr_colind = NULL, *addr_rowptr = NULL;
	starpu_ssize_t allocated_memory;

	/* we need the 3 arrays to be allocated */
	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *) data_interface_;

	uint32_t nnz = csr_interface->nnz;
	uint32_t nrow = csr_interface->nrow;
	size_t elemsize = csr_interface->elemsize;

	if (nnz)
	{
		addr_nzval = starpu_malloc_on_node(dst_node, nnz*elemsize);
		if (!addr_nzval)
			goto fail_nzval;
		addr_colind = (uint32_t*) starpu_malloc_on_node(dst_node, nnz*sizeof(uint32_t));
		if (!addr_colind)
			goto fail_colind;
	}
	else
	{
		addr_nzval = 0;
		addr_colind = NULL;
	}
	addr_rowptr = (uint32_t*) starpu_malloc_on_node(dst_node, (nrow+1)*sizeof(uint32_t));
	if (!addr_rowptr)
		goto fail_rowptr;

	/* allocation succeeded */
	allocated_memory =
		nnz*elemsize + nnz*sizeof(uint32_t) + (nrow+1)*sizeof(uint32_t);

	/* update the data properly in consequence */
	csr_interface->nzval = addr_nzval;
	csr_interface->colind = addr_colind;
	csr_interface->rowptr = addr_rowptr;

	return allocated_memory;

fail_rowptr:
	if (nnz)
		starpu_free_on_node(dst_node, (uintptr_t) addr_colind, nnz*sizeof(uint32_t));
fail_colind:
	if (nnz)
		starpu_free_on_node(dst_node, addr_nzval, nnz*elemsize);
fail_nzval:
	/* allocation failed */
	return -ENOMEM;
}

static void free_csr_buffer_on_node(void *data_interface, unsigned node)
{
	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *) data_interface;
	uint32_t nnz = csr_interface->nnz;
	uint32_t nrow = csr_interface->nrow;
	size_t elemsize = csr_interface->elemsize;

	if (nnz)
	{
		starpu_free_on_node(node, csr_interface->nzval, nnz*elemsize);
		starpu_free_on_node(node, (uintptr_t) csr_interface->colind, nnz*sizeof(uint32_t));
	}
	starpu_free_on_node(node, (uintptr_t) csr_interface->rowptr, (nrow+1)*sizeof(uint32_t));
}

/* as not all platform easily have a BLAS lib installed ... */
static int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data)
{
	struct starpu_csr_interface *src_csr = (struct starpu_csr_interface *) src_interface;
	struct starpu_csr_interface *dst_csr = (struct starpu_csr_interface *) dst_interface;

	uint32_t nnz = src_csr->nnz;
	uint32_t nrow = src_csr->nrow;
	size_t elemsize = src_csr->elemsize;
	int ret = 0;

	if (nnz)
	{
		if (starpu_interface_copy(src_csr->nzval, 0, src_node, dst_csr->nzval, 0, dst_node, nnz*elemsize, async_data))
			ret = -EAGAIN;

		if (starpu_interface_copy((uintptr_t)src_csr->colind, 0, src_node, (uintptr_t)dst_csr->colind, 0, dst_node, nnz*sizeof(uint32_t), async_data))
			ret = -EAGAIN;
	}

	if (starpu_interface_copy((uintptr_t)src_csr->rowptr, 0, src_node, (uintptr_t)dst_csr->rowptr, 0, dst_node, (nrow+1)*sizeof(uint32_t), async_data))
		ret = -EAGAIN;

	starpu_interface_data_copy(src_node, dst_node, nnz*elemsize + (nnz+nrow+1)*sizeof(uint32_t));

	return ret;
}

static starpu_ssize_t describe(void *data_interface, char *buf, size_t size)
{
	struct starpu_csr_interface *csr = (struct starpu_csr_interface *) data_interface;
	return snprintf(buf, size, "C%ux%ux%u",
			(unsigned) csr->nnz,
			(unsigned) csr->nrow,
			(unsigned) csr->elemsize);
}

static int pack_data(starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_csr_interface *csr = (struct starpu_csr_interface *) starpu_data_get_interface_on_node(handle, node);

	// We first pack colind
	*count = csr->nnz * sizeof(csr->colind[0]);
	// Then rowptr
	*count += (csr->nrow + 1) * sizeof(csr->rowptr[0]);
	// Then nnzval
	*count += csr->nnz * csr->elemsize;

	if (ptr != NULL)
	{
		*ptr = (void *)starpu_malloc_on_node_flags(node, *count, 0);
		char *tmp = *ptr;
		if (csr->nnz)
		{
			memcpy(tmp, (void*)csr->colind, csr->nnz * sizeof(csr->colind[0]));
			tmp += csr->nnz * sizeof(csr->colind[0]);
			memcpy(tmp, (void*)csr->rowptr, (csr->nrow + 1) * sizeof(csr->rowptr[0]));
			tmp += (csr->nrow + 1) * sizeof(csr->rowptr[0]);
		}
		memcpy(tmp, (void*)csr->nzval, csr->nnz * csr->elemsize);
	}

	return 0;
}

static int unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_csr_interface *csr = (struct starpu_csr_interface *) starpu_data_get_interface_on_node(handle, node);

	STARPU_ASSERT(count == (csr->nnz * sizeof(csr->colind[0]))+((csr->nrow + 1) * sizeof(csr->rowptr[0]))+(csr->nnz * csr->elemsize));

	char *tmp = ptr;
	if (csr->nnz)
	{
		memcpy((void*)csr->colind, tmp, csr->nnz * sizeof(csr->colind[0]));
		tmp += csr->nnz * sizeof(csr->colind[0]);
		memcpy((void*)csr->rowptr, tmp, (csr->nrow + 1) * sizeof(csr->rowptr[0]));
		tmp += (csr->nrow + 1) * sizeof(csr->rowptr[0]);
	}
	memcpy((void*)csr->nzval, tmp, csr->nnz * csr->elemsize);

	starpu_free_on_node_flags(node, (uintptr_t)ptr, count, 0);

	return 0;
}
