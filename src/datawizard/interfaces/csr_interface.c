/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2013  Université de Bordeaux 1
 * Copyright (C) 2010  Mehdi Juhoor <mjuhoor@gmail.com>
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

static int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data);

static struct starpu_data_copy_methods csr_copy_data_methods_s =
{
	.any_to_any = copy_any_to_any,
};

static void register_csr_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface);
static ssize_t allocate_csr_buffer_on_node(void *data_interface_, unsigned dst_node);
static void free_csr_buffer_on_node(void *data_interface, unsigned node);
static size_t csr_interface_get_size(starpu_data_handle_t handle);
static int csr_compare(void *data_interface_a, void *data_interface_b);
static uint32_t footprint_csr_interface_crc32(starpu_data_handle_t handle);

static struct starpu_data_interface_ops interface_csr_ops =
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
};

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

		local_interface->rowptr = csr_interface->rowptr;
		local_interface->nnz = csr_interface->nnz;
		local_interface->nrow = csr_interface->nrow;
		local_interface->firstentry = csr_interface->firstentry;
		local_interface->elemsize = csr_interface->elemsize;

	}
}

/* declare a new data with the BLAS interface */
void starpu_csr_data_register(starpu_data_handle_t *handleptr, unsigned home_node,
		uint32_t nnz, uint32_t nrow, uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, size_t elemsize)
{
	struct starpu_csr_interface csr_interface =
	{
		.nnz = nnz,
		.nrow = nrow,
		.nzval = nzval,
		.colind = colind,
		.rowptr = rowptr,
		.firstentry = firstentry,
		.elemsize = elemsize
	};

	starpu_data_register(handleptr, home_node, &csr_interface, &interface_csr_ops);
}

static uint32_t footprint_csr_interface_crc32(starpu_data_handle_t handle)
{
	return starpu_crc32_be(starpu_csr_get_nnz(handle), 0);
}

static int csr_compare(void *data_interface_a, void *data_interface_b)
{
	struct starpu_csr_interface *csr_a = (struct starpu_csr_interface *) data_interface_a;
	struct starpu_csr_interface *csr_b = (struct starpu_csr_interface *) data_interface_b;

	/* Two matricess are considered compatible if they have the same size */
	return ((csr_a->nnz == csr_b->nnz)
			&& (csr_a->nrow == csr_b->nrow)
			&& (csr_a->elemsize == csr_b->elemsize));
}

/* offer an access to the data parameters */
uint32_t starpu_csr_get_nnz(starpu_data_handle_t handle)
{
	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	return csr_interface->nnz;
}

uint32_t starpu_csr_get_nrow(starpu_data_handle_t handle)
{
	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	return csr_interface->nrow;
}

uint32_t starpu_csr_get_firstentry(starpu_data_handle_t handle)
{
	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	return csr_interface->firstentry;
}

size_t starpu_csr_get_elemsize(starpu_data_handle_t handle)
{
	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	return csr_interface->elemsize;
}

uintptr_t starpu_csr_get_local_nzval(starpu_data_handle_t handle)
{
	unsigned node;
	node = _starpu_memory_node_get_local_key();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *)
		starpu_data_get_interface_on_node(handle, node);

	return csr_interface->nzval;
}

uint32_t *starpu_csr_get_local_colind(starpu_data_handle_t handle)
{
	unsigned node;
	node = _starpu_memory_node_get_local_key();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *)
		starpu_data_get_interface_on_node(handle, node);

	return csr_interface->colind;
}

uint32_t *starpu_csr_get_local_rowptr(starpu_data_handle_t handle)
{
	unsigned node;
	node = _starpu_memory_node_get_local_key();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *)
		starpu_data_get_interface_on_node(handle, node);

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
static ssize_t allocate_csr_buffer_on_node(void *data_interface_, unsigned dst_node)
{
	uintptr_t addr_nzval = 0;
	uint32_t *addr_colind = NULL, *addr_rowptr = NULL;
	ssize_t allocated_memory;

	/* we need the 3 arrays to be allocated */
	struct starpu_csr_interface *csr_interface = (struct starpu_csr_interface *) data_interface_;

	uint32_t nnz = csr_interface->nnz;
	uint32_t nrow = csr_interface->nrow;
	size_t elemsize = csr_interface->elemsize;

	addr_nzval = starpu_allocate_buffer_on_node(dst_node, nnz*elemsize);
	if (!addr_nzval)
		goto fail_nzval;
	addr_colind = (uint32_t*) starpu_allocate_buffer_on_node(dst_node, nnz*sizeof(uint32_t));
	if (!addr_colind)
		goto fail_colind;
	addr_rowptr = (uint32_t*) starpu_allocate_buffer_on_node(dst_node, (nrow+1)*sizeof(uint32_t));
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
	starpu_free_buffer_on_node(dst_node, (uintptr_t) addr_colind, nnz*sizeof(uint32_t));
fail_colind:
	starpu_free_buffer_on_node(dst_node, addr_nzval, nnz*elemsize);
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

	starpu_free_buffer_on_node(node, csr_interface->nzval, nnz*elemsize);
	starpu_free_buffer_on_node(node, (uintptr_t) csr_interface->colind, nnz*sizeof(uint32_t));
	starpu_free_buffer_on_node(node, (uintptr_t) csr_interface->rowptr, (nrow+1)*sizeof(uint32_t));
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

	if (starpu_interface_copy(src_csr->nzval, src_node, 0, dst_csr->nzval, dst_node, 0, nnz*elemsize, async_data))
		ret = -EAGAIN;

	if (starpu_interface_copy((uintptr_t)src_csr->colind, src_node, 0, (uintptr_t)dst_csr->colind, dst_node, 0, nnz*sizeof(uint32_t), async_data))
		ret = -EAGAIN;

	if (starpu_interface_copy((uintptr_t)src_csr->rowptr, src_node, 0, (uintptr_t)dst_csr->rowptr, dst_node, 0, (nrow+1)*sizeof(uint32_t), async_data))
		ret = -EAGAIN;

	_STARPU_TRACE_DATA_COPY(src_node, dst_node, nnz*elemsize + (nnz+nrow+1)*sizeof(uint32_t));

	return ret;
}
