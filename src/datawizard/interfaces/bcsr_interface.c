/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <common/config.h>

#include <datawizard/data_parameters.h>
#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>
#include <datawizard/hierarchy.h>
#include <common/hash.h>

/*
 * BCSR : blocked CSR, we use blocks of size (r x c)
 */

static int dummy_copy_ram_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node);
#ifdef USE_CUDA
static int copy_ram_to_cublas(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node);
static int copy_cublas_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node);
#endif

static const struct copy_data_methods_s bcsr_copy_data_methods_s = {
	.ram_to_ram = dummy_copy_ram_to_ram,
	.ram_to_spu = NULL,
#ifdef USE_CUDA
	.ram_to_cuda = copy_ram_to_cublas,
	.cuda_to_ram = copy_cublas_to_ram,
#endif
	.cuda_to_cuda = NULL,
	.cuda_to_spu = NULL,
	.spu_to_ram = NULL,
	.spu_to_cuda = NULL,
	.spu_to_spu = NULL
};

static void register_bcsr_handle(starpu_data_handle handle, uint32_t home_node, void *interface);
static size_t allocate_bcsr_buffer_on_node(starpu_data_handle handle, uint32_t dst_node);
static void liberate_bcsr_buffer_on_node(void *interface, uint32_t node);
static size_t bcsr_interface_get_size(starpu_data_handle handle);
static uint32_t footprint_bcsr_interface_crc32(starpu_data_handle handle, uint32_t hstate);

struct data_interface_ops_t interface_bcsr_ops = {
	.register_data_handle = register_bcsr_handle,
	.allocate_data_on_node = allocate_bcsr_buffer_on_node,
	.liberate_data_on_node = liberate_bcsr_buffer_on_node,
	.copy_methods = &bcsr_copy_data_methods_s,
	.get_size = bcsr_interface_get_size,
	.interfaceid = STARPU_BCSCR_INTERFACE_ID,
	.interface_size = sizeof(starpu_bcsr_interface_t),
	.footprint = footprint_bcsr_interface_crc32
};

static void register_bcsr_handle(starpu_data_handle handle, uint32_t home_node, void *interface)
{
	starpu_bcsr_interface_t *bcsr_interface = interface;

	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		starpu_bcsr_interface_t *local_interface =
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node) {
			local_interface->nzval = bcsr_interface->nzval;
			local_interface->colind = bcsr_interface->colind;
			local_interface->rowptr = bcsr_interface->rowptr;
		}
		else {
			local_interface->nzval = 0;
			local_interface->colind = NULL;
			local_interface->rowptr = NULL;
		}

		local_interface->nnz = bcsr_interface->nnz;
		local_interface->nrow = bcsr_interface->nrow;
		local_interface->firstentry = bcsr_interface->firstentry;
		local_interface->r = bcsr_interface->r;
		local_interface->c = bcsr_interface->c;
		local_interface->elemsize = bcsr_interface->elemsize;
	}
}

void starpu_register_bcsr_data(starpu_data_handle *handleptr, uint32_t home_node,
		uint32_t nnz, uint32_t nrow, uintptr_t nzval, uint32_t *colind,
		uint32_t *rowptr, uint32_t firstentry,
		uint32_t r, uint32_t c, size_t elemsize)
{
	starpu_bcsr_interface_t interface = {
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

	register_data_handle(handleptr, home_node, &interface, &interface_bcsr_ops);
}

static inline uint32_t footprint_bcsr_interface_generic(uint32_t (*hash_func)(uint32_t input, uint32_t hstate), starpu_data_handle handle, uint32_t hstate)
{
	uint32_t hash;

	hash = hstate;
	hash = hash_func(starpu_get_bcsr_nnz(handle), hash);
	hash = hash_func(starpu_get_bcsr_c(handle), hash);
	hash = hash_func(starpu_get_bcsr_r(handle), hash);

	return hash;
}

static uint32_t footprint_bcsr_interface_crc32(starpu_data_handle handle, uint32_t hstate)
{
	return footprint_bcsr_interface_generic(crc32_be, handle, hstate);
}

/* offer an access to the data parameters */
uint32_t starpu_get_bcsr_nnz(starpu_data_handle handle)
{
	starpu_bcsr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->nnz;
}

uint32_t starpu_get_bcsr_nrow(starpu_data_handle handle)
{
	starpu_bcsr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->nrow;
}

uint32_t starpu_get_bcsr_firstentry(starpu_data_handle handle)
{
	starpu_bcsr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->firstentry;
}

uint32_t starpu_get_bcsr_r(starpu_data_handle handle)
{
	starpu_bcsr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->r;
}

uint32_t starpu_get_bcsr_c(starpu_data_handle handle)
{
	starpu_bcsr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->c;
}

size_t starpu_get_bcsr_elemsize(starpu_data_handle handle)
{
	starpu_bcsr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->elemsize;
}

uintptr_t starpu_get_bcsr_local_nzval(starpu_data_handle handle)
{
	unsigned node;
	node = get_local_memory_node();

	STARPU_ASSERT(starpu_test_if_data_is_allocated_on_node(handle, node));

	starpu_bcsr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, node);
	
	return interface->nzval;
}

uint32_t *starpu_get_bcsr_local_colind(starpu_data_handle handle)
{
	/* XXX 0 */
	starpu_bcsr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->colind;
}

uint32_t *starpu_get_bcsr_local_rowptr(starpu_data_handle handle)
{
	/* XXX 0 */
	starpu_bcsr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->rowptr;
}


static size_t bcsr_interface_get_size(starpu_data_handle handle)
{
	size_t size;

	uint32_t nnz = starpu_get_bcsr_nnz(handle);
	uint32_t nrow = starpu_get_bcsr_nrow(handle);
	uint32_t r = starpu_get_bcsr_r(handle);
	uint32_t c = starpu_get_bcsr_c(handle);
	size_t elemsize = starpu_get_bcsr_elemsize(handle);

	size = nnz*r*c*elemsize + nnz*sizeof(uint32_t) + (nrow+1)*sizeof(uint32_t); 

	return size;
}


/* memory allocation/deallocation primitives for the BLAS interface */

/* returns the size of the allocated area */
static size_t allocate_bcsr_buffer_on_node(starpu_data_handle handle, uint32_t dst_node)
{
	uintptr_t addr_nzval;
	uint32_t *addr_colind, *addr_rowptr;
	size_t allocated_memory;

	/* we need the 3 arrays to be allocated */
	starpu_bcsr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, dst_node);

	uint32_t nnz = interface->nnz;
	uint32_t nrow = interface->nrow;
	size_t elemsize = interface->elemsize;

	uint32_t r = interface->r;
	uint32_t c = interface->c;

	node_kind kind = get_node_kind(dst_node);

	switch(kind) {
		case RAM:
			addr_nzval = (uintptr_t)malloc(nnz*r*c*elemsize);
			if (!addr_nzval)
				goto fail_nzval;

			addr_colind = malloc(nnz*sizeof(uint32_t));
			if (!addr_colind)
				goto fail_colind;

			addr_rowptr = malloc((nrow+1)*sizeof(uint32_t));
			if (!addr_rowptr)
				goto fail_rowptr;

			break;
#ifdef USE_CUDA
		case CUDA_RAM:
			cublasAlloc(nnz*r*c, elemsize, (void **)&addr_nzval);
			if (!addr_nzval)
				goto fail_nzval;

			cublasAlloc(nnz, sizeof(uint32_t), (void **)&addr_colind);
			if (!addr_colind)
				goto fail_colind;

			cublasAlloc((nrow+1), sizeof(uint32_t), (void **)&addr_rowptr);
			if (!addr_rowptr)
				goto fail_rowptr;

			break;
#endif
		default:
			assert(0);
	}

	/* allocation succeeded */
	allocated_memory = 
		nnz*r*c*elemsize + nnz*sizeof(uint32_t) + (nrow+1)*sizeof(uint32_t);

	/* update the data properly in consequence */
	interface->nzval = addr_nzval;
	interface->colind = addr_colind;
	interface->rowptr = addr_rowptr;
	
	return allocated_memory;

fail_rowptr:
	switch(kind) {
		case RAM:
			free((void *)addr_colind);
#ifdef USE_CUDA
		case CUDA_RAM:
			cublasFree((void*)addr_colind);
			break;
#endif
		default:
			assert(0);
	}

fail_colind:
	switch(kind) {
		case RAM:
			free((void *)addr_nzval);
#ifdef USE_CUDA
		case CUDA_RAM:
			cublasFree((void*)addr_nzval);
			break;
#endif
		default:
			assert(0);
	}

fail_nzval:

	/* allocation failed */
	allocated_memory = 0;

	return allocated_memory;
}

static void liberate_bcsr_buffer_on_node(void *interface, uint32_t node)
{
	starpu_bcsr_interface_t *bcsr_interface = interface;	

	node_kind kind = get_node_kind(node);
	switch(kind) {
		case RAM:
			free((void*)bcsr_interface->nzval);
			free((void*)bcsr_interface->colind);
			free((void*)bcsr_interface->rowptr);
			break;
#ifdef USE_CUDA
		case CUDA_RAM:
			cublasFree((void*)bcsr_interface->nzval);
			cublasFree((void*)bcsr_interface->colind);
			cublasFree((void*)bcsr_interface->rowptr);
			break;
#endif
		default:
			assert(0);
	}
}

#ifdef USE_CUDA
static int copy_cublas_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node)
{
	starpu_bcsr_interface_t *src_bcsr;
	starpu_bcsr_interface_t *dst_bcsr;

	src_bcsr = starpu_data_get_interface_on_node(handle, src_node);
	dst_bcsr = starpu_data_get_interface_on_node(handle, dst_node);

	uint32_t nnz = src_bcsr->nnz;
	uint32_t nrow = src_bcsr->nrow;
	size_t elemsize = src_bcsr->elemsize;

	uint32_t r = src_bcsr->r;
	uint32_t c = src_bcsr->c;

	cublasGetVector(nnz*r*c, elemsize, (uint8_t *)src_bcsr->nzval, 1, 
			 		   (uint8_t *)dst_bcsr->nzval, 1);

	cublasGetVector(nnz, sizeof(uint32_t), (uint8_t *)src_bcsr->colind, 1, 
						(uint8_t *)dst_bcsr->colind, 1);

	cublasGetVector((nrow+1), sizeof(uint32_t), (uint8_t *)src_bcsr->rowptr, 1, 
						(uint8_t *)dst_bcsr->rowptr, 1);
	
	TRACE_DATA_COPY(src_node, dst_node, nnz*r*c*elemsize + (nnz+nrow+1)*sizeof(uint32_t));

	return 0;
}

static int copy_ram_to_cublas(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node)
{
	starpu_bcsr_interface_t *src_bcsr;
	starpu_bcsr_interface_t *dst_bcsr;

	src_bcsr = starpu_data_get_interface_on_node(handle, src_node);
	dst_bcsr = starpu_data_get_interface_on_node(handle, dst_node);

	uint32_t nnz = src_bcsr->nnz;
	uint32_t nrow = src_bcsr->nrow;
	size_t elemsize = src_bcsr->elemsize;

	uint32_t r = src_bcsr->r;
	uint32_t c = src_bcsr->c;

	cublasSetVector(nnz*r*c, elemsize, (uint8_t *)src_bcsr->nzval, 1, 
					(uint8_t *)dst_bcsr->nzval, 1);

	cublasSetVector(nnz, sizeof(uint32_t), (uint8_t *)src_bcsr->colind, 1, 
						(uint8_t *)dst_bcsr->colind, 1);

	cublasSetVector((nrow+1), sizeof(uint32_t), (uint8_t *)src_bcsr->rowptr, 1, 
						(uint8_t *)dst_bcsr->rowptr, 1);
	
	TRACE_DATA_COPY(src_node, dst_node, nnz*r*c*elemsize + (nnz+nrow+1)*sizeof(uint32_t));

	return 0;
}
#endif // USE_CUDA

/* as not all platform easily have a BLAS lib installed ... */
static int dummy_copy_ram_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node)
{
	starpu_bcsr_interface_t *src_bcsr;
	starpu_bcsr_interface_t *dst_bcsr;

	src_bcsr = starpu_data_get_interface_on_node(handle, src_node);
	dst_bcsr = starpu_data_get_interface_on_node(handle, dst_node);

	uint32_t nnz = src_bcsr->nnz;
	uint32_t nrow = src_bcsr->nrow;
	size_t elemsize = src_bcsr->elemsize;

	uint32_t r = src_bcsr->r;
	uint32_t c = src_bcsr->c;

	memcpy((void *)dst_bcsr->nzval, (void *)src_bcsr->nzval, nnz*elemsize*r*c);

	memcpy((void *)dst_bcsr->colind, (void *)src_bcsr->colind, nnz*sizeof(uint32_t));

	memcpy((void *)dst_bcsr->rowptr, (void *)src_bcsr->rowptr, (nrow+1)*sizeof(uint32_t));

	TRACE_DATA_COPY(src_node, dst_node, nnz*elemsize*r*c + (nnz+nrow+1)*sizeof(uint32_t));

	return 0;
}
