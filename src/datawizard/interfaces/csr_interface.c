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

#include <datawizard/data_parameters.h>
#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>
#include <datawizard/hierarchy.h>

#include <common/hash.h>

#include <starpu.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

static int dummy_copy_ram_to_ram(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node);
#ifdef USE_CUDA
static int copy_ram_to_cublas(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node);
static int copy_cublas_to_ram(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node);
#endif

static const struct copy_data_methods_s csr_copy_data_methods_s = {
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


static size_t allocate_csr_buffer_on_node(struct starpu_data_state_t *state, uint32_t dst_node);
static void liberate_csr_buffer_on_node(starpu_data_interface_t *interface, uint32_t node);
static size_t dump_csr_interface(starpu_data_interface_t *interface, void *_buffer);
static size_t csr_interface_get_size(struct starpu_data_state_t *state);
static uint32_t footprint_csr_interface_crc32(data_state *state, uint32_t hstate);

struct data_interface_ops_t interface_csr_ops = {
	.allocate_data_on_node = allocate_csr_buffer_on_node,
	.liberate_data_on_node = liberate_csr_buffer_on_node,
	.copy_methods = &csr_copy_data_methods_s,
	.dump_data_interface = dump_csr_interface,
	.get_size = csr_interface_get_size,
	.interfaceid = CSR_INTERFACE,
	.footprint = footprint_csr_interface_crc32
};

/* declare a new data with the BLAS interface */
void starpu_register_csr_data(struct starpu_data_state_t **handle, uint32_t home_node,
		uint32_t nnz, uint32_t nrow, uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, size_t elemsize)
{
	struct starpu_data_state_t *state = calloc(1, sizeof(struct starpu_data_state_t));
	STARPU_ASSERT(state);

	STARPU_ASSERT(handle);
	*handle = state;

	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		starpu_csr_interface_t *local_interface = &state->interface[node].csr;

		if (node == home_node) {
			local_interface->nzval = nzval;
			local_interface->colind = colind;
			local_interface->rowptr = rowptr;
		}
		else {
			local_interface->nzval = 0;
			local_interface->colind = NULL;
			local_interface->rowptr = NULL;
		}

		local_interface->nnz = nnz;
		local_interface->nrow = nrow;
		local_interface->firstentry = firstentry;
		local_interface->elemsize = elemsize;

	}

	state->ops = &interface_csr_ops;

	register_new_data(state, home_node, 0);
}

static inline uint32_t footprint_csr_interface_generic(uint32_t (*hash_func)(uint32_t input, uint32_t hstate), data_state *state, uint32_t hstate)
{
	uint32_t hash;

	hash = hstate;
	hash = hash_func(starpu_get_csr_nnz(state), hash);

	return hash;
}

static uint32_t footprint_csr_interface_crc32(data_state *state, uint32_t hstate)
{
	return footprint_csr_interface_generic(crc32_be, state, hstate);
}



struct dumped_csr_interface_s {
	uint32_t nnz;
	uint32_t nrow;
	uintptr_t nzval;
	uint32_t *colind;
	uint32_t *rowptr;
	uint32_t firstentry;
	uint32_t elemsize;
}  __attribute__ ((packed));

static size_t dump_csr_interface(starpu_data_interface_t *interface, void *_buffer)
{
	/* yes, that's DIRTY ... */
	struct dumped_csr_interface_s *buffer = _buffer;

	buffer->nnz = (*interface).csr.nnz;
	buffer->nrow = (*interface).csr.nrow;
	buffer->nzval = (*interface).csr.nzval;
	buffer->colind = (*interface).csr.colind;
	buffer->rowptr = (*interface).csr.rowptr;
	buffer->firstentry = (*interface).csr.firstentry;
	buffer->elemsize = (*interface).csr.elemsize;

	return (sizeof(struct dumped_csr_interface_s));
}

/* offer an access to the data parameters */
uint32_t starpu_get_csr_nnz(struct starpu_data_state_t *state)
{
	return (state->interface[0].csr.nnz);
}

uint32_t starpu_get_csr_nrow(struct starpu_data_state_t *state)
{
	return (state->interface[0].csr.nrow);
}

uint32_t starpu_get_csr_firstentry(struct starpu_data_state_t *state)
{
	return (state->interface[0].csr.firstentry);
}

size_t starpu_get_csr_elemsize(struct starpu_data_state_t *state)
{
	return (state->interface[0].csr.elemsize);
}

uintptr_t starpu_get_csr_local_nzval(struct starpu_data_state_t *state)
{
	unsigned node;
	node = get_local_memory_node();

	STARPU_ASSERT(state->per_node[node].allocated);

	return (state->interface[node].csr.nzval);
}

uint32_t *starpu_get_csr_local_colind(struct starpu_data_state_t *state)
{
	unsigned node;
	node = get_local_memory_node();

	STARPU_ASSERT(state->per_node[node].allocated);

	return (state->interface[node].csr.colind);
}

uint32_t *starpu_get_csr_local_rowptr(struct starpu_data_state_t *state)
{
	unsigned node;
	node = get_local_memory_node();

	STARPU_ASSERT(state->per_node[node].allocated);

	return (state->interface[node].csr.rowptr);
}

static size_t csr_interface_get_size(struct starpu_data_state_t *state)
{
	size_t size;

	uint32_t nnz = starpu_get_csr_nnz(state);
	uint32_t nrow = starpu_get_csr_nrow(state);
	size_t elemsize = starpu_get_csr_elemsize(state);

	size = nnz*elemsize + nnz*sizeof(uint32_t) + (nrow+1)*sizeof(uint32_t);

	return size;
}

/* memory allocation/deallocation primitives for the BLAS interface */

/* returns the size of the allocated area */
static size_t allocate_csr_buffer_on_node(struct starpu_data_state_t *state, uint32_t dst_node)
{
	uintptr_t addr_nzval;
	uint32_t *addr_colind, *addr_rowptr;
	size_t allocated_memory;

	/* we need the 3 arrays to be allocated */

	uint32_t nnz = state->interface[dst_node].csr.nnz;
	uint32_t nrow = state->interface[dst_node].csr.nrow;
	size_t elemsize = state->interface[dst_node].csr.elemsize;

	node_kind kind = get_node_kind(dst_node);

	switch(kind) {
		case RAM:
			addr_nzval = (uintptr_t)malloc(nnz*elemsize);
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
			cublasAlloc(nnz, elemsize, (void **)&addr_nzval);
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
		nnz*elemsize + nnz*sizeof(uint32_t) + (nrow+1)*sizeof(uint32_t);

	/* update the data properly in consequence */
	state->interface[dst_node].csr.nzval = addr_nzval;
	state->interface[dst_node].csr.colind = addr_colind;
	state->interface[dst_node].csr.rowptr = addr_rowptr;
	
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

static void liberate_csr_buffer_on_node(starpu_data_interface_t *interface, uint32_t node)
{
	node_kind kind = get_node_kind(node);
	switch(kind) {
		case RAM:
			free((void*)interface->csr.nzval);
			free((void*)interface->csr.colind);
			free((void*)interface->csr.rowptr);
			break;
#ifdef USE_CUDA
		case CUDA_RAM:
			cublasFree((void*)interface->csr.nzval);
			cublasFree((void*)interface->csr.colind);
			cublasFree((void*)interface->csr.rowptr);
			break;
#endif
		default:
			assert(0);
	}
}

#ifdef USE_CUDA
static int copy_cublas_to_ram(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node)
{
	starpu_csr_interface_t *src_csr;
	starpu_csr_interface_t *dst_csr;

	src_csr = &state->interface[src_node].csr;
	dst_csr = &state->interface[dst_node].csr;

	uint32_t nnz = src_csr->nnz;
	uint32_t nrow = src_csr->nrow;
	size_t elemsize = src_csr->elemsize;

	cublasGetVector(nnz, elemsize, (uint8_t *)src_csr->nzval, 1, 
					(uint8_t *)dst_csr->nzval, 1);

	cublasGetVector(nnz, sizeof(uint32_t), (uint8_t *)src_csr->colind, 1, 
						(uint8_t *)dst_csr->colind, 1);

	cublasGetVector((nrow+1), sizeof(uint32_t), (uint8_t *)src_csr->rowptr, 1, 
						(uint8_t *)dst_csr->rowptr, 1);
	
	TRACE_DATA_COPY(src_node, dst_node, nnz*elemsize + (nnz+nrow+1)*sizeof(uint32_t));

	return 0;
}

static int copy_ram_to_cublas(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node)
{
	starpu_csr_interface_t *src_csr;
	starpu_csr_interface_t *dst_csr;

	src_csr = &state->interface[src_node].csr;
	dst_csr = &state->interface[dst_node].csr;

	uint32_t nnz = src_csr->nnz;
	uint32_t nrow = src_csr->nrow;
	size_t elemsize = src_csr->elemsize;

	cublasSetVector(nnz, elemsize, (uint8_t *)src_csr->nzval, 1, 
					(uint8_t *)dst_csr->nzval, 1);

	cublasSetVector(nnz, sizeof(uint32_t), (uint8_t *)src_csr->colind, 1, 
						(uint8_t *)dst_csr->colind, 1);

	cublasSetVector((nrow+1), sizeof(uint32_t), (uint8_t *)src_csr->rowptr, 1, 
						(uint8_t *)dst_csr->rowptr, 1);
	
	TRACE_DATA_COPY(src_node, dst_node, nnz*elemsize + (nnz+nrow+1)*sizeof(uint32_t));

	return 0;
}
#endif // USE_CUDA

/* as not all platform easily have a BLAS lib installed ... */
static int dummy_copy_ram_to_ram(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node)
{

	starpu_csr_interface_t *src_csr;
	starpu_csr_interface_t *dst_csr;

	src_csr = &state->interface[src_node].csr;
	dst_csr = &state->interface[dst_node].csr;

	uint32_t nnz = src_csr->nnz;
	uint32_t nrow = src_csr->nrow;
	size_t elemsize = src_csr->elemsize;

	memcpy((void *)dst_csr->nzval, (void *)src_csr->nzval, nnz*elemsize);

	memcpy((void *)dst_csr->colind, (void *)src_csr->colind, nnz*sizeof(uint32_t));

	memcpy((void *)dst_csr->rowptr, (void *)src_csr->rowptr, (nrow+1)*sizeof(uint32_t));

	TRACE_DATA_COPY(src_node, dst_node, nnz*elemsize + (nnz+nrow+1)*sizeof(uint32_t));

	return 0;
}
