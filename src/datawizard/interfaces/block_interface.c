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

#include <common/config.h>
#include <datawizard/data_parameters.h>
#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>
#include <datawizard/hierarchy.h>

#include <common/hash.h>

#include <starpu.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

size_t allocate_block_buffer_on_node(data_state *state, uint32_t dst_node);
void liberate_block_buffer_on_node(starpu_data_interface_t *interface, uint32_t node);
int do_copy_block_buffer_1_to_1(data_state *state, uint32_t src_node, uint32_t dst_node);
size_t dump_block_interface(starpu_data_interface_t *interface, void *buffer);
size_t block_interface_get_size(struct starpu_data_state_t *state);
uint32_t footprint_block_interface_crc32(data_state *state, uint32_t hstate);
void display_block_interface(data_state *state, FILE *f);
#ifdef USE_GORDON
int convert_block_to_gordon(starpu_data_interface_t *interface, uint64_t *ptr, gordon_strideSize_t *ss);
#endif

struct data_interface_ops_t interface_block_ops = {
	.allocate_data_on_node = allocate_block_buffer_on_node,
	.liberate_data_on_node = liberate_block_buffer_on_node,
	.copy_data_1_to_1 = do_copy_block_buffer_1_to_1,
	.dump_data_interface = dump_block_interface,
	.get_size = block_interface_get_size,
	.footprint = footprint_block_interface_crc32,
#ifdef USE_GORDON
	.convert_to_gordon = convert_block_to_gordon,
#endif
	.interfaceid = BLOCK_INTERFACE, 
	.display = display_block_interface
};

#ifdef USE_GORDON
int convert_block_to_gordon(starpu_data_interface_t *interface, uint64_t *ptr, gordon_strideSize_t *ss) 
{
	/* TODO */
	STARPU_ASSERT(0);

	return 0;
}
#endif

/* declare a new data with the BLAS interface */
void starpu_register_block_data(struct starpu_data_state_t **handle, uint32_t home_node,
			uintptr_t ptr, uint32_t ldy, uint32_t ldz, uint32_t nx,
			uint32_t ny, uint32_t nz, size_t elemsize)
{
	struct starpu_data_state_t *state = calloc(1, sizeof(struct starpu_data_state_t));
	STARPU_ASSERT(state);

	STARPU_ASSERT(handle);
	*handle = state;

	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		starpu_block_interface_t *local_interface = &state->interface[node].block;

		if (node == home_node) {
			local_interface->ptr = ptr;
			local_interface->ldy  = ldy;
			local_interface->ldz  = ldz;
		}
		else {
			local_interface->ptr = 0;
			local_interface->ldy  = 0;
			local_interface->ldz  = 0;
		}

		local_interface->nx = nx;
		local_interface->ny = ny;
		local_interface->nz = nz;
		local_interface->elemsize = elemsize;
	}

	state->ops = &interface_block_ops;

	register_new_data(state, home_node, 0);
}

static inline uint32_t footprint_block_interface_generic(uint32_t (*hash_func)(uint32_t input, uint32_t hstate), data_state *state, uint32_t hstate)
{
	uint32_t hash;

	hash = hstate;
	hash = hash_func(starpu_get_block_nx(state), hash);
	hash = hash_func(starpu_get_block_ny(state), hash);
	hash = hash_func(starpu_get_block_nz(state), hash);

	return hash;
}

uint32_t footprint_block_interface_crc32(data_state *state, uint32_t hstate)
{
	return footprint_block_interface_generic(crc32_be, state, hstate);
}

struct dumped_block_interface_s {
	uintptr_t ptr;
	uint32_t nx;
	uint32_t ny;
	uint32_t nz;
	uint32_t ldy;
	uint32_t ldz;
} __attribute__ ((packed));

void display_block_interface(data_state *state, FILE *f)
{
	starpu_block_interface_t *interface;

	interface = &state->interface[0].block;

	fprintf(f, "%d\t%d\t%d\t", interface->nx, interface->ny, interface->nz);
}

size_t dump_block_interface(starpu_data_interface_t *interface, void *_buffer)
{
	/* yes, that's DIRTY ... */
	struct dumped_block_interface_s *buffer = _buffer;

	buffer->ptr = (*interface).block.ptr;
	buffer->nx = (*interface).block.nx;
	buffer->ny = (*interface).block.ny;
	buffer->nz = (*interface).block.nz;
	buffer->ldy = (*interface).block.ldy;
	buffer->ldz = (*interface).block.ldz;

	return (sizeof(struct dumped_block_interface_s));
}

size_t block_interface_get_size(struct starpu_data_state_t *state)
{
	size_t size;
	starpu_block_interface_t *interface;

	interface = &state->interface[0].block;

	size = interface->nx*interface->ny*interface->nz*interface->elemsize; 

	return size;
}

/* offer an access to the data parameters */
uint32_t starpu_get_block_nx(data_state *state)
{
	return (state->interface[0].block.nx);
}

uint32_t starpu_get_block_ny(data_state *state)
{
	return (state->interface[0].block.ny);
}

uint32_t starpu_get_block_nz(data_state *state)
{
	return (state->interface[0].block.nz);
}

uint32_t starpu_get_block_local_ldy(data_state *state)
{
	unsigned node;
	node = get_local_memory_node();

	STARPU_ASSERT(state->per_node[node].allocated);

	return (state->interface[node].block.ldy);
}

uint32_t starpu_get_block_local_ldz(data_state *state)
{
	unsigned node;
	node = get_local_memory_node();

	STARPU_ASSERT(state->per_node[node].allocated);

	return (state->interface[node].block.ldz);
}

uintptr_t starpu_get_block_local_ptr(data_state *state)
{
	unsigned node;
	node = get_local_memory_node();

	STARPU_ASSERT(state->per_node[node].allocated);

	return (state->interface[node].block.ptr);
}

/* memory allocation/deallocation primitives for the BLAS interface */

/* returns the size of the allocated area */
size_t allocate_block_buffer_on_node(data_state *state, uint32_t dst_node)
{
	uintptr_t addr = 0;
	unsigned fail = 0;
	size_t allocated_memory;

#ifdef USE_CUDA
	cublasStatus status;
#endif
	uint32_t nx = state->interface[dst_node].block.nx;
	uint32_t ny = state->interface[dst_node].block.ny;
	uint32_t nz = state->interface[dst_node].block.nz;
	size_t elemsize = state->interface[dst_node].block.elemsize;

	node_kind kind = get_node_kind(dst_node);

	switch(kind) {
		case RAM:
			addr = (uintptr_t)malloc(nx*ny*nz*elemsize);
			if (!addr) 
				fail = 1;

			break;
#ifdef USE_CUDA
		case CUDA_RAM:
			status = cublasAlloc(nx*ny*nz, elemsize, (void **)&addr);

			if (!addr || status != CUBLAS_STATUS_SUCCESS)
			{
				STARPU_ASSERT(status != CUBLAS_STATUS_INTERNAL_ERROR);
				STARPU_ASSERT(status != CUBLAS_STATUS_NOT_INITIALIZED);
				STARPU_ASSERT(status != CUBLAS_STATUS_INVALID_VALUE);
				STARPU_ASSERT(status == CUBLAS_STATUS_ALLOC_FAILED);
				fail = 1;
			}

			break;
#endif
		default:
			assert(0);
	}

	if (!fail) {
		/* allocation succeeded */
		allocated_memory = nx*ny*nz*elemsize;

		/* update the data properly in consequence */
		state->interface[dst_node].block.ptr = addr;
		state->interface[dst_node].block.ldy = nx;
		state->interface[dst_node].block.ldz = nx*ny;
	} else {
		/* allocation failed */
		allocated_memory = 0;
	}
	
	return allocated_memory;
}

void liberate_block_buffer_on_node(starpu_data_interface_t *interface, uint32_t node)
{
#ifdef USE_CUDA
	cublasStatus status;
#endif

	node_kind kind = get_node_kind(node);
	switch(kind) {
		case RAM:
			free((void*)interface->block.ptr);
			break;
#ifdef USE_CUDA
		case CUDA_RAM:
			status = cublasFree((void*)interface->block.ptr);
			
			STARPU_ASSERT(status != CUBLAS_STATUS_INTERNAL_ERROR);
			STARPU_ASSERT(status == CUBLAS_STATUS_SUCCESS);

			break;
#endif
		default:
			assert(0);
	}
}

#ifdef USE_CUDA
static void copy_cublas_to_ram(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	starpu_block_interface_t *src_block;
	starpu_block_interface_t *dst_block;

	src_block = &state->interface[src_node].block;
	dst_block = &state->interface[dst_node].block;

	if ((src_block->nx == src_block->ldy) && (src_block->ldy == dst_block->ldy))
	{
		/* we are lucky */
		cublasGetMatrix(src_block->nx*src_block->ny, src_block->nz, src_block->elemsize,
			(uint8_t *)src_block->ptr, src_block->ldz,
			(uint8_t *)dst_block->ptr, dst_block->ldz);
	}
	else {
		unsigned layer;
		for (layer = 0; layer < src_block->nz; layer++)
		{
			uint8_t *src_ptr = ((uint8_t *)src_block->ptr) 
						+ src_block->ldz*src_block->elemsize;
			uint8_t *dst_ptr = ((uint8_t *)dst_block->ptr) 
						+ dst_block->ldz*dst_block->elemsize;

			cublasGetMatrix(src_block->nx, src_block->ny, src_block->elemsize,
				src_ptr, src_block->ldy, dst_ptr, dst_block->ldy);
		}
	}

	TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->elemsize*src_block->elemsize);
}

static void copy_ram_to_cublas(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	starpu_block_interface_t *src_block;
	starpu_block_interface_t *dst_block;

	src_block = &state->interface[src_node].block;
	dst_block = &state->interface[dst_node].block;

	if ((src_block->nx == src_block->ldy) && (src_block->ldy == dst_block->ldy))
	{
		/* we are lucky */
		cublasSetMatrix(src_block->nx*src_block->ny, src_block->nz, src_block->elemsize,
			(uint8_t *)src_block->ptr, src_block->ldz,
			(uint8_t *)dst_block->ptr, dst_block->ldz);
	}
	else {
		unsigned layer;
		for (layer = 0; layer < src_block->nz; layer++)
		{
			uint8_t *src_ptr = ((uint8_t *)src_block->ptr) 
						+ src_block->ldz*src_block->elemsize;
			uint8_t *dst_ptr = ((uint8_t *)dst_block->ptr) 
						+ dst_block->ldz*dst_block->elemsize;

			cublasSetMatrix(src_block->nx, src_block->ny, src_block->elemsize,
				src_ptr, src_block->ldy, dst_ptr, dst_block->ldy);
		}
	}

	TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);
}
#endif // USE_CUDA

/* as not all platform easily have a BLAS lib installed ... */
static void dummy_copy_ram_to_ram(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	uint32_t nx = state->interface[dst_node].block.nx;
	uint32_t ny = state->interface[dst_node].block.ny;
	uint32_t nz = state->interface[dst_node].block.nz;
	size_t elemsize = state->interface[dst_node].block.elemsize;

	uint32_t ldy_src = state->interface[src_node].block.ldy;
	uint32_t ldz_src = state->interface[src_node].block.ldz;
	uint32_t ldy_dst = state->interface[dst_node].block.ldy;
	uint32_t ldz_dst = state->interface[dst_node].block.ldz;

	uintptr_t ptr_src = state->interface[src_node].block.ptr;
	uintptr_t ptr_dst = state->interface[dst_node].block.ptr;

	unsigned y, z;
	for (z = 0; z < nz; z++)
	for (y = 0; y < ny; y++)
	{
		uint32_t src_offset = (y*ldy_src + y*z*ldz_src)*elemsize;
		uint32_t dst_offset = (y*ldy_dst + y*z*ldz_dst)*elemsize;

		memcpy((void *)(ptr_dst + dst_offset), 
			(void *)(ptr_src + src_offset), nx*elemsize);
	}

	TRACE_DATA_COPY(src_node, dst_node, nx*ny*nz*elemsize);
}


int do_copy_block_buffer_1_to_1(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	node_kind src_kind = get_node_kind(src_node);
	node_kind dst_kind = get_node_kind(dst_node);

	switch (dst_kind) {
	case RAM:
		switch (src_kind) {
			case RAM:
				/* RAM -> RAM */
				 dummy_copy_ram_to_ram(state, src_node, dst_node);
				 break;
#ifdef USE_CUDA
			case CUDA_RAM:
				/* CUBLAS_RAM -> RAM */
				if (get_local_memory_node() == src_node)
				{
					/* only the proper CUBLAS thread can initiate this directly ! */
					copy_cublas_to_ram(state, src_node, dst_node);
				}
				else
				{
					/* put a request to the corresponding GPU */
		//			fprintf(stderr, "post_data_request state %p src %d dst %d\n", state, src_node, dst_node);
					post_data_request(state, src_node, dst_node);
		//			fprintf(stderr, "post %p OK\n", state);
				}
				break;
#endif
			case SPU_LS:
				STARPU_ASSERT(0); // TODO
				break;
			case UNUSED:
				printf("error node %d UNUSED\n", src_node);
			default:
				assert(0);
				break;
		}
		break;
#ifdef USE_CUDA
	case CUDA_RAM:
		switch (src_kind) {
			case RAM:
				/* RAM -> CUBLAS_RAM */
				/* only the proper CUBLAS thread can initiate this ! */
				STARPU_ASSERT(get_local_memory_node() == dst_node);
				copy_ram_to_cublas(state, src_node, dst_node);
				break;
			case CUDA_RAM:
			case SPU_LS:
				STARPU_ASSERT(0); // TODO 
				break;
			case UNUSED:
			default:
				STARPU_ASSERT(0);
				break;
		}
		break;
#endif
	case SPU_LS:
		STARPU_ASSERT(0); // TODO
		break;
	case UNUSED:
	default:
		assert(0);
		break;
	}

	return 0;
}

