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

static int dummy_copy_ram_to_ram(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node);
#ifdef USE_CUDA
static int copy_ram_to_cublas(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node);
static int copy_cublas_to_ram(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node);
static int copy_ram_to_cublas_async(data_state *state, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream);
static int copy_cublas_to_ram_async(data_state *state, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream);
#endif

static const struct copy_data_methods_s block_copy_data_methods_s = {
	.ram_to_ram = dummy_copy_ram_to_ram,
	.ram_to_spu = NULL,
#ifdef USE_CUDA
	.ram_to_cuda = copy_ram_to_cublas,
	.cuda_to_ram = copy_cublas_to_ram,
	.ram_to_cuda_async = copy_ram_to_cublas_async,
	.cuda_to_ram_async = copy_cublas_to_ram_async,
#endif
	.cuda_to_cuda = NULL,
	.cuda_to_spu = NULL,
	.spu_to_ram = NULL,
	.spu_to_cuda = NULL,
	.spu_to_spu = NULL
};


static size_t allocate_block_buffer_on_node(data_state *state, uint32_t dst_node);
static void liberate_block_buffer_on_node(starpu_data_interface_t *interface, uint32_t node);
static size_t dump_block_interface(starpu_data_interface_t *interface, void *buffer);
static size_t block_interface_get_size(struct starpu_data_state_t *state);
static uint32_t footprint_block_interface_crc32(data_state *state, uint32_t hstate);
static void display_block_interface(data_state *state, FILE *f);
#ifdef USE_GORDON
static int convert_block_to_gordon(starpu_data_interface_t *interface, uint64_t *ptr, gordon_strideSize_t *ss);
#endif

struct data_interface_ops_t interface_block_ops = {
	.allocate_data_on_node = allocate_block_buffer_on_node,
	.liberate_data_on_node = liberate_block_buffer_on_node,
	.copy_methods = &block_copy_data_methods_s,
	.dump_data_interface = dump_block_interface,
	.get_size = block_interface_get_size,
	.footprint = footprint_block_interface_crc32,
#ifdef USE_GORDON
	.convert_to_gordon = convert_block_to_gordon,
#endif
	.interfaceid = STARPU_BLOCK_INTERFACE_ID, 
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

static uint32_t footprint_block_interface_crc32(data_state *state, uint32_t hstate)
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

static void display_block_interface(data_state *state, FILE *f)
{
	starpu_block_interface_t *interface;

	interface = &state->interface[0].block;

	fprintf(f, "%u\t%u\t%u\t", interface->nx, interface->ny, interface->nz);
}

static size_t dump_block_interface(starpu_data_interface_t *interface, void *_buffer)
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

static size_t block_interface_get_size(struct starpu_data_state_t *state)
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

/* memory allocation/deallocation primitives for the BLOCK interface */

/* returns the size of the allocated area */
static size_t allocate_block_buffer_on_node(data_state *state, uint32_t dst_node)
{
	uintptr_t addr = 0;
	unsigned fail = 0;
	size_t allocated_memory;

#ifdef USE_CUDA
	cudaError_t status;
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
			status = cudaMalloc((void **)&addr, nx*ny*nz*elemsize);

			//fprintf(stderr, "cudaMalloc -> addr %p\n", addr);

			if (!addr || status != cudaSuccess)
			{
				if (STARPU_UNLIKELY(status != cudaErrorMemoryAllocation))
					CUDA_REPORT_ERROR(status);

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

static void liberate_block_buffer_on_node(starpu_data_interface_t *interface, uint32_t node)
{
#ifdef USE_CUDA
	cudaError_t status;
#endif

	node_kind kind = get_node_kind(node);
	switch(kind) {
		case RAM:
			free((void*)interface->block.ptr);
			break;
#ifdef USE_CUDA
		case CUDA_RAM:
			status = cudaFree((void*)interface->blas.ptr);
			if (STARPU_UNLIKELY(status))
				CUDA_REPORT_ERROR(status);

			break;
#endif
		default:
			assert(0);
	}
}

#ifdef USE_CUDA
static int copy_cublas_to_ram(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	starpu_block_interface_t *src_block;
	starpu_block_interface_t *dst_block;

	src_block = &state->interface[src_node].block;
	dst_block = &state->interface[dst_node].block;

	//fprintf(stderr, "COPY BLOCK -> RAM nx %d ny %d nz %d SRC ldy %d DST ldy %d\n", src_block->nx,  src_block->ny,  src_block->nz,  src_block->ldy, dst_block->ldy);

	if ((src_block->nx == src_block->ldy) && (src_block->ldy == dst_block->ldy))
	{
		/* we are lucky */
		cublasStatus st;
		st = cublasGetMatrix(src_block->nx*src_block->ny, src_block->nz, src_block->elemsize,
			(uint8_t *)src_block->ptr, src_block->ldz,
			(uint8_t *)dst_block->ptr, dst_block->ldz);
		STARPU_ASSERT(st == CUBLAS_STATUS_SUCCESS);
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
	
	cudaThreadSynchronize();

	TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->elemsize*src_block->elemsize);

	return 0;
}

static int copy_cublas_to_ram_async(data_state *state, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream)
{
	starpu_block_interface_t *src_block;
	starpu_block_interface_t *dst_block;

	src_block = &state->interface[src_node].block;
	dst_block = &state->interface[dst_node].block;

	uint32_t nx = src_block->nx;
	uint32_t ny = src_block->ny;
	uint32_t nz = src_block->nz;
	size_t elemsize = src_block->elemsize;

	cudaError_t cures;

	int ret;

	/* We may have a contiguous buffer for the entire block, or contiguous
	 * plans within the block, we can avoid many small transfers that way */
	if ((nx == src_block->ldy) && (src_block->ldy == dst_block->ldy))
	{
		/* Is that a single contiguous buffer ? */
		if (((nx*ny) == src_block->ldz) && (src_block->ldz == dst_block->ldz))
		{
			cures = cudaMemcpyAsync((char *)dst_block->ptr, (char *)src_block->ptr,
					nx*ny*nz*elemsize, cudaMemcpyDeviceToHost, *stream);
			if (STARPU_UNLIKELY(cures))
			{
				cures = cudaMemcpy((char *)dst_block->ptr, (char *)src_block->ptr,
					nx*nx*ny*elemsize, cudaMemcpyDeviceToHost);
				if (STARPU_UNLIKELY(cures))
					CUDA_REPORT_ERROR(cures);
				cudaThreadSynchronize();

				ret = 0;
			}
			else {
				ret = EAGAIN;
			}
			
		}
		else {
			/* Are all plans contiguous */
			cures = cudaMemcpy2DAsync((char *)dst_block->ptr, dst_block->ldz*elemsize,
					(char *)src_block->ptr, src_block->ldz*elemsize,
					nx*ny*elemsize, nz, cudaMemcpyDeviceToHost, *stream);
			if (STARPU_UNLIKELY(cures))
			{
				cures = cudaMemcpy2D((char *)dst_block->ptr, dst_block->ldz*elemsize,
						(char *)src_block->ptr, src_block->ldz*elemsize,
						nx*ny*elemsize, nz, cudaMemcpyDeviceToHost);
				if (STARPU_UNLIKELY(cures))
					CUDA_REPORT_ERROR(cures);
				cudaThreadSynchronize();

				ret = 0;
			}
			else {
				ret = EAGAIN;
			}
		}
	}
	else {
		/* Default case: we transfer all lines one by one: ny*nz transfers */
		unsigned layer;
		for (layer = 0; layer < src_block->nz; layer++)
		{
			uint8_t *src_ptr = ((uint8_t *)src_block->ptr) 
						+ src_block->ldz*src_block->elemsize;
			uint8_t *dst_ptr = ((uint8_t *)dst_block->ptr) 
						+ dst_block->ldz*dst_block->elemsize;

			cures = cudaMemcpy2DAsync((char *)dst_ptr, dst_block->ldy*elemsize,
					(char *)src_ptr, src_block->ldy*elemsize,
					nx*elemsize, ny, cudaMemcpyDeviceToHost, *stream);

			if (STARPU_UNLIKELY(cures))
			{
				/* I don't know how to do that "better" */
				goto no_async_default;
			}

		}
		
		ret = EAGAIN;

	}

	TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);

	return ret;

no_async_default:

	{
	unsigned layer;
	for (layer = 0; layer < src_block->nz; layer++)
	{
		uint8_t *src_ptr = ((uint8_t *)src_block->ptr) 
					+ src_block->ldz*src_block->elemsize;
		uint8_t *dst_ptr = ((uint8_t *)dst_block->ptr) 
					+ dst_block->ldz*dst_block->elemsize;

		cures = cudaMemcpy2D((char *)dst_ptr, dst_block->ldy*elemsize,
				(char *)src_ptr, src_block->ldy*elemsize,
				nx*elemsize, ny, cudaMemcpyDeviceToHost);

		if (STARPU_UNLIKELY(cures))
			CUDA_REPORT_ERROR(cures);
		
	}
	cudaThreadSynchronize();

	TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);
	return 0;
	}
}



static int copy_ram_to_cublas_async(data_state *state, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream)
{
	starpu_block_interface_t *src_block;
	starpu_block_interface_t *dst_block;

	src_block = &state->interface[src_node].block;
	dst_block = &state->interface[dst_node].block;

	uint32_t nx = src_block->nx;
	uint32_t ny = src_block->ny;
	uint32_t nz = src_block->nz;
	size_t elemsize = src_block->elemsize;

	cudaError_t cures;

	int ret;

	/* We may have a contiguous buffer for the entire block, or contiguous
	 * plans within the block, we can avoid many small transfers that way */
	if ((nx == src_block->ldy) && (src_block->ldy == dst_block->ldy))
	{
		/* Is that a single contiguous buffer ? */
		if (((nx*ny) == src_block->ldz) && (src_block->ldz == dst_block->ldz))
		{
			cures = cudaMemcpyAsync((char *)dst_block->ptr, (char *)src_block->ptr,
					nx*ny*nz*elemsize, cudaMemcpyHostToDevice, *stream);
			if (STARPU_UNLIKELY(cures))
			{
				cures = cudaMemcpy((char *)dst_block->ptr, (char *)src_block->ptr,
					nx*nx*ny*elemsize, cudaMemcpyHostToDevice);
				if (STARPU_UNLIKELY(cures))
					CUDA_REPORT_ERROR(cures);
				cudaThreadSynchronize();

				ret = 0;
			}
			else {
				ret = EAGAIN;
			}
			
		}
		else {
			/* Are all plans contiguous */
			cures = cudaMemcpy2DAsync((char *)dst_block->ptr, dst_block->ldz*elemsize,
					(char *)src_block->ptr, src_block->ldz*elemsize,
					nx*ny*elemsize, nz, cudaMemcpyHostToDevice, *stream);
			if (STARPU_UNLIKELY(cures))
			{
				cures = cudaMemcpy2D((char *)dst_block->ptr, dst_block->ldz*elemsize,
						(char *)src_block->ptr, src_block->ldz*elemsize,
						nx*ny*elemsize, nz, cudaMemcpyHostToDevice);
				if (STARPU_UNLIKELY(cures))
					CUDA_REPORT_ERROR(cures);
				cudaThreadSynchronize();

				ret = 0;
			}
			else {
				ret = EAGAIN;
			}
		}
	}
	else {
		/* Default case: we transfer all lines one by one: ny*nz transfers */
		unsigned layer;
		for (layer = 0; layer < src_block->nz; layer++)
		{
			uint8_t *src_ptr = ((uint8_t *)src_block->ptr) 
						+ src_block->ldz*src_block->elemsize;
			uint8_t *dst_ptr = ((uint8_t *)dst_block->ptr) 
						+ dst_block->ldz*dst_block->elemsize;

			cures = cudaMemcpy2DAsync((char *)dst_ptr, dst_block->ldy*elemsize,
					(char *)src_ptr, src_block->ldy*elemsize,
					nx*elemsize, ny, cudaMemcpyHostToDevice, *stream);

			if (STARPU_UNLIKELY(cures))
			{
				/* I don't know how to do that "better" */
				goto no_async_default;
			}

		}
		
		ret = EAGAIN;

	}

	TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);

	return ret;

no_async_default:

	{
	unsigned layer;
	for (layer = 0; layer < src_block->nz; layer++)
	{
		uint8_t *src_ptr = ((uint8_t *)src_block->ptr) 
					+ src_block->ldz*src_block->elemsize;
		uint8_t *dst_ptr = ((uint8_t *)dst_block->ptr) 
					+ dst_block->ldz*dst_block->elemsize;

		cures = cudaMemcpy2D((char *)dst_ptr, dst_block->ldy*elemsize,
				(char *)src_ptr, src_block->ldy*elemsize,
				nx*elemsize, ny, cudaMemcpyHostToDevice);

		if (STARPU_UNLIKELY(cures))
			CUDA_REPORT_ERROR(cures);
		
	}
	cudaThreadSynchronize();

	TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);
	return 0;
	}
}

static int copy_ram_to_cublas(data_state *state, uint32_t src_node, uint32_t dst_node)
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

	cudaThreadSynchronize();

	TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);

	return 0;
}
#endif // USE_CUDA

/* as not all platform easily have a BLAS lib installed ... */
static int dummy_copy_ram_to_ram(data_state *state, uint32_t src_node, uint32_t dst_node)
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

	return 0;
}
