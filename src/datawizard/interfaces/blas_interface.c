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
#include <cuda_runtime.h>
#endif

static int dummy_copy_ram_to_ram(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node);
#ifdef USE_CUDA
static int copy_ram_to_cublas(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node);
static int copy_cublas_to_ram(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node);
static int copy_ram_to_cublas_async(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream);
static int copy_cublas_to_ram_async(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream);
#endif

static const struct copy_data_methods_s blas_copy_data_methods_s = {
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

static size_t allocate_blas_buffer_on_node(data_state *state, uint32_t dst_node);
static void liberate_blas_buffer_on_node(starpu_data_interface_t *interface, uint32_t node);
static size_t dump_blas_interface(starpu_data_interface_t *interface, void *buffer);
static size_t blas_interface_get_size(struct starpu_data_state_t *state);
static uint32_t footprint_blas_interface_crc32(data_state *state, uint32_t hstate);
static void display_blas_interface(data_state *state, FILE *f);
#ifdef USE_GORDON
static int convert_blas_to_gordon(starpu_data_interface_t *interface, uint64_t *ptr, gordon_strideSize_t *ss); 
#endif

struct data_interface_ops_t interface_blas_ops = {
	.allocate_data_on_node = allocate_blas_buffer_on_node,
	.liberate_data_on_node = liberate_blas_buffer_on_node,
	.copy_methods = &blas_copy_data_methods_s,
	.dump_data_interface = dump_blas_interface,
	.get_size = blas_interface_get_size,
	.footprint = footprint_blas_interface_crc32,
#ifdef USE_GORDON
	.convert_to_gordon = convert_blas_to_gordon,
#endif
	.interfaceid = STARPU_BLAS_INTERFACE_ID, 
	.display = display_blas_interface
};

#ifdef USE_GORDON
static int convert_blas_to_gordon(starpu_data_interface_t *interface, uint64_t *ptr, gordon_strideSize_t *ss) 
{
	size_t elemsize = (*interface).blas.elemsize;
	uint32_t nx = (*interface).blas.nx;
	uint32_t ny = (*interface).blas.ny;
	uint32_t ld = (*interface).blas.ld;

	*ptr = (*interface).blas.ptr;

	/* The gordon_stride_init function may use a contiguous buffer
 	 * in case nx = ld (in that case, (*ss).size = elemsize*nx*ny */
	*ss = gordon_stride_init(ny, nx*elemsize, ld*elemsize);

	return 0;
}
#endif

/* declare a new data with the BLAS interface */
void starpu_register_blas_data(struct starpu_data_state_t **handle, uint32_t home_node,
			uintptr_t ptr, uint32_t ld, uint32_t nx,
			uint32_t ny, size_t elemsize)
{
	struct starpu_data_state_t *state =
		starpu_data_state_create(sizeof(starpu_blas_interface_t));

	STARPU_ASSERT(handle);
	*handle = state;

	unsigned node;
	for (node = 0; node < MAXNODES; node++)
	{
		starpu_blas_interface_t *local_interface =
			starpu_data_get_interface_on_node(state, node);

		if (node == home_node) {
			local_interface->ptr = ptr;
			local_interface->ld  = ld;
		}
		else {
			local_interface->ptr = 0;
			local_interface->ld  = 0;
		}

		local_interface->nx = nx;
		local_interface->ny = ny;
		local_interface->elemsize = elemsize;
	}

	state->ops = &interface_blas_ops;

	register_new_data(state, home_node, 0);
}

static inline uint32_t footprint_blas_interface_generic(uint32_t (*hash_func)(uint32_t input, uint32_t hstate), data_state *state, uint32_t hstate)
{
	uint32_t hash;

	hash = hstate;
	hash = hash_func(starpu_get_blas_nx(state), hash);
	hash = hash_func(starpu_get_blas_ny(state), hash);

	return hash;
}

static uint32_t footprint_blas_interface_crc32(data_state *state, uint32_t hstate)
{
	return footprint_blas_interface_generic(crc32_be, state, hstate);
}

struct dumped_blas_interface_s {
	uintptr_t ptr;
	uint32_t nx;
	uint32_t ny;
	uint32_t ld;
} __attribute__ ((packed));

static void display_blas_interface(data_state *state, FILE *f)
{
	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(state, 0);

	fprintf(f, "%u\t%u\t", interface->nx, interface->ny);
}

static size_t dump_blas_interface(starpu_data_interface_t *interface, void *_buffer)
{
	/* yes, that's DIRTY ... */
	struct dumped_blas_interface_s *buffer = _buffer;

	buffer->ptr = (*interface).blas.ptr;
	buffer->nx = (*interface).blas.nx;
	buffer->ny = (*interface).blas.ny;
	buffer->ld = (*interface).blas.ld;

	return (sizeof(struct dumped_blas_interface_s));
}

static size_t blas_interface_get_size(struct starpu_data_state_t *state)
{
	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(state, 0);

	size_t size;
	size = (size_t)interface->nx*interface->ny*interface->elemsize; 

	return size;
}

/* offer an access to the data parameters */
uint32_t starpu_get_blas_nx(data_state *state)
{
	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(state, 0);

	return interface->nx;
}

uint32_t starpu_get_blas_ny(data_state *state)
{
	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(state, 0);

	return interface->ny;
}

uint32_t starpu_get_blas_local_ld(data_state *state)
{
	unsigned node;
	node = get_local_memory_node();

	STARPU_ASSERT(state->per_node[node].allocated);

	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(state, node);

	return interface->ld;
}

uintptr_t starpu_get_blas_local_ptr(data_state *state)
{
	unsigned node;
	node = get_local_memory_node();

	STARPU_ASSERT(state->per_node[node].allocated);

	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(state, node);

	return interface->ptr;
}

size_t starpu_get_blas_elemsize(data_state *state)
{
	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(state, 0);

	return interface->elemsize;
}

/* memory allocation/deallocation primitives for the BLAS interface */

/* returns the size of the allocated area */
static size_t allocate_blas_buffer_on_node(data_state *state, uint32_t dst_node)
{
	uintptr_t addr = 0;
	unsigned fail = 0;
	size_t allocated_memory;

#ifdef USE_CUDA
	cudaError_t status;
	size_t pitch;
#endif

	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(state, dst_node);

	uint32_t nx = interface->nx;
	uint32_t ny = interface->ny;
	uint32_t ld = nx; // by default
	size_t elemsize = interface->elemsize;

	node_kind kind = get_node_kind(dst_node);

	switch(kind) {
		case RAM:
			addr = (uintptr_t)malloc((size_t)nx*ny*elemsize);
			if (!addr) 
				fail = 1;

			break;
#ifdef USE_CUDA
		case CUDA_RAM:
			status = cudaMallocPitch((void **)&addr, &pitch, (size_t)nx*elemsize, (size_t)ny);
			if (!addr || status != cudaSuccess)
			{
				if (STARPU_UNLIKELY(status != cudaErrorMemoryAllocation))
					 CUDA_REPORT_ERROR(status);
					
				fail = 1;
			}

			ld = (uint32_t)(pitch/elemsize);

			break;
#endif
		default:
			assert(0);
	}

	if (!fail) {
		/* allocation succeeded */
		allocated_memory = (size_t)nx*ny*elemsize;

		/* update the data properly in consequence */
		interface->ptr = addr;
		interface->ld = ld;
	} else {
		/* allocation failed */
		allocated_memory = 0;
	}
	
	return allocated_memory;
}

static void liberate_blas_buffer_on_node(starpu_data_interface_t *interface, uint32_t node)
{
#ifdef USE_CUDA
	cudaError_t status;
#endif

	node_kind kind = get_node_kind(node);
	switch(kind) {
		case RAM:
			free((void*)interface->blas.ptr);
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
	starpu_blas_interface_t *src_blas;
	starpu_blas_interface_t *dst_blas;

	src_blas = starpu_data_get_interface_on_node(state, src_node);
	dst_blas = starpu_data_get_interface_on_node(state, dst_node);

	size_t elemsize = src_blas->elemsize;

	cudaError_t cures;
	cures = cudaMemcpy2D((char *)dst_blas->ptr, dst_blas->ld*elemsize,
			(char *)src_blas->ptr, src_blas->ld*elemsize,
			src_blas->nx*elemsize, src_blas->ny, cudaMemcpyDeviceToHost);
	if (STARPU_UNLIKELY(cures))
		CUDA_REPORT_ERROR(cures);

	TRACE_DATA_COPY(src_node, dst_node, (size_t)src_blas->nx*src_blas->ny*src_blas->elemsize);

	return 0;
}

static int copy_ram_to_cublas(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	starpu_blas_interface_t *src_blas;
	starpu_blas_interface_t *dst_blas;

	src_blas = starpu_data_get_interface_on_node(state, src_node);
	dst_blas = starpu_data_get_interface_on_node(state, dst_node);

	size_t elemsize = src_blas->elemsize;

	cudaError_t cures;
	cures = cudaMemcpy2D((char *)dst_blas->ptr, dst_blas->ld*elemsize,
			(char *)src_blas->ptr, src_blas->ld*elemsize,
			src_blas->nx*elemsize, src_blas->ny, cudaMemcpyHostToDevice);
	if (STARPU_UNLIKELY(cures))
		CUDA_REPORT_ERROR(cures);
		
	cures = cudaThreadSynchronize();
	if (STARPU_UNLIKELY(cures))
		CUDA_REPORT_ERROR(cures);
		
	TRACE_DATA_COPY(src_node, dst_node, (size_t)src_blas->nx*src_blas->ny*src_blas->elemsize);

	return 0;
}

static int copy_cublas_to_ram_async(data_state *state, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream)
{
	starpu_blas_interface_t *src_blas;
	starpu_blas_interface_t *dst_blas;

	src_blas = starpu_data_get_interface_on_node(state, src_node);
	dst_blas = starpu_data_get_interface_on_node(state, dst_node);

	size_t elemsize = src_blas->elemsize;

	cudaError_t cures;	
	cures = cudaMemcpy2DAsync((char *)dst_blas->ptr, dst_blas->ld*elemsize,
			(char *)src_blas->ptr, (size_t)src_blas->ld*elemsize,
			(size_t)src_blas->nx*elemsize, src_blas->ny,
			cudaMemcpyDeviceToHost, *stream);
	if (cures)
	{
		cures = cudaMemcpy2D((char *)dst_blas->ptr, dst_blas->ld*elemsize,
			(char *)src_blas->ptr, (size_t)src_blas->ld*elemsize,
			(size_t)src_blas->nx*elemsize, (size_t)src_blas->ny,
			cudaMemcpyDeviceToHost);

		if (STARPU_UNLIKELY(cures))
			CUDA_REPORT_ERROR(cures);

		cures = cudaThreadSynchronize();
		if (STARPU_UNLIKELY(cures))
			CUDA_REPORT_ERROR(cures);
		

		return 0;
	}

	TRACE_DATA_COPY(src_node, dst_node, (size_t)src_blas->nx*src_blas->ny*src_blas->elemsize);

	return EAGAIN;
}

static int copy_ram_to_cublas_async(struct starpu_data_state_t *state, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream)
{
	starpu_blas_interface_t *src_blas;
	starpu_blas_interface_t *dst_blas;

	src_blas = starpu_data_get_interface_on_node(state, src_node);
	dst_blas = starpu_data_get_interface_on_node(state, dst_node);

	size_t elemsize = src_blas->elemsize;

	cudaError_t cures;
	cures = cudaMemcpy2DAsync((char *)dst_blas->ptr, dst_blas->ld*elemsize,
				(char *)src_blas->ptr, src_blas->ld*elemsize,
				src_blas->nx*elemsize, src_blas->ny,
				cudaMemcpyHostToDevice, *stream);
	if (cures)
	{
		cures = cudaMemcpy2D((char *)dst_blas->ptr, dst_blas->ld*elemsize,
				(char *)src_blas->ptr, src_blas->ld*elemsize,
				src_blas->nx*elemsize, src_blas->ny, cudaMemcpyHostToDevice);
		cudaThreadSynchronize();

		if (STARPU_UNLIKELY(cures))
			CUDA_REPORT_ERROR(cures);

		return 0;
	}

	TRACE_DATA_COPY(src_node, dst_node, (size_t)src_blas->nx*src_blas->ny*src_blas->elemsize);

	return EAGAIN;
}

#endif // USE_CUDA

/* as not all platform easily have a BLAS lib installed ... */
static int dummy_copy_ram_to_ram(data_state *state, uint32_t src_node, uint32_t dst_node)
{
	starpu_blas_interface_t *src_blas;
	starpu_blas_interface_t *dst_blas;

	src_blas = starpu_data_get_interface_on_node(state, src_node);
	dst_blas = starpu_data_get_interface_on_node(state, dst_node);

	unsigned y;
	uint32_t nx = dst_blas->nx;
	uint32_t ny = dst_blas->ny;
	size_t elemsize = dst_blas->elemsize;

	uint32_t ld_src = src_blas->ld;
	uint32_t ld_dst = dst_blas->ld;

	uintptr_t ptr_src = src_blas->ptr;
	uintptr_t ptr_dst = dst_blas->ptr;


	for (y = 0; y < ny; y++)
	{
		uint32_t src_offset = y*ld_src*elemsize;
		uint32_t dst_offset = y*ld_dst*elemsize;

		memcpy((void *)(ptr_dst + dst_offset), 
			(void *)(ptr_src + src_offset), nx*elemsize);
	}

	TRACE_DATA_COPY(src_node, dst_node, (size_t)nx*ny*elemsize);

	return 0;
}
