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
#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>
#include <datawizard/hierarchy.h>

#include <common/hash.h>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

static int dummy_copy_ram_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node);
#ifdef USE_CUDA
static int copy_ram_to_cuda(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node);
static int copy_cuda_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node);
static int copy_ram_to_cuda_async(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream);
static int copy_cuda_to_ram_async(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream);
#endif

static const struct copy_data_methods_s blas_copy_data_methods_s = {
	.ram_to_ram = dummy_copy_ram_to_ram,
	.ram_to_spu = NULL,
#ifdef USE_CUDA
	.ram_to_cuda = copy_ram_to_cuda,
	.cuda_to_ram = copy_cuda_to_ram,
	.ram_to_cuda_async = copy_ram_to_cuda_async,
	.cuda_to_ram_async = copy_cuda_to_ram_async,
#endif
	.cuda_to_cuda = NULL,
	.cuda_to_spu = NULL,
	.spu_to_ram = NULL,
	.spu_to_cuda = NULL,
	.spu_to_spu = NULL
};

static void register_blas_handle(starpu_data_handle handle, uint32_t home_node, void *interface);
static size_t allocate_blas_buffer_on_node(starpu_data_handle handle, uint32_t dst_node);
static void liberate_blas_buffer_on_node(void *interface, uint32_t node);
static size_t blas_interface_get_size(starpu_data_handle handle);
static uint32_t footprint_blas_interface_crc32(starpu_data_handle handle);
static void display_blas_interface(starpu_data_handle handle, FILE *f);
#ifdef USE_GORDON
static int convert_blas_to_gordon(void *interface, uint64_t *ptr, gordon_strideSize_t *ss); 
#endif

struct data_interface_ops_t interface_blas_ops = {
	.register_data_handle = register_blas_handle,
	.allocate_data_on_node = allocate_blas_buffer_on_node,
	.liberate_data_on_node = liberate_blas_buffer_on_node,
	.copy_methods = &blas_copy_data_methods_s,
	.get_size = blas_interface_get_size,
	.footprint = footprint_blas_interface_crc32,
#ifdef USE_GORDON
	.convert_to_gordon = convert_blas_to_gordon,
#endif
	.interfaceid = STARPU_BLAS_INTERFACE_ID, 
	.interface_size = sizeof(starpu_blas_interface_t),
	.display = display_blas_interface
};

#ifdef USE_GORDON
static int convert_blas_to_gordon(void *interface, uint64_t *ptr, gordon_strideSize_t *ss) 
{
	size_t elemsize = GET_BLAS_ELEMSIZE(interface);
	uint32_t nx = GET_BLAS_NX(interface);
	uint32_t ny = GET_BLAS_NY(interface);
	uint32_t ld = GET_BLAS_LD(interface);

	*ptr = GET_BLAS_PTR(interface);

	/* The gordon_stride_init function may use a contiguous buffer
 	 * in case nx = ld (in that case, (*ss).size = elemsize*nx*ny */
	*ss = gordon_stride_init(ny, nx*elemsize, ld*elemsize);

	return 0;
}
#endif

static void register_blas_handle(starpu_data_handle handle, uint32_t home_node, void *interface)
{
	starpu_blas_interface_t *blas_interface = interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		starpu_blas_interface_t *local_interface =
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node) {
			local_interface->ptr = blas_interface->ptr;
			local_interface->ld  = blas_interface->ld;
		}
		else {
			local_interface->ptr = 0;
			local_interface->ld  = 0;
		}

		local_interface->nx = blas_interface->nx;
		local_interface->ny = blas_interface->ny;
		local_interface->elemsize = blas_interface->elemsize;
	}
}

/* declare a new data with the BLAS interface */
void starpu_register_blas_data(starpu_data_handle *handleptr, uint32_t home_node,
			uintptr_t ptr, uint32_t ld, uint32_t nx,
			uint32_t ny, size_t elemsize)
{
	starpu_blas_interface_t interface = {
		.ptr = ptr,
		.ld = ld,
		.nx = nx,
		.ny = ny,
		.elemsize = elemsize
	};

	register_data_handle(handleptr, home_node, &interface, &interface_blas_ops);
}

static uint32_t footprint_blas_interface_crc32(starpu_data_handle handle)
{
	return crc32_be(starpu_get_blas_nx(handle), starpu_get_blas_ny(handle));
}

static void display_blas_interface(starpu_data_handle handle, FILE *f)
{
	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	fprintf(f, "%u\t%u\t", interface->nx, interface->ny);
}

static size_t blas_interface_get_size(starpu_data_handle handle)
{
	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	size_t size;
	size = (size_t)interface->nx*interface->ny*interface->elemsize; 

	return size;
}

/* offer an access to the data parameters */
uint32_t starpu_get_blas_nx(starpu_data_handle handle)
{
	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->nx;
}

uint32_t starpu_get_blas_ny(starpu_data_handle handle)
{
	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->ny;
}

uint32_t starpu_get_blas_local_ld(starpu_data_handle handle)
{
	unsigned node;
	node = get_local_memory_node();

	STARPU_ASSERT(starpu_test_if_data_is_allocated_on_node(handle, node));

	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(handle, node);

	return interface->ld;
}

uintptr_t starpu_get_blas_local_ptr(starpu_data_handle handle)
{
	unsigned node;
	node = get_local_memory_node();

	STARPU_ASSERT(starpu_test_if_data_is_allocated_on_node(handle, node));

	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(handle, node);

	return interface->ptr;
}

size_t starpu_get_blas_elemsize(starpu_data_handle handle)
{
	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->elemsize;
}

/* memory allocation/deallocation primitives for the BLAS interface */

/* returns the size of the allocated area */
static size_t allocate_blas_buffer_on_node(starpu_data_handle handle, uint32_t dst_node)
{
	uintptr_t addr = 0;
	unsigned fail = 0;
	size_t allocated_memory;

#ifdef USE_CUDA
	cudaError_t status;
	size_t pitch;
#endif

	starpu_blas_interface_t *interface =
		starpu_data_get_interface_on_node(handle, dst_node);

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

static void liberate_blas_buffer_on_node(void *interface, uint32_t node)
{
	starpu_blas_interface_t *blas_interface = interface;

#ifdef USE_CUDA
	cudaError_t status;
#endif

	node_kind kind = get_node_kind(node);
	switch(kind) {
		case RAM:
			free((void*)blas_interface->ptr);
			break;
#ifdef USE_CUDA
		case CUDA_RAM:
			status = cudaFree((void*)blas_interface->ptr);			
			if (STARPU_UNLIKELY(status))
				CUDA_REPORT_ERROR(status);

			break;
#endif
		default:
			assert(0);
	}
}

#ifdef USE_CUDA
static int copy_cuda_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node)
{
	starpu_blas_interface_t *src_blas;
	starpu_blas_interface_t *dst_blas;

	src_blas = starpu_data_get_interface_on_node(handle, src_node);
	dst_blas = starpu_data_get_interface_on_node(handle, dst_node);

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

static int copy_ram_to_cuda(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node)
{
	starpu_blas_interface_t *src_blas;
	starpu_blas_interface_t *dst_blas;

	src_blas = starpu_data_get_interface_on_node(handle, src_node);
	dst_blas = starpu_data_get_interface_on_node(handle, dst_node);

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

static int copy_cuda_to_ram_async(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream)
{
	starpu_blas_interface_t *src_blas;
	starpu_blas_interface_t *dst_blas;

	src_blas = starpu_data_get_interface_on_node(handle, src_node);
	dst_blas = starpu_data_get_interface_on_node(handle, dst_node);

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

static int copy_ram_to_cuda_async(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream)
{
	starpu_blas_interface_t *src_blas;
	starpu_blas_interface_t *dst_blas;

	src_blas = starpu_data_get_interface_on_node(handle, src_node);
	dst_blas = starpu_data_get_interface_on_node(handle, dst_node);

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
static int dummy_copy_ram_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node)
{
	starpu_blas_interface_t *src_blas;
	starpu_blas_interface_t *dst_blas;

	src_blas = starpu_data_get_interface_on_node(handle, src_node);
	dst_blas = starpu_data_get_interface_on_node(handle, dst_node);

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
