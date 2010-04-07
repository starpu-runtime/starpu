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
#include <datawizard/copy_driver.h>
#include <datawizard/hierarchy.h>

#include <common/hash.h>

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#endif

static int dummy_copy_ram_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node);
#ifdef STARPU_USE_CUDA
static int copy_ram_to_cuda(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node);
static int copy_cuda_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node);
static int copy_ram_to_cuda_async(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream);
static int copy_cuda_to_ram_async(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream);
#endif

static const struct starpu_copy_data_methods_s vector_copy_data_methods_s = {
	.ram_to_ram = dummy_copy_ram_to_ram,
	.ram_to_spu = NULL,
#ifdef STARPU_USE_CUDA
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

static void register_vector_handle(starpu_data_handle handle, uint32_t home_node, void *interface);
static size_t allocate_vector_buffer_on_node(starpu_data_handle handle, uint32_t dst_node);
static void liberate_vector_buffer_on_node(void *interface, uint32_t node);
static size_t vector_interface_get_size(starpu_data_handle handle);
static uint32_t footprint_vector_interface_crc32(starpu_data_handle handle);
static void display_vector_interface(starpu_data_handle handle, FILE *f);
#ifdef STARPU_USE_GORDON
static int convert_vector_to_gordon(void *interface, uint64_t *ptr, gordon_strideSize_t *ss); 
#endif

static struct starpu_data_interface_ops_t interface_vector_ops = {
	.register_data_handle = register_vector_handle,
	.allocate_data_on_node = allocate_vector_buffer_on_node,
	.liberate_data_on_node = liberate_vector_buffer_on_node,
	.copy_methods = &vector_copy_data_methods_s,
	.get_size = vector_interface_get_size,
	.footprint = footprint_vector_interface_crc32,
#ifdef STARPU_USE_GORDON
	.convert_to_gordon = convert_vector_to_gordon,
#endif
	.interfaceid = STARPU_VECTOR_INTERFACE_ID,
	.interface_size = sizeof(starpu_vector_interface_t), 
	.display = display_vector_interface
};

static void register_vector_handle(starpu_data_handle handle, uint32_t home_node, void *interface)
{
	starpu_vector_interface_t *vector_interface = interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		starpu_vector_interface_t *local_interface = 
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node) {
			local_interface->ptr = vector_interface->ptr;
		}
		else {
			local_interface->ptr = 0;
		}

		local_interface->nx = vector_interface->nx;
		local_interface->elemsize = vector_interface->elemsize;
	}
}

#ifdef STARPU_USE_GORDON
int convert_vector_to_gordon(void *interface, uint64_t *ptr, gordon_strideSize_t *ss) 
{
	starpu_vector_interface_t *vector_interface = interface;
	
	*ptr = vector_interface->ptr;
	(*ss).size = vector_interface->nx * vector_interface->elemsize;

	return 0;
}
#endif

/* declare a new data with the vector interface */
void starpu_register_vector_data(starpu_data_handle *handleptr, uint32_t home_node,
                        uintptr_t ptr, uint32_t nx, size_t elemsize)
{
	starpu_vector_interface_t vector = {
		.ptr = ptr,
		.nx = nx,
		.elemsize = elemsize
	};	

	_starpu_register_data_handle(handleptr, home_node, &vector, &interface_vector_ops); 
}


static uint32_t footprint_vector_interface_crc32(starpu_data_handle handle)
{
	return _starpu_crc32_be(starpu_get_vector_nx(handle), 0);
}

static void display_vector_interface(starpu_data_handle handle, FILE *f)
{
	starpu_vector_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	fprintf(f, "%u\t", interface->nx);
}

static size_t vector_interface_get_size(starpu_data_handle handle)
{
	size_t size;
	starpu_vector_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	size = interface->nx*interface->elemsize;

	return size;
}

/* offer an access to the data parameters */
uint32_t starpu_get_vector_nx(starpu_data_handle handle)
{
	starpu_vector_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->nx;
}

uintptr_t starpu_get_vector_local_ptr(starpu_data_handle handle)
{
	unsigned node;
	node = _starpu_get_local_memory_node();

	STARPU_ASSERT(starpu_test_if_data_is_allocated_on_node(handle, node));

	starpu_vector_interface_t *interface =
		starpu_data_get_interface_on_node(handle, node);

	return interface->ptr;
}

size_t starpu_get_vector_elemsize(starpu_data_handle handle)
{
	starpu_vector_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->elemsize;
}

/* memory allocation/deallocation primitives for the vector interface */

/* returns the size of the allocated area */
static size_t allocate_vector_buffer_on_node(starpu_data_handle handle, uint32_t dst_node)
{
	starpu_vector_interface_t *interface =
		starpu_data_get_interface_on_node(handle, dst_node);

	unsigned fail = 0;
	uintptr_t addr = 0;
	size_t allocated_memory;

	uint32_t nx = interface->nx;
	size_t elemsize = interface->elemsize;

	starpu_node_kind kind = _starpu_get_node_kind(dst_node);

#ifdef STARPU_USE_CUDA
	cudaError_t status;
#endif

	switch(kind) {
		case STARPU_RAM:
			addr = (uintptr_t)malloc(nx*elemsize);
			if (!addr)
				fail = 1;
			break;
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_RAM:
			status = cudaMalloc((void **)&addr, nx*elemsize);
			if (!addr || (status != cudaSuccess))
			{
				if (STARPU_UNLIKELY(status != cudaErrorMemoryAllocation))
					STARPU_CUDA_REPORT_ERROR(status);

				fail = 1;
			}
			break;
#endif
		default:
			assert(0);
	}

	if (fail)
		return 0;

	/* allocation succeeded */
	allocated_memory = nx*elemsize;

	/* update the data properly in consequence */
	interface->ptr = addr;
	
	return allocated_memory;
}

static void liberate_vector_buffer_on_node(void *interface, uint32_t node)
{
	starpu_vector_interface_t *vector_interface = interface;

	starpu_node_kind kind = _starpu_get_node_kind(node);
	switch(kind) {
		case STARPU_RAM:
			free((void*)vector_interface->ptr);
			break;
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_RAM:
			cudaFree((void*)vector_interface->ptr);
			break;
#endif
		default:
			assert(0);
	}
}

#ifdef STARPU_USE_CUDA
static int copy_cuda_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node)
{
	starpu_vector_interface_t *src_vector;
	starpu_vector_interface_t *dst_vector;

	src_vector = starpu_data_get_interface_on_node(handle, src_node);
	dst_vector = starpu_data_get_interface_on_node(handle, dst_node);

	cudaError_t cures;
	cures = cudaMemcpy((char *)dst_vector->ptr, (char *)src_vector->ptr, src_vector->nx*src_vector->elemsize, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	STARPU_TRACE_DATA_COPY(src_node, dst_node, src_vector->nx*src_vector->elemsize);

	return 0;
}

static int copy_ram_to_cuda(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node)
{
	starpu_vector_interface_t *src_vector;
	starpu_vector_interface_t *dst_vector;

	src_vector = starpu_data_get_interface_on_node(handle, src_node);
	dst_vector = starpu_data_get_interface_on_node(handle, dst_node);

	cudaError_t cures;
	cures = cudaMemcpy((char *)dst_vector->ptr, (char *)src_vector->ptr, src_vector->nx*src_vector->elemsize, cudaMemcpyHostToDevice);
	cudaThreadSynchronize();

	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	STARPU_TRACE_DATA_COPY(src_node, dst_node, src_vector->nx*src_vector->elemsize);

	return 0;
}

static int copy_cuda_to_ram_async(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream)
{
	starpu_vector_interface_t *src_vector;
	starpu_vector_interface_t *dst_vector;

	src_vector = starpu_data_get_interface_on_node(handle, src_node);
	dst_vector = starpu_data_get_interface_on_node(handle, dst_node);

	cudaError_t cures;
	cures = cudaMemcpyAsync((char *)dst_vector->ptr, (char *)src_vector->ptr, src_vector->nx*src_vector->elemsize, cudaMemcpyDeviceToHost, *stream);
	if (cures)
	{
		/* do it in a synchronous fashion */
		cures = cudaMemcpy((char *)dst_vector->ptr, (char *)src_vector->ptr, src_vector->nx*src_vector->elemsize, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();

		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);

		return 0;
	}

	STARPU_TRACE_DATA_COPY(src_node, dst_node, src_vector->nx*src_vector->elemsize);

	return EAGAIN;
}

static int copy_ram_to_cuda_async(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node, cudaStream_t *stream)
{
	starpu_vector_interface_t *src_vector;
	starpu_vector_interface_t *dst_vector;

	src_vector = starpu_data_get_interface_on_node(handle, src_node);
	dst_vector = starpu_data_get_interface_on_node(handle, dst_node);

	cudaError_t cures;
	
	cures = cudaMemcpyAsync((char *)dst_vector->ptr, (char *)src_vector->ptr, src_vector->nx*src_vector->elemsize, cudaMemcpyHostToDevice, *stream);
	if (cures)
	{
		/* do it in a synchronous fashion */
		cures = cudaMemcpy((char *)dst_vector->ptr, (char *)src_vector->ptr, src_vector->nx*src_vector->elemsize, cudaMemcpyHostToDevice);
		cudaThreadSynchronize();

		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);

		return 0;
	}

	STARPU_TRACE_DATA_COPY(src_node, dst_node, src_vector->nx*src_vector->elemsize);

	return EAGAIN;
}


#endif // STARPU_USE_CUDA

static int dummy_copy_ram_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node)
{
	starpu_vector_interface_t *src_vector;
	starpu_vector_interface_t *dst_vector;

	src_vector = starpu_data_get_interface_on_node(handle, src_node);
	dst_vector = starpu_data_get_interface_on_node(handle, dst_node);

	uint32_t nx = dst_vector->nx;
	size_t elemsize = dst_vector->elemsize;

	uintptr_t ptr_src = src_vector->ptr;
	uintptr_t ptr_dst = dst_vector->ptr;

	memcpy((void *)ptr_dst, (void *)ptr_src, nx*elemsize);

	STARPU_TRACE_DATA_COPY(src_node, dst_node, nx*elemsize);

	return 0;
}
