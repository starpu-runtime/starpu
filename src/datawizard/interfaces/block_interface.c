/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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
#include <datawizard/filters.h>

#include <common/hash.h>

#include <starpu_cuda.h>
#include <starpu_opencl.h>
#include <drivers/opencl/driver_opencl.h>

static int dummy_copy_ram_to_ram(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)));
#ifdef STARPU_USE_CUDA
static int copy_ram_to_cuda(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)));
static int copy_cuda_to_ram(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)));
static int copy_ram_to_cuda_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), cudaStream_t *stream);
static int copy_cuda_to_ram_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), cudaStream_t *stream);
#endif
#ifdef STARPU_USE_OPENCL
static int copy_ram_to_opencl(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)));
static int copy_opencl_to_ram(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)));
static int copy_ram_to_opencl_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), void *_event);
static int copy_opencl_to_ram_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), void *_event);
#endif

static const struct starpu_data_copy_methods block_copy_data_methods_s = {
	.ram_to_ram = dummy_copy_ram_to_ram,
	.ram_to_spu = NULL,
#ifdef STARPU_USE_CUDA
	.ram_to_cuda = copy_ram_to_cuda,
	.cuda_to_ram = copy_cuda_to_ram,
	.ram_to_cuda_async = copy_ram_to_cuda_async,
	.cuda_to_ram_async = copy_cuda_to_ram_async,
#endif
#ifdef STARPU_USE_OPENCL
	.ram_to_opencl = copy_ram_to_opencl,
	.opencl_to_ram = copy_opencl_to_ram,
        .ram_to_opencl_async = copy_ram_to_opencl_async,
	.opencl_to_ram_async = copy_opencl_to_ram_async,
#endif
	.cuda_to_cuda = NULL,
	.cuda_to_spu = NULL,
	.spu_to_ram = NULL,
	.spu_to_cuda = NULL,
	.spu_to_spu = NULL
};


static void register_block_handle(starpu_data_handle handle, uint32_t home_node, void *interface);
static size_t allocate_block_buffer_on_node(void *interface_, uint32_t dst_node);
static void free_block_buffer_on_node(void *interface, uint32_t node);
static size_t block_interface_get_size(starpu_data_handle handle);
static uint32_t footprint_block_interface_crc32(starpu_data_handle handle);
static int block_compare(void *interface_a, void *interface_b);
static void display_block_interface(starpu_data_handle handle, FILE *f);
#ifdef STARPU_USE_GORDON
static int convert_block_to_gordon(void *interface, uint64_t *ptr, gordon_strideSize_t *ss);
#endif

static struct starpu_data_interface_ops_t interface_block_ops = {
	.register_data_handle = register_block_handle,
	.allocate_data_on_node = allocate_block_buffer_on_node,
	.free_data_on_node = free_block_buffer_on_node,
	.copy_methods = &block_copy_data_methods_s,
	.get_size = block_interface_get_size,
	.footprint = footprint_block_interface_crc32,
	.compare = block_compare,
#ifdef STARPU_USE_GORDON
	.convert_to_gordon = convert_block_to_gordon,
#endif
	.interfaceid = STARPU_BLOCK_INTERFACE_ID, 
	.interface_size = sizeof(starpu_block_interface_t),
	.display = display_block_interface
};

#ifdef STARPU_USE_GORDON
int convert_block_to_gordon(void *interface, uint64_t *ptr, gordon_strideSize_t *ss) 
{
	/* TODO */
	STARPU_ABORT();

	return 0;
}
#endif

static void register_block_handle(starpu_data_handle handle, uint32_t home_node, void *interface)
{
	starpu_block_interface_t *block_interface = interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		starpu_block_interface_t *local_interface =
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node) {
			local_interface->ptr = block_interface->ptr;
                        local_interface->dev_handle = block_interface->dev_handle;
                        local_interface->offset = block_interface->offset;
			local_interface->ldy  = block_interface->ldy;
			local_interface->ldz  = block_interface->ldz;
		}
		else {
			local_interface->ptr = 0;
                        local_interface->dev_handle = 0;
                        local_interface->offset = 0;
			local_interface->ldy  = 0;
			local_interface->ldz  = 0;
		}

		local_interface->nx = block_interface->nx;
		local_interface->ny = block_interface->ny;
		local_interface->nz = block_interface->nz;
		local_interface->elemsize = block_interface->elemsize;
	}
}

/* declare a new data with the BLAS interface */
void starpu_block_data_register(starpu_data_handle *handleptr, uint32_t home_node,
			uintptr_t ptr, uint32_t ldy, uint32_t ldz, uint32_t nx,
			uint32_t ny, uint32_t nz, size_t elemsize)
{
	starpu_block_interface_t interface = {
		.ptr = ptr,
                .dev_handle = ptr,
                .offset = 0,
		.ldy = ldy,
		.ldz = ldz,
		.nx = nx,
		.ny = ny,
		.nz = nz,
		.elemsize = elemsize
	};

	starpu_data_register(handleptr, home_node, &interface, &interface_block_ops);
}

static uint32_t footprint_block_interface_crc32(starpu_data_handle handle)
{
	uint32_t hash;

	hash = _starpu_crc32_be(starpu_block_get_nx(handle), 0);
	hash = _starpu_crc32_be(starpu_block_get_ny(handle), hash);
	hash = _starpu_crc32_be(starpu_block_get_nz(handle), hash);

	return hash;
}

static int block_compare(void *interface_a, void *interface_b)
{
	starpu_block_interface_t *block_a = interface_a;
	starpu_block_interface_t *block_b = interface_b;

	/* Two matricess are considered compatible if they have the same size */
	return ((block_a->nx == block_b->nx)
			&& (block_a->ny == block_b->ny)
			&& (block_a->nz == block_b->nz)
			&& (block_a->elemsize == block_b->elemsize));
}

static void display_block_interface(starpu_data_handle handle, FILE *f)
{
	starpu_block_interface_t *interface;

	interface = starpu_data_get_interface_on_node(handle, 0);

	fprintf(f, "%u\t%u\t%u\t", interface->nx, interface->ny, interface->nz);
}

static size_t block_interface_get_size(starpu_data_handle handle)
{
	size_t size;
	starpu_block_interface_t *interface;

	interface = starpu_data_get_interface_on_node(handle, 0);

	size = interface->nx*interface->ny*interface->nz*interface->elemsize; 

	return size;
}

/* offer an access to the data parameters */
uint32_t starpu_block_get_nx(starpu_data_handle handle)
{
	starpu_block_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->nx;
}

uint32_t starpu_block_get_ny(starpu_data_handle handle)
{
	starpu_block_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->ny;
}

uint32_t starpu_block_get_nz(starpu_data_handle handle)
{
	starpu_block_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->nz;
}

uint32_t starpu_block_get_local_ldy(starpu_data_handle handle)
{
	unsigned node;
	node = _starpu_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));
	
	starpu_block_interface_t *interface =
		starpu_data_get_interface_on_node(handle, node);

	return interface->ldy;
}

uint32_t starpu_block_get_local_ldz(starpu_data_handle handle)
{
	unsigned node;
	node = _starpu_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	starpu_block_interface_t *interface =
		starpu_data_get_interface_on_node(handle, node);

	return interface->ldz;
}

uintptr_t starpu_block_get_local_ptr(starpu_data_handle handle)
{
	unsigned node;
	node = _starpu_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	starpu_block_interface_t *interface =
		starpu_data_get_interface_on_node(handle, node);

	return interface->ptr;
}

size_t starpu_block_get_elemsize(starpu_data_handle handle)
{
	starpu_block_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->elemsize;
}


/* memory allocation/deallocation primitives for the BLOCK interface */

/* returns the size of the allocated area */
static size_t allocate_block_buffer_on_node(void *interface_, uint32_t dst_node)
{
	uintptr_t addr = 0;
	unsigned fail = 0;
	size_t allocated_memory;

#ifdef STARPU_USE_CUDA
	cudaError_t status;
#endif
	starpu_block_interface_t *dst_block = interface_;

	uint32_t nx = dst_block->nx;
	uint32_t ny = dst_block->ny;
	uint32_t nz = dst_block->nz;
	size_t elemsize = dst_block->elemsize;

	starpu_node_kind kind = _starpu_get_node_kind(dst_node);

	switch(kind) {
		case STARPU_CPU_RAM:
			addr = (uintptr_t)malloc(nx*ny*nz*elemsize);
			if (!addr) 
				fail = 1;

			break;
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_RAM:
			status = cudaMalloc((void **)&addr, nx*ny*nz*elemsize);

			//fprintf(stderr, "cudaMalloc -> addr %p\n", addr);

			if (!addr || status != cudaSuccess)
			{
				if (STARPU_UNLIKELY(status != cudaErrorMemoryAllocation))
					STARPU_CUDA_REPORT_ERROR(status);

				fail = 1;
			}

			break;
#endif
#ifdef STARPU_USE_OPENCL
	        case STARPU_OPENCL_RAM:
			{
                                int ret;
                                void *ptr;
                                ret = _starpu_opencl_allocate_memory(&ptr, nx*ny*nz*elemsize, CL_MEM_READ_WRITE);
                                addr = (uintptr_t)ptr;
				if (ret) {
					fail = 1;
				}
				break;
			}
#endif
		default:
			assert(0);
	}

	if (!fail) {
		/* allocation succeeded */
		allocated_memory = nx*ny*nz*elemsize;

		/* update the data properly in consequence */
		dst_block->ptr = addr;
                dst_block->dev_handle = addr;
                dst_block->offset = 0;
		dst_block->ldy = nx;
		dst_block->ldz = nx*ny;
	} else {
		/* allocation failed */
		allocated_memory = 0;
	}
	
	return allocated_memory;
}

static void free_block_buffer_on_node(void *interface, uint32_t node)
{
	starpu_block_interface_t *block_interface = interface;

#ifdef STARPU_USE_CUDA
	cudaError_t status;
#endif

	starpu_node_kind kind = _starpu_get_node_kind(node);
	switch(kind) {
		case STARPU_CPU_RAM:
			free((void*)block_interface->ptr);
			break;
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_RAM:
			status = cudaFree((void*)block_interface->ptr);
			if (STARPU_UNLIKELY(status))
				STARPU_CUDA_REPORT_ERROR(status);

			break;
#endif
#ifdef STARPU_USE_OPENCL
                case STARPU_OPENCL_RAM:
                        clReleaseMemObject((void *)block_interface->ptr);
                        break;
#endif
		default:
			assert(0);
	}
}

#ifdef STARPU_USE_CUDA
static int copy_cuda_to_ram(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)))
{
	starpu_block_interface_t *src_block = src_interface;
	starpu_block_interface_t *dst_block = dst_interface;

	uint32_t nx = src_block->nx;
	uint32_t ny = src_block->ny;
	uint32_t nz = src_block->nz;
	size_t elemsize = src_block->elemsize;

	cudaError_t cures;

	if ((nx == src_block->ldy) && (src_block->ldy == dst_block->ldy))
	{
		/* Is that a single contiguous buffer ? */
		if (((nx*ny) == src_block->ldz) && (src_block->ldz == dst_block->ldz))
		{
                        cures = cudaMemcpy((char *)dst_block->ptr, (char *)src_block->ptr,
                                           nx*ny*nz*elemsize, cudaMemcpyDeviceToHost);
                        if (STARPU_UNLIKELY(cures))
                                STARPU_CUDA_REPORT_ERROR(cures);
                }
		else {
			/* Are all plans contiguous */
                        cures = cudaMemcpy2D((char *)dst_block->ptr, dst_block->ldz*elemsize,
                                             (char *)src_block->ptr, src_block->ldz*elemsize,
                                             nx*ny*elemsize, nz, cudaMemcpyDeviceToHost);
                        if (STARPU_UNLIKELY(cures))
                                STARPU_CUDA_REPORT_ERROR(cures);
                }
	}
	else {
		/* Default case: we transfer all lines one by one: ny*nz transfers */
		unsigned layer;
		for (layer = 0; layer < src_block->nz; layer++)
		{
			uint8_t *src_ptr = ((uint8_t *)src_block->ptr) + layer*src_block->ldz*src_block->elemsize;
			uint8_t *dst_ptr = ((uint8_t *)dst_block->ptr) + layer*dst_block->ldz*dst_block->elemsize;

			cures = cudaMemcpy2D((char *)dst_ptr, dst_block->ldy*elemsize,
                                             (char *)src_ptr, src_block->ldy*elemsize,
                                             nx*elemsize, ny, cudaMemcpyDeviceToHost);

			if (STARPU_UNLIKELY(cures))
				STARPU_CUDA_REPORT_ERROR(cures);
		}
	}

	cudaThreadSynchronize();

	STARPU_TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->elemsize*src_block->elemsize);

	return 0;
}

static int copy_cuda_to_ram_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), cudaStream_t *stream)
{
	starpu_block_interface_t *src_block = src_interface;
	starpu_block_interface_t *dst_block = dst_interface;

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
					nx*ny*nz*elemsize, cudaMemcpyDeviceToHost);
				if (STARPU_UNLIKELY(cures))
					STARPU_CUDA_REPORT_ERROR(cures);
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
					STARPU_CUDA_REPORT_ERROR(cures);
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
			uint8_t *src_ptr = ((uint8_t *)src_block->ptr) + layer*src_block->ldz*src_block->elemsize;
			uint8_t *dst_ptr = ((uint8_t *)dst_block->ptr) + layer*dst_block->ldz*dst_block->elemsize;

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

	STARPU_TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);

	return ret;

no_async_default:

	{
	unsigned layer;
	for (layer = 0; layer < src_block->nz; layer++)
	{
		uint8_t *src_ptr = ((uint8_t *)src_block->ptr) + layer*src_block->ldz*src_block->elemsize;
		uint8_t *dst_ptr = ((uint8_t *)dst_block->ptr) + layer*dst_block->ldz*dst_block->elemsize;

		cures = cudaMemcpy2D((char *)dst_ptr, dst_block->ldy*elemsize,
                                     (char *)src_ptr, src_block->ldy*elemsize,
                                     nx*elemsize, ny, cudaMemcpyDeviceToHost);

		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);
	}
	cudaThreadSynchronize();

	STARPU_TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);
	return 0;
	}
}



static int copy_ram_to_cuda_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), cudaStream_t *stream)
{
	starpu_block_interface_t *src_block = src_interface;
	starpu_block_interface_t *dst_block = dst_interface;

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
					nx*ny*nz*elemsize, cudaMemcpyHostToDevice);
				if (STARPU_UNLIKELY(cures))
					STARPU_CUDA_REPORT_ERROR(cures);
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
					STARPU_CUDA_REPORT_ERROR(cures);
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
			uint8_t *src_ptr = ((uint8_t *)src_block->ptr) + layer*src_block->ldz*src_block->elemsize;
			uint8_t *dst_ptr = ((uint8_t *)dst_block->ptr) + layer*dst_block->ldz*dst_block->elemsize;

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

	STARPU_TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);

	return ret;

no_async_default:

	{
	unsigned layer;
	for (layer = 0; layer < src_block->nz; layer++)
	{
		uint8_t *src_ptr = ((uint8_t *)src_block->ptr) + layer*src_block->ldz*src_block->elemsize;
		uint8_t *dst_ptr = ((uint8_t *)dst_block->ptr) + layer*dst_block->ldz*dst_block->elemsize;

		cures = cudaMemcpy2D((char *)dst_ptr, dst_block->ldy*elemsize,
                                     (char *)src_ptr, src_block->ldy*elemsize,
                                     nx*elemsize, ny, cudaMemcpyHostToDevice);

		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);
	}
	cudaThreadSynchronize();

	STARPU_TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);
	return 0;
	}
}

static int copy_ram_to_cuda(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)))
{
	starpu_block_interface_t *src_block = src_interface;
	starpu_block_interface_t *dst_block = dst_interface;

	uint32_t nx = src_block->nx;
	uint32_t ny = src_block->ny;
	uint32_t nz = src_block->nz;
	size_t elemsize = src_block->elemsize;

	cudaError_t cures;

	/* We may have a contiguous buffer for the entire block, or contiguous
	 * plans within the block, we can avoid many small transfers that way */
	if ((nx == src_block->ldy) && (src_block->ldy == dst_block->ldy))
	{
		/* Is that a single contiguous buffer ? */
		if (((nx*ny) == src_block->ldz) && (src_block->ldz == dst_block->ldz))
		{
			cures = cudaMemcpy((char *)dst_block->ptr, (char *)src_block->ptr,
                                           nx*ny*nz*elemsize, cudaMemcpyHostToDevice);
                }
                else {
			/* Are all plans contiguous */
			cures = cudaMemcpy2D((char *)dst_block->ptr, dst_block->ldz*elemsize,
                                             (char *)src_block->ptr, src_block->ldz*elemsize,
                                             nx*ny*elemsize, nz, cudaMemcpyHostToDevice);
                }
		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);
	}
	else {
		/* Default case: we transfer all lines one by one: ny*nz transfers */
		unsigned layer;
		for (layer = 0; layer < src_block->nz; layer++)
		{
			uint8_t *src_ptr = ((uint8_t *)src_block->ptr) + layer*src_block->ldz*src_block->elemsize;
			uint8_t *dst_ptr = ((uint8_t *)dst_block->ptr) + layer*dst_block->ldz*dst_block->elemsize;

			cures = cudaMemcpy2D((char *)dst_ptr, dst_block->ldy*elemsize,
                                             (char *)src_ptr, src_block->ldy*elemsize,
                                             nx*elemsize, ny, cudaMemcpyHostToDevice);

			if (STARPU_UNLIKELY(cures))
				STARPU_CUDA_REPORT_ERROR(cures);
		}
	}

	cudaThreadSynchronize();

	STARPU_TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);

	return 0;
}
#endif // STARPU_USE_CUDA

#ifdef STARPU_USE_OPENCL
static int copy_ram_to_opencl_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), void *_event)
{
	starpu_block_interface_t *src_block = src_interface;
	starpu_block_interface_t *dst_block = dst_interface;
        int err,ret;

	uint32_t nx = src_block->nx;
	uint32_t ny = src_block->ny;

	/* We may have a contiguous buffer for the entire block, or contiguous
	 * plans within the block, we can avoid many small transfers that way */
	if ((nx == src_block->ldy) && (src_block->ldy == dst_block->ldy))
	{
		/* Is that a single contiguous buffer ? */
		if (((nx*ny) == src_block->ldz) && (src_block->ldz == dst_block->ldz))
		{
                        err = _starpu_opencl_copy_ram_to_opencl_async_sync((void*)src_block->ptr, (cl_mem)dst_block->dev_handle,
                                                                           src_block->nx*src_block->ny*src_block->nz*src_block->elemsize,
                                                                           dst_block->offset, (cl_event*)_event, &ret);
                        if (STARPU_UNLIKELY(err))
                                STARPU_OPENCL_REPORT_ERROR(err);
                }
		else {
			/* Are all plans contiguous */
                        /* XXX non contiguous buffers are not properly supported yet. (TODO) */
                        STARPU_ASSERT(0);
                }
        }
	else {
		/* Default case: we transfer all lines one by one: ny*nz transfers */
		unsigned layer;
		for (layer = 0; layer < src_block->nz; layer++)
		{
                        unsigned j;
                        for(j=0 ; j<src_block->ny ; j++) {
                                void *ptr = (void*)src_block->ptr+(layer*src_block->ldz*src_block->elemsize)+(j*src_block->ldy*src_block->elemsize);
                                err = _starpu_opencl_copy_ram_to_opencl(ptr, (cl_mem)dst_block->dev_handle,
                                                                        src_block->nx*src_block->elemsize,
                                                                        layer*dst_block->ldz*dst_block->elemsize + j*dst_block->ldy*dst_block->elemsize
                                                                        + dst_block->offset, NULL);
                                if (STARPU_UNLIKELY(err))
                                        STARPU_OPENCL_REPORT_ERROR(err);
                        }

                        //                        int *foo = (int *)(src_block->ptr+(layer*src_block->ldz*src_block->elemsize));
                        //                        fprintf(stderr, "layer %d --> value %d\n", layer, foo[1]);
                        //                        const size_t buffer_origin[3] = {layer*src_block->ldz*src_block->elemsize, 0, 0};
                        //                        //const size_t buffer_origin[3] = {0, 0, 0};
                        //                        const size_t host_origin[3] = {layer*dst_block->ldz*dst_block->elemsize+dst_block->offset, 0, 0};
                        //                        size_t region[3] = {src_block->nx*src_block->elemsize,src_block->ny, 1};
                        //                        size_t buffer_row_pitch=region[0];
                        //                        size_t buffer_slice_pitch=region[1] * buffer_row_pitch;
                        //                        size_t host_row_pitch=region[0];
                        //                        size_t host_slice_pitch=region[1] * host_row_pitch;
                        //
                        //                        _starpu_opencl_copy_rect_ram_to_opencl((void *)src_block->ptr, (cl_mem)dst_block->dev_handle,
                        //                                                               buffer_origin, host_origin, region,
                        //                                                               buffer_row_pitch, buffer_slice_pitch,
                        //                                                               host_row_pitch, host_slice_pitch, NULL);
                }
        }

	STARPU_TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);

	return ret;
}

static int copy_opencl_to_ram_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), void *_event)
{
	starpu_block_interface_t *src_block = src_interface;
	starpu_block_interface_t *dst_block = dst_interface;
        int err, ret;

	/* We may have a contiguous buffer for the entire block, or contiguous
	 * plans within the block, we can avoid many small transfers that way */
	if ((src_block->nx == src_block->ldy) && (src_block->ldy == dst_block->ldy))
	{
		/* Is that a single contiguous buffer ? */
		if (((src_block->nx*src_block->ny) == src_block->ldz) && (src_block->ldz == dst_block->ldz))
		{
                        err = _starpu_opencl_copy_opencl_to_ram_async_sync((cl_mem)src_block->dev_handle, (void*)dst_block->ptr,
                                                                           src_block->nx*src_block->ny*src_block->nz*src_block->elemsize,
                                                                           src_block->offset, (cl_event*)_event, &ret);
                        if (STARPU_UNLIKELY(err))
                                STARPU_OPENCL_REPORT_ERROR(err);
                }
                else {
			/* Are all plans contiguous */
                        /* XXX non contiguous buffers are not properly supported yet. (TODO) */
                        STARPU_ASSERT(0);
                }
        }
	else {
		/* Default case: we transfer all lines one by one: ny*nz transfers */
                /* XXX non contiguous buffers are not properly supported yet. (TODO) */
		unsigned layer;
		for (layer = 0; layer < src_block->nz; layer++)
		{
                        unsigned j;
                        for(j=0 ; j<src_block->ny ; j++) {
                                void *ptr = (void *)dst_block->ptr+(layer*dst_block->ldz*dst_block->elemsize)+(j*dst_block->ldy*dst_block->elemsize);
                                err = _starpu_opencl_copy_opencl_to_ram((void*)src_block->dev_handle, ptr,
                                                                        src_block->nx*src_block->elemsize,
                                                                        layer*src_block->ldz*src_block->elemsize+j*src_block->ldy*src_block->elemsize+
                                                                        src_block->offset, NULL);
                        }
                        //                        const size_t buffer_origin[3] = {src_block->offset, 0, 0};
                        //                        const size_t host_origin[3] = {layer*src_block->ldz*src_block->elemsize, 0, 0};
                        //                        size_t region[3] = {src_block->nx*src_block->elemsize,src_block->ny, 1};
                        //                        size_t buffer_row_pitch=region[0];
                        //                        size_t buffer_slice_pitch=region[1] * buffer_row_pitch;
                        //                        size_t host_row_pitch=region[0];
                        //                        size_t host_slice_pitch=region[1] * host_row_pitch;
                        //
                        //                        _starpu_opencl_copy_rect_opencl_to_ram((cl_mem)src_block->dev_handle, (void *)dst_block->ptr,
                        //                                                               buffer_origin, host_origin, region,
                        //                                                               buffer_row_pitch, buffer_slice_pitch,
                        //                                                               host_row_pitch, host_slice_pitch, NULL);
                }
        }

	STARPU_TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);

	return ret;
}

static int copy_ram_to_opencl(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)))
{
        return copy_ram_to_opencl_async(src_interface, src_node, dst_interface, dst_node, NULL);
}

static int copy_opencl_to_ram(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)))
{
        return copy_opencl_to_ram_async(src_interface, src_node, dst_interface, dst_node, NULL);
}

#endif

/* as not all platform easily have a BLAS lib installed ... */
static int dummy_copy_ram_to_ram(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)))
{
	starpu_block_interface_t *src_block = src_interface;
	starpu_block_interface_t *dst_block = dst_interface;

	uint32_t nx = dst_block->nx;
	uint32_t ny = dst_block->ny;
	uint32_t nz = dst_block->nz;
	size_t elemsize = dst_block->elemsize;

	uint32_t ldy_src = src_block->ldy;
	uint32_t ldz_src = src_block->ldz;
	uint32_t ldy_dst = dst_block->ldy;
	uint32_t ldz_dst = dst_block->ldz;

	uintptr_t ptr_src = src_block->ptr;
	uintptr_t ptr_dst = dst_block->ptr;

	unsigned y, z;
	for (z = 0; z < nz; z++)
	for (y = 0; y < ny; y++)
	{
		uint32_t src_offset = (y*ldy_src + y*z*ldz_src)*elemsize;
		uint32_t dst_offset = (y*ldy_dst + y*z*ldz_dst)*elemsize;

		memcpy((void *)(ptr_dst + dst_offset), 
			(void *)(ptr_src + src_offset), nx*elemsize);
	}

	STARPU_TRACE_DATA_COPY(src_node, dst_node, nx*ny*nz*elemsize);

	return 0;
}
