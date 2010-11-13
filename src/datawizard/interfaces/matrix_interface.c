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

static int copy_ram_to_ram(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)));
#ifdef STARPU_USE_CUDA
static int copy_ram_to_cuda(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)));
static int copy_cuda_to_ram(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)));
static int copy_cuda_to_cuda(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)));
static int copy_ram_to_cuda_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), cudaStream_t *stream);
static int copy_cuda_to_ram_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), cudaStream_t *stream);
#endif
#ifdef STARPU_USE_OPENCL
static int copy_ram_to_opencl(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)));
static int copy_opencl_to_ram(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)));
static int copy_ram_to_opencl_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), void *_event);
static int copy_opencl_to_ram_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), void *_event);
#endif

static const struct starpu_data_copy_methods matrix_copy_data_methods_s = {
	.ram_to_ram = copy_ram_to_ram,
	.ram_to_spu = NULL,
#ifdef STARPU_USE_CUDA
	.ram_to_cuda = copy_ram_to_cuda,
	.cuda_to_ram = copy_cuda_to_ram,
	.ram_to_cuda_async = copy_ram_to_cuda_async,
	.cuda_to_ram_async = copy_cuda_to_ram_async,
	.cuda_to_cuda = copy_cuda_to_cuda,
#endif
#ifdef STARPU_USE_OPENCL
	.ram_to_opencl = copy_ram_to_opencl,
	.opencl_to_ram = copy_opencl_to_ram,
        .ram_to_opencl_async = copy_ram_to_opencl_async,
	.opencl_to_ram_async = copy_opencl_to_ram_async,
#endif
	.cuda_to_spu = NULL,
	.spu_to_ram = NULL,
	.spu_to_cuda = NULL,
	.spu_to_spu = NULL
};

static void register_matrix_handle(starpu_data_handle handle, uint32_t home_node, void *interface);
static ssize_t allocate_matrix_buffer_on_node(void *interface_, uint32_t dst_node);
static void free_matrix_buffer_on_node(void *interface, uint32_t node);
static size_t matrix_interface_get_size(starpu_data_handle handle);
static uint32_t footprint_matrix_interface_crc32(starpu_data_handle handle);
static int matrix_compare(void *interface_a, void *interface_b);
static void display_matrix_interface(starpu_data_handle handle, FILE *f);
#ifdef STARPU_USE_GORDON
static int convert_matrix_to_gordon(void *interface, uint64_t *ptr, gordon_strideSize_t *ss); 
#endif

struct starpu_data_interface_ops_t _starpu_interface_matrix_ops = {
	.register_data_handle = register_matrix_handle,
	.allocate_data_on_node = allocate_matrix_buffer_on_node,
	.free_data_on_node = free_matrix_buffer_on_node,
	.copy_methods = &matrix_copy_data_methods_s,
	.get_size = matrix_interface_get_size,
	.footprint = footprint_matrix_interface_crc32,
	.compare = matrix_compare,
#ifdef STARPU_USE_GORDON
	.convert_to_gordon = convert_matrix_to_gordon,
#endif
	.interfaceid = STARPU_MATRIX_INTERFACE_ID, 
	.interface_size = sizeof(starpu_matrix_interface_t),
	.display = display_matrix_interface
};

#ifdef STARPU_USE_GORDON
static int convert_matrix_to_gordon(void *interface, uint64_t *ptr, gordon_strideSize_t *ss) 
{
	size_t elemsize = GET_MATRIX_ELEMSIZE(interface);
	uint32_t nx = STARPU_MATRIX_GET_NX(interface);
	uint32_t ny = STARPU_MATRIX_GET_NY(interface);
	uint32_t ld = STARPU_MATRIX_GET_LD(interface);

	*ptr = STARPU_MATRIX_GET_PTR(interface);

	/* The gordon_stride_init function may use a contiguous buffer
 	 * in case nx = ld (in that case, (*ss).size = elemsize*nx*ny */
	*ss = gordon_stride_init(ny, nx*elemsize, ld*elemsize);

	return 0;
}
#endif

static void register_matrix_handle(starpu_data_handle handle, uint32_t home_node, void *interface)
{
	starpu_matrix_interface_t *matrix_interface = interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		starpu_matrix_interface_t *local_interface =
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node) {
			local_interface->ptr = matrix_interface->ptr;
                        local_interface->dev_handle = matrix_interface->dev_handle;
                        local_interface->offset = matrix_interface->offset;
			local_interface->ld  = matrix_interface->ld;
		}
		else {
			local_interface->ptr = 0;
			local_interface->dev_handle = 0;
			local_interface->offset = 0;
			local_interface->ld  = 0;
		}

		local_interface->nx = matrix_interface->nx;
		local_interface->ny = matrix_interface->ny;
		local_interface->elemsize = matrix_interface->elemsize;
	}
}

/* declare a new data with the matrix interface */
void starpu_matrix_data_register(starpu_data_handle *handleptr, uint32_t home_node,
			uintptr_t ptr, uint32_t ld, uint32_t nx,
			uint32_t ny, size_t elemsize)
{
	starpu_matrix_interface_t interface = {
		.ptr = ptr,
		.ld = ld,
		.nx = nx,
		.ny = ny,
		.elemsize = elemsize,
                .dev_handle = ptr,
                .offset = 0
	};

	starpu_data_register(handleptr, home_node, &interface, &_starpu_interface_matrix_ops);
}

static uint32_t footprint_matrix_interface_crc32(starpu_data_handle handle)
{
	return _starpu_crc32_be(starpu_matrix_get_nx(handle), starpu_matrix_get_ny(handle));
}

static int matrix_compare(void *interface_a, void *interface_b)
{
	starpu_matrix_interface_t *matrix_a = interface_a;
	starpu_matrix_interface_t *matrix_b = interface_b;

	/* Two matricess are considered compatible if they have the same size */
	return ((matrix_a->nx == matrix_b->nx)
			&& (matrix_a->ny == matrix_b->ny)
			&& (matrix_a->elemsize == matrix_b->elemsize));
}

static void display_matrix_interface(starpu_data_handle handle, FILE *f)
{
	starpu_matrix_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	fprintf(f, "%u\t%u\t", interface->nx, interface->ny);
}

static size_t matrix_interface_get_size(starpu_data_handle handle)
{
	starpu_matrix_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	size_t size;
	size = (size_t)interface->nx*interface->ny*interface->elemsize; 

	return size;
}

/* offer an access to the data parameters */
uint32_t starpu_matrix_get_nx(starpu_data_handle handle)
{
	starpu_matrix_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->nx;
}

uint32_t starpu_matrix_get_ny(starpu_data_handle handle)
{
	starpu_matrix_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->ny;
}

uint32_t starpu_matrix_get_local_ld(starpu_data_handle handle)
{
	unsigned node;
	node = _starpu_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	starpu_matrix_interface_t *interface =
		starpu_data_get_interface_on_node(handle, node);

	return interface->ld;
}

uintptr_t starpu_matrix_get_local_ptr(starpu_data_handle handle)
{
	unsigned node;
	node = _starpu_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	starpu_matrix_interface_t *interface =
		starpu_data_get_interface_on_node(handle, node);

	return interface->ptr;
}

size_t starpu_matrix_get_elemsize(starpu_data_handle handle)
{
	starpu_matrix_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->elemsize;
}

/* memory allocation/deallocation primitives for the matrix interface */

/* returns the size of the allocated area */
static ssize_t allocate_matrix_buffer_on_node(void *interface_, uint32_t dst_node)
{
	uintptr_t addr = 0;
	unsigned fail = 0;
	ssize_t allocated_memory;

#ifdef STARPU_USE_CUDA
	cudaError_t status;
#endif

	starpu_matrix_interface_t *interface = interface_;

	uint32_t nx = interface->nx;
	uint32_t ny = interface->ny;
	uint32_t ld = nx; // by default
	size_t elemsize = interface->elemsize;

	starpu_node_kind kind = _starpu_get_node_kind(dst_node);

	switch(kind) {
		case STARPU_CPU_RAM:
			addr = (uintptr_t)malloc((size_t)nx*ny*elemsize);
			if (!addr) 
				fail = 1;

			break;
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_RAM:
			status = cudaMalloc((void **)&addr, (size_t)nx*ny*elemsize);
			if (!addr || status != cudaSuccess)
			{
				if (STARPU_UNLIKELY(status != cudaErrorMemoryAllocation))
					 STARPU_CUDA_REPORT_ERROR(status);
					
				fail = 1;
			}

			ld = nx;

			break;
#endif
#ifdef STARPU_USE_OPENCL
	        case STARPU_OPENCL_RAM:
			{
                                int ret;
                                void *ptr;
                                ret = _starpu_opencl_allocate_memory(&ptr, nx*ny*elemsize, CL_MEM_READ_WRITE);
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
		allocated_memory = (size_t)nx*ny*elemsize;

		/* update the data properly in consequence */
		interface->ptr = addr;
                interface->dev_handle = addr;
                interface->offset = 0;
		interface->ld = ld;
	} else {
		/* allocation failed */
		allocated_memory = -ENOMEM;
	}
	
	return allocated_memory;
}

static void free_matrix_buffer_on_node(void *interface, uint32_t node)
{
	starpu_matrix_interface_t *matrix_interface = interface;

#ifdef STARPU_USE_CUDA
	cudaError_t status;
#endif

	starpu_node_kind kind = _starpu_get_node_kind(node);
	switch(kind) {
		case STARPU_CPU_RAM:
			free((void*)matrix_interface->ptr);
			break;
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_RAM:
			status = cudaFree((void*)matrix_interface->ptr);			
			if (STARPU_UNLIKELY(status))
				STARPU_CUDA_REPORT_ERROR(status);

			break;
#endif
#ifdef STARPU_USE_OPENCL
                case STARPU_OPENCL_RAM:
                        clReleaseMemObject((void *)matrix_interface->ptr);
                        break;
#endif
		default:
			assert(0);
	}
}

#ifdef STARPU_USE_CUDA
static int copy_cuda_common(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), enum cudaMemcpyKind kind)
{
	starpu_matrix_interface_t *src_matrix = src_interface;
	starpu_matrix_interface_t *dst_matrix = dst_interface;

	size_t elemsize = src_matrix->elemsize;

	cudaError_t cures;
	cures = cudaMemcpy2D((char *)dst_matrix->ptr, dst_matrix->ld*elemsize,
			(char *)src_matrix->ptr, src_matrix->ld*elemsize,
			src_matrix->nx*elemsize, src_matrix->ny, kind);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	STARPU_TRACE_DATA_COPY(src_node, dst_node, (size_t)src_matrix->nx*src_matrix->ny*src_matrix->elemsize);

	return 0;
}


static int copy_cuda_to_ram(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)))
{
	return copy_cuda_common(src_interface, src_node, dst_interface, dst_node, cudaMemcpyDeviceToHost);
}

static int copy_ram_to_cuda(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)))
{
	return copy_cuda_common(src_interface, src_node, dst_interface, dst_node, cudaMemcpyHostToDevice);
}

static int copy_cuda_to_cuda(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)))
{
	return copy_cuda_common(src_interface, src_node, dst_interface, dst_node, cudaMemcpyDeviceToDevice);
}

static int copy_cuda_to_ram_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), cudaStream_t *stream)
{
	starpu_matrix_interface_t *src_matrix = src_interface;
	starpu_matrix_interface_t *dst_matrix = dst_interface;

	size_t elemsize = src_matrix->elemsize;

	cudaError_t cures;	
	cures = cudaMemcpy2DAsync((char *)dst_matrix->ptr, dst_matrix->ld*elemsize,
			(char *)src_matrix->ptr, (size_t)src_matrix->ld*elemsize,
			(size_t)src_matrix->nx*elemsize, src_matrix->ny,
			cudaMemcpyDeviceToHost, *stream);
	if (cures)
	{
		cures = cudaMemcpy2D((char *)dst_matrix->ptr, dst_matrix->ld*elemsize,
			(char *)src_matrix->ptr, (size_t)src_matrix->ld*elemsize,
			(size_t)src_matrix->nx*elemsize, (size_t)src_matrix->ny,
			cudaMemcpyDeviceToHost);

		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);

		cures = cudaThreadSynchronize();
		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);
		

		return 0;
	}

	STARPU_TRACE_DATA_COPY(src_node, dst_node, (size_t)src_matrix->nx*src_matrix->ny*src_matrix->elemsize);

	return -EAGAIN;
}

static int copy_ram_to_cuda_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), cudaStream_t *stream)
{
	starpu_matrix_interface_t *src_matrix = src_interface;
	starpu_matrix_interface_t *dst_matrix = dst_interface;

	size_t elemsize = src_matrix->elemsize;

	cudaError_t cures;
	cures = cudaMemcpy2DAsync((char *)dst_matrix->ptr, dst_matrix->ld*elemsize,
				(char *)src_matrix->ptr, src_matrix->ld*elemsize,
				src_matrix->nx*elemsize, src_matrix->ny,
				cudaMemcpyHostToDevice, *stream);
	if (cures)
	{
		cures = cudaMemcpy2D((char *)dst_matrix->ptr, dst_matrix->ld*elemsize,
				(char *)src_matrix->ptr, src_matrix->ld*elemsize,
				src_matrix->nx*elemsize, src_matrix->ny, cudaMemcpyHostToDevice);
		cudaThreadSynchronize();

		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);

		return 0;
	}

	STARPU_TRACE_DATA_COPY(src_node, dst_node, (size_t)src_matrix->nx*src_matrix->ny*src_matrix->elemsize);

	return -EAGAIN;
}

#endif // STARPU_USE_CUDA

#ifdef STARPU_USE_OPENCL
static int copy_ram_to_opencl_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), void *_event)
{
	starpu_matrix_interface_t *src_matrix = src_interface;
	starpu_matrix_interface_t *dst_matrix = dst_interface;
        int err,ret;

	/* XXX non contiguous matrices are not supported with OpenCL yet ! (TODO) */
	STARPU_ASSERT((src_matrix->ld == src_matrix->nx) && (dst_matrix->ld == dst_matrix->nx));

	err = _starpu_opencl_copy_ram_to_opencl_async_sync((void*)src_matrix->ptr, (cl_mem)dst_matrix->dev_handle,
                                                           src_matrix->nx*src_matrix->ny*src_matrix->elemsize,
                                                           dst_matrix->offset, (cl_event*)_event, &ret);
        if (STARPU_UNLIKELY(err))
                STARPU_OPENCL_REPORT_ERROR(err);

	STARPU_TRACE_DATA_COPY(src_node, dst_node, src_matrix->nx*src_matrix->ny*src_matrix->elemsize);

	return ret;
}

static int copy_opencl_to_ram_async(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)), void *_event)
{
	starpu_matrix_interface_t *src_matrix = src_interface;
	starpu_matrix_interface_t *dst_matrix = dst_interface;
        int err, ret;

	/* XXX non contiguous matrices are not supported with OpenCL yet ! (TODO) */
	STARPU_ASSERT((src_matrix->ld == src_matrix->nx) && (dst_matrix->ld == dst_matrix->nx));

        err = _starpu_opencl_copy_opencl_to_ram_async_sync((cl_mem)src_matrix->dev_handle, (void*)dst_matrix->ptr,
                                                           src_matrix->nx*src_matrix->ny*src_matrix->elemsize,
                                                           src_matrix->offset, (cl_event*)_event, &ret);

        if (STARPU_UNLIKELY(err))
                STARPU_OPENCL_REPORT_ERROR(err);

	STARPU_TRACE_DATA_COPY(src_node, dst_node, src_matrix->nx*src_matrix->ny*src_matrix->elemsize);

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

/* as not all platform easily have a  lib installed ... */
static int copy_ram_to_ram(void *src_interface, unsigned src_node __attribute__((unused)), void *dst_interface, unsigned dst_node __attribute__((unused)))
{
	starpu_matrix_interface_t *src_matrix = src_interface;
	starpu_matrix_interface_t *dst_matrix = dst_interface;

	unsigned y;
	uint32_t nx = dst_matrix->nx;
	uint32_t ny = dst_matrix->ny;
	size_t elemsize = dst_matrix->elemsize;

	uint32_t ld_src = src_matrix->ld;
	uint32_t ld_dst = dst_matrix->ld;

	uintptr_t ptr_src = src_matrix->ptr;
	uintptr_t ptr_dst = dst_matrix->ptr;


	for (y = 0; y < ny; y++)
	{
		uint32_t src_offset = y*ld_src*elemsize;
		uint32_t dst_offset = y*ld_dst*elemsize;

		memcpy((void *)(ptr_dst + dst_offset), 
			(void *)(ptr_src + src_offset), nx*elemsize);
	}

	STARPU_TRACE_DATA_COPY(src_node, dst_node, (size_t)nx*ny*elemsize);

	return 0;
}
