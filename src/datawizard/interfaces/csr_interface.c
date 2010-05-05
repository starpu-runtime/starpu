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
#include <datawizard/filters.h>

#include <common/hash.h>

#ifdef STARPU_USE_OPENCL
#include <starpu_opencl.h>
#include <drivers/opencl/driver_opencl.h>
#endif

static int dummy_copy_ram_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node);
#ifdef STARPU_USE_CUDA
static int copy_ram_to_cuda(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node);
static int copy_cuda_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node);
#endif
#ifdef STARPU_USE_OPENCL
static int copy_ram_to_opencl(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node);
static int copy_opencl_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node);
#endif

static const struct starpu_copy_data_methods_s csr_copy_data_methods_s = {
	.ram_to_ram = dummy_copy_ram_to_ram,
	.ram_to_spu = NULL,
#ifdef STARPU_USE_CUDA
	.ram_to_cuda = copy_ram_to_cuda,
	.cuda_to_ram = copy_cuda_to_ram,
#endif
#ifdef STARPU_USE_OPENCL
	.ram_to_opencl = copy_ram_to_opencl,
	.opencl_to_ram = copy_opencl_to_ram,
#endif
	.cuda_to_cuda = NULL,
	.cuda_to_spu = NULL,
	.spu_to_ram = NULL,
	.spu_to_cuda = NULL,
	.spu_to_spu = NULL
};

static void register_csr_handle(starpu_data_handle handle, uint32_t home_node, void *interface);
static size_t allocate_csr_buffer_on_node(starpu_data_handle handle, uint32_t dst_node);
static void liberate_csr_buffer_on_node(void *interface, uint32_t node);
static size_t csr_interface_get_size(starpu_data_handle handle);
static uint32_t footprint_csr_interface_crc32(starpu_data_handle handle);

static struct starpu_data_interface_ops_t interface_csr_ops = {
	.register_data_handle = register_csr_handle,
	.allocate_data_on_node = allocate_csr_buffer_on_node,
	.liberate_data_on_node = liberate_csr_buffer_on_node,
	.copy_methods = &csr_copy_data_methods_s,
	.get_size = csr_interface_get_size,
	.interfaceid = STARPU_CSR_INTERFACE_ID,
	.interface_size = sizeof(starpu_csr_interface_t),
	.footprint = footprint_csr_interface_crc32
};

static void register_csr_handle(starpu_data_handle handle, uint32_t home_node, void *interface)
{
	starpu_csr_interface_t *csr_interface = interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		starpu_csr_interface_t *local_interface =
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node) {
			local_interface->nzval = csr_interface->nzval;
			local_interface->colind = csr_interface->colind;
			local_interface->rowptr = csr_interface->rowptr;
		}
		else {
			local_interface->nzval = 0;
			local_interface->colind = NULL;
			local_interface->rowptr = NULL;
		}

		local_interface->nnz = csr_interface->nnz;
		local_interface->nrow = csr_interface->nrow;
		local_interface->firstentry = csr_interface->firstentry;
		local_interface->elemsize = csr_interface->elemsize;

	}
}

/* declare a new data with the BLAS interface */
void starpu_csr_data_register(starpu_data_handle *handleptr, uint32_t home_node,
		uint32_t nnz, uint32_t nrow, uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, size_t elemsize)
{
	starpu_csr_interface_t interface = {
		.nnz = nnz,
		.nrow = nrow,
		.nzval = nzval,
		.colind = colind,
		.rowptr = rowptr,
		.firstentry = firstentry,
		.elemsize = elemsize
	};

	_starpu_register_data_handle(handleptr, home_node, &interface, &interface_csr_ops);
}

static uint32_t footprint_csr_interface_crc32(starpu_data_handle handle)
{
	return _starpu_crc32_be(starpu_csr_get_nnz(handle), 0);
}

/* offer an access to the data parameters */
uint32_t starpu_csr_get_nnz(starpu_data_handle handle)
{
	starpu_csr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->nnz;
}

uint32_t starpu_csr_get_nrow(starpu_data_handle handle)
{
	starpu_csr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->nrow;
}

uint32_t starpu_csr_get_firstentry(starpu_data_handle handle)
{
	starpu_csr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->firstentry;
}

size_t starpu_csr_get_elemsize(starpu_data_handle handle)
{
	starpu_csr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, 0);

	return interface->elemsize;
}

uintptr_t starpu_csr_get_local_nzval(starpu_data_handle handle)
{
	unsigned node;
	node = _starpu_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	starpu_csr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, node);

	return interface->nzval;
}

uint32_t *starpu_csr_get_local_colind(starpu_data_handle handle)
{
	unsigned node;
	node = _starpu_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	starpu_csr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, node);

	return interface->colind;
}

uint32_t *starpu_csr_get_local_rowptr(starpu_data_handle handle)
{
	unsigned node;
	node = _starpu_get_local_memory_node();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	starpu_csr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, node);

	return interface->rowptr;
}

static size_t csr_interface_get_size(starpu_data_handle handle)
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
static size_t allocate_csr_buffer_on_node(starpu_data_handle handle, uint32_t dst_node)
{
	uintptr_t addr_nzval;
	uint32_t *addr_colind, *addr_rowptr;
	size_t allocated_memory;

	/* we need the 3 arrays to be allocated */
	starpu_csr_interface_t *interface =
		starpu_data_get_interface_on_node(handle, dst_node);

	uint32_t nnz = interface->nnz;
	uint32_t nrow = interface->nrow;
	size_t elemsize = interface->elemsize;

	starpu_node_kind kind = _starpu_get_node_kind(dst_node);

	switch(kind) {
		case STARPU_RAM:
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
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_RAM:
			cudaMalloc((void **)&addr_nzval, nnz*elemsize);
			if (!addr_nzval)
				goto fail_nzval;

			cudaMalloc((void **)&addr_colind, nnz*sizeof(uint32_t));
			if (!addr_colind)
				goto fail_colind;

			cudaMalloc((void **)&addr_rowptr, (nrow+1)*sizeof(uint32_t));
			if (!addr_rowptr)
				goto fail_rowptr;

			break;
#endif
#ifdef STARPU_USE_OPENCL
	        case STARPU_OPENCL_RAM:
			{
                                int ret;
                                void *ptr;

                                ret = _starpu_opencl_allocate_memory(&ptr, nnz*elemsize, CL_MEM_READ_WRITE);
                                addr_nzval = (uintptr_t)ptr;
				if (ret) goto fail_nzval;

                                ret = _starpu_opencl_allocate_memory(&ptr, nnz*sizeof(uint32_t), CL_MEM_READ_WRITE);
                                addr_colind = ptr;
				if (ret) goto fail_colind;

                                ret = _starpu_opencl_allocate_memory(&ptr, (nrow+1)*sizeof(uint32_t), CL_MEM_READ_WRITE);
                                addr_rowptr = ptr;
				if (ret) goto fail_rowptr;

				break;
			}
#endif
		default:
			assert(0);
	}

	/* allocation succeeded */
	allocated_memory = 
		nnz*elemsize + nnz*sizeof(uint32_t) + (nrow+1)*sizeof(uint32_t);

	/* update the data properly in consequence */
	interface->nzval = addr_nzval;
	interface->colind = addr_colind;
	interface->rowptr = addr_rowptr;
	
	return allocated_memory;

fail_rowptr:
	switch(kind) {
		case STARPU_RAM:
			free((void *)addr_colind);
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_RAM:
			cudaFree((void*)addr_colind);
			break;
#endif
#ifdef STARPU_USE_OPENCL
		case STARPU_OPENCL_RAM:
			clReleaseMemObject((void*)addr_colind);
			break;
#endif
		default:
			assert(0);
	}

fail_colind:
	switch(kind) {
		case STARPU_RAM:
			free((void *)addr_nzval);
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_RAM:
			cudaFree((void*)addr_nzval);
			break;
#endif
#ifdef STARPU_USE_OPENCL
		case STARPU_OPENCL_RAM:
			clReleaseMemObject((void*)addr_nzval);
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

static void liberate_csr_buffer_on_node(void *interface, uint32_t node)
{
	starpu_csr_interface_t *csr_interface = interface;	

	starpu_node_kind kind = _starpu_get_node_kind(node);
	switch(kind) {
		case STARPU_RAM:
			free((void*)csr_interface->nzval);
			free((void*)csr_interface->colind);
			free((void*)csr_interface->rowptr);
			break;
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_RAM:
			cudaFree((void*)csr_interface->nzval);
			cudaFree((void*)csr_interface->colind);
			cudaFree((void*)csr_interface->rowptr);
			break;
#endif
#ifdef STARPU_USE_OPENCL
		case STARPU_OPENCL_RAM:
			clReleaseMemObject((void*)csr_interface->nzval);
			clReleaseMemObject((void*)csr_interface->colind);
			clReleaseMemObject((void*)csr_interface->rowptr);
			break;
#endif
		default:
			assert(0);
	}
}

#ifdef STARPU_USE_CUDA
static int copy_cuda_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node)
{
	starpu_csr_interface_t *src_csr;
	starpu_csr_interface_t *dst_csr;

	src_csr = starpu_data_get_interface_on_node(handle, src_node);
	dst_csr = starpu_data_get_interface_on_node(handle, dst_node);

	uint32_t nnz = src_csr->nnz;
	uint32_t nrow = src_csr->nrow;
	size_t elemsize = src_csr->elemsize;

	cudaError_t cures;

	cures = cudaMemcpy((char *)dst_csr->nzval, (char *)src_csr->nzval, nnz*elemsize, cudaMemcpyDeviceToHost);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	cures = cudaMemcpy((char *)dst_csr->colind, (char *)src_csr->colind, nnz*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	cures = cudaMemcpy((char *)dst_csr->rowptr, (char *)src_csr->rowptr, (nrow+1)*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	cudaThreadSynchronize();

	STARPU_TRACE_DATA_COPY(src_node, dst_node, nnz*elemsize + (nnz+nrow+1)*sizeof(uint32_t));

	return 0;
}

static int copy_ram_to_cuda(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node)
{
	starpu_csr_interface_t *src_csr;
	starpu_csr_interface_t *dst_csr;

	src_csr = starpu_data_get_interface_on_node(handle, src_node);
	dst_csr = starpu_data_get_interface_on_node(handle, dst_node);

	uint32_t nnz = src_csr->nnz;
	uint32_t nrow = src_csr->nrow;
	size_t elemsize = src_csr->elemsize;

	cudaError_t cures;

	cures = cudaMemcpy((char *)dst_csr->nzval, (char *)src_csr->nzval, nnz*elemsize, cudaMemcpyHostToDevice);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	cures = cudaMemcpy((char *)dst_csr->colind, (char *)src_csr->colind, nnz*sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	cures = cudaMemcpy((char *)dst_csr->rowptr, (char *)src_csr->rowptr, (nrow+1)*sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	cudaThreadSynchronize();

	STARPU_TRACE_DATA_COPY(src_node, dst_node, nnz*elemsize + (nnz+nrow+1)*sizeof(uint32_t));

	return 0;
}
#endif // STARPU_USE_CUDA

#ifdef STARPU_USE_OPENCL
static int copy_opencl_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node)
{
	starpu_csr_interface_t *src_csr;
	starpu_csr_interface_t *dst_csr;

	src_csr = starpu_data_get_interface_on_node(handle, src_node);
	dst_csr = starpu_data_get_interface_on_node(handle, dst_node);

	uint32_t nnz = src_csr->nnz;
	uint32_t nrow = src_csr->nrow;
	size_t elemsize = src_csr->elemsize;

        int err;

        err = _starpu_opencl_copy_from_opencl((cl_mem)src_csr->nzval, (void *)dst_csr->nzval, nnz*elemsize, 0, NULL);
	if (STARPU_UNLIKELY(err))
                STARPU_OPENCL_REPORT_ERROR(err);

	err = _starpu_opencl_copy_from_opencl((cl_mem)src_csr->colind, (void *)dst_csr->colind, nnz*sizeof(uint32_t), 0, NULL);
        if (STARPU_UNLIKELY(err))
                STARPU_OPENCL_REPORT_ERROR(err);

        err = _starpu_opencl_copy_from_opencl((cl_mem)src_csr->rowptr, (void *)dst_csr->rowptr, (nrow+1)*sizeof(uint32_t), 0, NULL);
	if (STARPU_UNLIKELY(err))
                STARPU_OPENCL_REPORT_ERROR(err);

	STARPU_TRACE_DATA_COPY(src_node, dst_node, nnz*elemsize + (nnz+nrow+1)*sizeof(uint32_t));

	return 0;
}

static int copy_ram_to_opencl(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node)
{
	starpu_csr_interface_t *src_csr;
	starpu_csr_interface_t *dst_csr;

	src_csr = starpu_data_get_interface_on_node(handle, src_node);
	dst_csr = starpu_data_get_interface_on_node(handle, dst_node);

	uint32_t nnz = src_csr->nnz;
	uint32_t nrow = src_csr->nrow;
	size_t elemsize = src_csr->elemsize;

        int err;

        err = _starpu_opencl_copy_to_opencl((void *)src_csr->nzval, (cl_mem)dst_csr->nzval, nnz*elemsize, 0, NULL);
	if (STARPU_UNLIKELY(err))
                STARPU_OPENCL_REPORT_ERROR(err);

	err = _starpu_opencl_copy_to_opencl((void *)src_csr->colind, (cl_mem)dst_csr->colind, nnz*sizeof(uint32_t), 0, NULL);
        if (STARPU_UNLIKELY(err))
                STARPU_OPENCL_REPORT_ERROR(err);

        err = _starpu_opencl_copy_to_opencl((void *)src_csr->rowptr, (cl_mem)dst_csr->rowptr, (nrow+1)*sizeof(uint32_t), 0, NULL);
	if (STARPU_UNLIKELY(err))
                STARPU_OPENCL_REPORT_ERROR(err);

	STARPU_TRACE_DATA_COPY(src_node, dst_node, nnz*elemsize + (nnz+nrow+1)*sizeof(uint32_t));

	return 0;
}
#endif // STARPU_USE_OPENCL

/* as not all platform easily have a BLAS lib installed ... */
static int dummy_copy_ram_to_ram(starpu_data_handle handle, uint32_t src_node, uint32_t dst_node)
{

	starpu_csr_interface_t *src_csr;
	starpu_csr_interface_t *dst_csr;

	src_csr = starpu_data_get_interface_on_node(handle, src_node);
	dst_csr = starpu_data_get_interface_on_node(handle, dst_node);

	uint32_t nnz = src_csr->nnz;
	uint32_t nrow = src_csr->nrow;
	size_t elemsize = src_csr->elemsize;

	memcpy((void *)dst_csr->nzval, (void *)src_csr->nzval, nnz*elemsize);

	memcpy((void *)dst_csr->colind, (void *)src_csr->colind, nnz*sizeof(uint32_t));

	memcpy((void *)dst_csr->rowptr, (void *)src_csr->rowptr, (nrow+1)*sizeof(uint32_t));

	STARPU_TRACE_DATA_COPY(src_node, dst_node, nnz*elemsize + (nnz+nrow+1)*sizeof(uint32_t));

	return 0;
}
