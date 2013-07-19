/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
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

#include <starpu_hash.h>

#include <starpu_cuda.h>
#include <starpu_opencl.h>
#include <drivers/opencl/driver_opencl.h>
#include <drivers/scc/driver_scc_source.h>
#include <drivers/mic/driver_mic_source.h>

#ifdef STARPU_USE_CUDA
static int copy_ram_to_cuda(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED);
static int copy_cuda_to_ram(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED);
static int copy_ram_to_cuda_async(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, cudaStream_t stream);
static int copy_cuda_to_ram_async(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, cudaStream_t stream);
static int copy_cuda_to_cuda(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED);
#endif
#ifdef STARPU_USE_OPENCL
static int copy_ram_to_opencl(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED);
static int copy_opencl_to_ram(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED);
static int copy_opencl_to_opencl(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED);
static int copy_ram_to_opencl_async(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, cl_event *event);
static int copy_opencl_to_ram_async(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, cl_event *event);
static int copy_opencl_to_opencl_async(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, cl_event *event);
#endif
static int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data);

static const struct starpu_data_copy_methods block_copy_data_methods_s =
{
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
	.opencl_to_opencl = copy_opencl_to_opencl,
        .ram_to_opencl_async = copy_ram_to_opencl_async,
	.opencl_to_ram_async = copy_opencl_to_ram_async,
	.opencl_to_opencl_async = copy_opencl_to_opencl_async,
#endif
	.any_to_any = copy_any_to_any,
};


static void register_block_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface);
static void *block_handle_to_pointer(starpu_data_handle_t data_handle, unsigned node);
static starpu_ssize_t allocate_block_buffer_on_node(void *data_interface_, unsigned dst_node);
static void free_block_buffer_on_node(void *data_interface, unsigned node);
static size_t block_interface_get_size(starpu_data_handle_t handle);
static uint32_t footprint_block_interface_crc32(starpu_data_handle_t handle);
static int block_compare(void *data_interface_a, void *data_interface_b);
static void display_block_interface(starpu_data_handle_t handle, FILE *f);
static int pack_block_handle(starpu_data_handle_t handle, unsigned node, void **ptr, ssize_t *count);
static int unpack_block_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);

struct starpu_data_interface_ops starpu_interface_block_ops =
{
	.register_data_handle = register_block_handle,
	.allocate_data_on_node = allocate_block_buffer_on_node,
	.handle_to_pointer = block_handle_to_pointer,
	.free_data_on_node = free_block_buffer_on_node,
	.copy_methods = &block_copy_data_methods_s,
	.get_size = block_interface_get_size,
	.footprint = footprint_block_interface_crc32,
	.compare = block_compare,
	.interfaceid = STARPU_BLOCK_INTERFACE_ID,
	.interface_size = sizeof(struct starpu_block_interface),
	.display = display_block_interface,
	.pack_data = pack_block_handle,
	.unpack_data = unpack_block_handle
};

static void *block_handle_to_pointer(starpu_data_handle_t handle, unsigned node)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_block_interface *block_interface = (struct starpu_block_interface *)
		starpu_data_get_interface_on_node(handle, node);

	return (void*) block_interface->ptr;
}

static void register_block_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface)
{
	struct starpu_block_interface *block_interface = (struct starpu_block_interface *) data_interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_block_interface *local_interface = (struct starpu_block_interface *)
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
		{
			local_interface->ptr = block_interface->ptr;
                        local_interface->dev_handle = block_interface->dev_handle;
                        local_interface->offset = block_interface->offset;
			local_interface->ldy  = block_interface->ldy;
			local_interface->ldz  = block_interface->ldz;
		}
		else
		{
			local_interface->ptr = 0;
                        local_interface->dev_handle = 0;
                        local_interface->offset = 0;
			local_interface->ldy  = 0;
			local_interface->ldz  = 0;
		}

		local_interface->id = block_interface->id;
		local_interface->nx = block_interface->nx;
		local_interface->ny = block_interface->ny;
		local_interface->nz = block_interface->nz;
		local_interface->elemsize = block_interface->elemsize;
	}
}

/* declare a new data with the BLAS interface */
void starpu_block_data_register(starpu_data_handle_t *handleptr, unsigned home_node,
			uintptr_t ptr, uint32_t ldy, uint32_t ldz, uint32_t nx,
			uint32_t ny, uint32_t nz, size_t elemsize)
{
	struct starpu_block_interface block_interface =
	{
		.id = STARPU_BLOCK_INTERFACE_ID,
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

#ifdef STARPU_USE_SCC
	_starpu_scc_set_offset_in_shared_memory((void*)block_interface.ptr,
			(void**)&(block_interface.dev_handle), &(block_interface.offset));
#endif

	starpu_data_register(handleptr, home_node, &block_interface, &starpu_interface_block_ops);
}

static uint32_t footprint_block_interface_crc32(starpu_data_handle_t handle)
{
	uint32_t hash;

	hash = starpu_hash_crc32c_be(starpu_block_get_nx(handle), 0);
	hash = starpu_hash_crc32c_be(starpu_block_get_ny(handle), hash);
	hash = starpu_hash_crc32c_be(starpu_block_get_nz(handle), hash);

	return hash;
}

static int block_compare(void *data_interface_a, void *data_interface_b)
{
	struct starpu_block_interface *block_a = (struct starpu_block_interface *) data_interface_a;
	struct starpu_block_interface *block_b = (struct starpu_block_interface *) data_interface_b;

	/* Two matricess are considered compatible if they have the same size */
	return ((block_a->nx == block_b->nx)
			&& (block_a->ny == block_b->ny)
			&& (block_a->nz == block_b->nz)
			&& (block_a->elemsize == block_b->elemsize));
}

static void display_block_interface(starpu_data_handle_t handle, FILE *f)
{
	struct starpu_block_interface *block_interface;

	block_interface = (struct starpu_block_interface *) starpu_data_get_interface_on_node(handle, 0);

	fprintf(f, "%u\t%u\t%u\t", block_interface->nx, block_interface->ny, block_interface->nz);
}

static int pack_block_handle(starpu_data_handle_t handle, unsigned node, void **ptr, ssize_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_block_interface *block_interface = (struct starpu_block_interface *)
		starpu_data_get_interface_on_node(handle, node);

	*count = block_interface->nx*block_interface->ny*block_interface->nz*block_interface->elemsize;

	if (ptr != NULL)
	{
		uint32_t z, y;
		void *block = (void *)block_interface->ptr;

		*ptr = malloc(*count);

		void *cur = *ptr;
		for(z=0 ; z<block_interface->nz ; z++)
		{
			void *block_z = block;
			for(y=0 ; y<block_interface->ny ; y++)
			{
				memcpy(cur, block, block_interface->nx*block_interface->elemsize);
				cur += block_interface->nx*block_interface->elemsize;
				block += block_interface->ldy * block_interface->elemsize;
			}
			block = block_z + block_interface->ldz * block_interface->elemsize;
		}
	}

	return 0;
}

static int unpack_block_handle(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_block_interface *block_interface = (struct starpu_block_interface *)
		starpu_data_get_interface_on_node(handle, node);

	STARPU_ASSERT(count == block_interface->elemsize * block_interface->nx * block_interface->ny * block_interface->nz);

	uint32_t z, y;
	void *cur = ptr;
	void *block = (void *)block_interface->ptr;
	for(z=0 ; z<block_interface->nz ; z++)
	{
		void *block_z = block;
		for(y=0 ; y<block_interface->ny ; y++)
		{
			memcpy(block, cur, block_interface->nx*block_interface->elemsize);
			cur += block_interface->nx*block_interface->elemsize;
			block += block_interface->ldy * block_interface->elemsize;
		}
		block = block_z + block_interface->ldz * block_interface->elemsize;
	}

	return 0;
}


static size_t block_interface_get_size(starpu_data_handle_t handle)
{
	size_t size;
	struct starpu_block_interface *block_interface;

	block_interface = (struct starpu_block_interface *) starpu_data_get_interface_on_node(handle, 0);

	size = block_interface->nx*block_interface->ny*block_interface->nz*block_interface->elemsize;

	return size;
}

/* offer an access to the data parameters */
uint32_t starpu_block_get_nx(starpu_data_handle_t handle)
{
	struct starpu_block_interface *block_interface = (struct starpu_block_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	return block_interface->nx;
}

uint32_t starpu_block_get_ny(starpu_data_handle_t handle)
{
	struct starpu_block_interface *block_interface = (struct starpu_block_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	return block_interface->ny;
}

uint32_t starpu_block_get_nz(starpu_data_handle_t handle)
{
	struct starpu_block_interface *block_interface = (struct starpu_block_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	return block_interface->nz;
}

uint32_t starpu_block_get_local_ldy(starpu_data_handle_t handle)
{
	unsigned node;
	node = _starpu_memory_node_get_local_key();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_block_interface *block_interface = (struct starpu_block_interface *)
		starpu_data_get_interface_on_node(handle, node);

	return block_interface->ldy;
}

uint32_t starpu_block_get_local_ldz(starpu_data_handle_t handle)
{
	unsigned node;
	node = _starpu_memory_node_get_local_key();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_block_interface *block_interface = (struct starpu_block_interface *)
		starpu_data_get_interface_on_node(handle, node);

	return block_interface->ldz;
}

uintptr_t starpu_block_get_local_ptr(starpu_data_handle_t handle)
{
	unsigned node;
	node = _starpu_memory_node_get_local_key();

	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_block_interface *block_interface = (struct starpu_block_interface *)
		starpu_data_get_interface_on_node(handle, node);

	return block_interface->ptr;
}

size_t starpu_block_get_elemsize(starpu_data_handle_t handle)
{
	struct starpu_block_interface *block_interface = (struct starpu_block_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	return block_interface->elemsize;
}


/* memory allocation/deallocation primitives for the BLOCK interface */

/* returns the size of the allocated area */
static starpu_ssize_t allocate_block_buffer_on_node(void *data_interface_, unsigned dst_node)
{
	uintptr_t addr = 0, handle;

	struct starpu_block_interface *dst_block = (struct starpu_block_interface *) data_interface_;

	uint32_t nx = dst_block->nx;
	uint32_t ny = dst_block->ny;
	uint32_t nz = dst_block->nz;
	size_t elemsize = dst_block->elemsize;

	starpu_ssize_t allocated_memory;

	handle = starpu_malloc_on_node(dst_node, nx*ny*nz*elemsize);

	if (!handle)
		return -ENOMEM;

	if (starpu_node_get_kind(dst_node) != STARPU_OPENCL_RAM)
		addr = handle;

	allocated_memory = nx*ny*nz*elemsize;

	/* update the data properly in consequence */
	dst_block->ptr = addr;
	dst_block->dev_handle = handle;
	dst_block->offset = 0;
	dst_block->ldy = nx;
	dst_block->ldz = nx*ny;

	return allocated_memory;
}

static void free_block_buffer_on_node(void *data_interface, unsigned node)
{
	struct starpu_block_interface *block_interface = (struct starpu_block_interface *) data_interface;
	uint32_t nx = block_interface->nx;
	uint32_t ny = block_interface->ny;
	uint32_t nz = block_interface->nz;
	size_t elemsize = block_interface->elemsize;

	starpu_free_on_node(node, block_interface->dev_handle, nx*ny*nz*elemsize);
}

#ifdef STARPU_USE_CUDA
static int copy_cuda_common(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, enum cudaMemcpyKind kind)
{
	struct starpu_block_interface *src_block = src_interface;
	struct starpu_block_interface *dst_block = dst_interface;

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
			starpu_cuda_copy_async_sync((void *)src_block->ptr, src_node, (void *)dst_block->ptr, dst_node, nx*ny*nz*elemsize, NULL, kind);
                }
		else
		{
			/* Are all plans contiguous */
                        cures = cudaMemcpy2D((char *)dst_block->ptr, dst_block->ldz*elemsize,
                                             (char *)src_block->ptr, src_block->ldz*elemsize,
                                             nx*ny*elemsize, nz, kind);
                        if (STARPU_UNLIKELY(cures))
                                STARPU_CUDA_REPORT_ERROR(cures);
                }
	}
	else
	{
		/* Default case: we transfer all blocks one by one: nz transfers */
		unsigned layer;
		for (layer = 0; layer < src_block->nz; layer++)
		{
			uint8_t *src_ptr = ((uint8_t *)src_block->ptr) + layer*src_block->ldz*src_block->elemsize;
			uint8_t *dst_ptr = ((uint8_t *)dst_block->ptr) + layer*dst_block->ldz*dst_block->elemsize;

			cures = cudaMemcpy2D((char *)dst_ptr, dst_block->ldy*elemsize,
                                             (char *)src_ptr, src_block->ldy*elemsize,
                                             nx*elemsize, ny, kind);

			if (STARPU_UNLIKELY(cures))
				STARPU_CUDA_REPORT_ERROR(cures);
		}
	}

	_STARPU_TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->elemsize*src_block->elemsize);

	return 0;
}

static int copy_cuda_async_common(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, cudaStream_t stream, enum cudaMemcpyKind kind)
{
	struct starpu_block_interface *src_block = src_interface;
	struct starpu_block_interface *dst_block = dst_interface;

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
			ret = starpu_cuda_copy_async_sync((void *)src_block->ptr, src_node, (void *)dst_block->ptr, dst_node, nx*ny*nz*elemsize, stream, kind);
		}
		else
		{
			/* Are all plans contiguous */
			_STARPU_TRACE_START_DRIVER_COPY_ASYNC(src_node, dst_node);
			cures = cudaMemcpy2DAsync((char *)dst_block->ptr, dst_block->ldz*elemsize,
					(char *)src_block->ptr, src_block->ldz*elemsize,
					nx*ny*elemsize, nz, kind, stream);
			_STARPU_TRACE_END_DRIVER_COPY_ASYNC(src_node, dst_node);
			if (STARPU_UNLIKELY(cures))
			{
				cures = cudaMemcpy2D((char *)dst_block->ptr, dst_block->ldz*elemsize,
						(char *)src_block->ptr, src_block->ldz*elemsize,
						nx*ny*elemsize, nz, kind);
				if (STARPU_UNLIKELY(cures))
					STARPU_CUDA_REPORT_ERROR(cures);

				ret = 0;
			}
			else
			{
				ret = -EAGAIN;
			}
		}
	}
	else
	{
		/* Default case: we transfer all blocks one by one: nz 2D transfers */
		unsigned layer;
		for (layer = 0; layer < src_block->nz; layer++)
		{
			uint8_t *src_ptr = ((uint8_t *)src_block->ptr) + layer*src_block->ldz*src_block->elemsize;
			uint8_t *dst_ptr = ((uint8_t *)dst_block->ptr) + layer*dst_block->ldz*dst_block->elemsize;

			_STARPU_TRACE_START_DRIVER_COPY_ASYNC(src_node, dst_node);
			cures = cudaMemcpy2DAsync((char *)dst_ptr, dst_block->ldy*elemsize,
                                                  (char *)src_ptr, src_block->ldy*elemsize,
                                                  nx*elemsize, ny, kind, stream);
			_STARPU_TRACE_END_DRIVER_COPY_ASYNC(src_node, dst_node);

			if (STARPU_UNLIKELY(cures))
			{
				/* I don't know how to do that "better" */
				goto no_async_default;
			}

		}

		ret = -EAGAIN;

	}

	_STARPU_TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);

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
                                     nx*elemsize, ny, kind);

		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);
	}

	_STARPU_TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);
	return 0;
	}
}

static int copy_cuda_to_ram(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node)
{
	return copy_cuda_common(src_interface, src_node, dst_interface, dst_node, cudaMemcpyDeviceToHost);
}

static int copy_ram_to_cuda(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED)
{
	return copy_cuda_common(src_interface, src_node, dst_interface, dst_node, cudaMemcpyHostToDevice);
}

static int copy_cuda_to_cuda(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED)
{
	return copy_cuda_common(src_interface, src_node, dst_interface, dst_node, cudaMemcpyDeviceToDevice);
}

static int copy_cuda_to_ram_async(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, cudaStream_t stream)
{
	return copy_cuda_async_common(src_interface, src_node, dst_interface, dst_node, stream, cudaMemcpyDeviceToHost);
}

static int copy_ram_to_cuda_async(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, cudaStream_t stream)
{
	return copy_cuda_async_common(src_interface, src_node, dst_interface, dst_node, stream, cudaMemcpyHostToDevice);
}
#endif // STARPU_USE_CUDA

#ifdef STARPU_USE_OPENCL
static int copy_opencl_common(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cl_event *event)
{
	struct starpu_block_interface *src_block = src_interface;
	struct starpu_block_interface *dst_block = dst_interface;
        int ret = 0;

	uint32_t nx = src_block->nx;
	uint32_t ny = src_block->ny;

	/* We may have a contiguous buffer for the entire block, or contiguous
	 * plans within the block, we can avoid many small transfers that way */
	if ((nx == src_block->ldy) && (src_block->ldy == dst_block->ldy))
	{
		/* Is that a single contiguous buffer ? */
		if (((nx*ny) == src_block->ldz) && (src_block->ldz == dst_block->ldz))
		{
			ret = starpu_opencl_copy_async_sync(src_block->dev_handle, src_block->offset, src_node,
								dst_block->dev_handle, dst_block->offset, dst_node,
							       src_block->nx*src_block->ny*src_block->nz*src_block->elemsize,
							       event);
                }
		else
		{
			/* Are all plans contiguous */
                        STARPU_ASSERT_MSG(0, "XXX non contiguous buffers are not properly supported in OpenCL yet. (TODO)");
                }
        }
	else
	{
		/* Default case: we transfer all lines one by one: ny*nz transfers */
		/* TODO: rect support */
		unsigned layer;
		for (layer = 0; layer < src_block->nz; layer++)
		{
                        unsigned j;
                        for(j=0 ; j<src_block->ny ; j++)
			{
				ret = starpu_opencl_copy_async_sync(src_block->dev_handle,
								    src_block->offset + layer*src_block->ldz*src_block->elemsize + j*src_block->ldy*src_block->elemsize,
								    src_node,
								    dst_block->dev_handle,
								    dst_block->offset + layer*dst_block->ldz*dst_block->elemsize + j*dst_block->ldy*dst_block->elemsize,
								    dst_node,
								       src_block->nx*src_block->elemsize,
								       event);
                        }
                }
        }

	_STARPU_TRACE_DATA_COPY(src_node, dst_node, src_block->nx*src_block->ny*src_block->nz*src_block->elemsize);

	return ret;
}

static int copy_ram_to_opencl_async(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cl_event *event)
{
	return copy_opencl_common(src_interface, src_node, dst_interface, dst_node, event);
}

static int copy_opencl_to_ram_async(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cl_event *event)
{
	return copy_opencl_common(src_interface, src_node, dst_interface, dst_node, event);
}

static int copy_opencl_to_opencl_async(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cl_event *event)
{
	return copy_opencl_common(src_interface, src_node, dst_interface, dst_node, event);
}

static int copy_ram_to_opencl(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED)
{
        return copy_ram_to_opencl_async(src_interface, src_node, dst_interface, dst_node, NULL);
}

static int copy_opencl_to_ram(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED)
{
        return copy_opencl_to_ram_async(src_interface, src_node, dst_interface, dst_node, NULL);
}

static int copy_opencl_to_opencl(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED)
{
	return copy_opencl_to_opencl_async(src_interface, src_node, dst_interface, dst_node, NULL);
}

#endif

static int copy_any_to_any(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data)
{
	struct starpu_block_interface *src_block = (struct starpu_block_interface *) src_interface;
	struct starpu_block_interface *dst_block = (struct starpu_block_interface *) dst_interface;
	int ret;

	uint32_t nx = dst_block->nx;
	uint32_t ny = dst_block->ny;
	uint32_t nz = dst_block->nz;
	size_t elemsize = dst_block->elemsize;

	uint32_t ldy_src = src_block->ldy;
	uint32_t ldz_src = src_block->ldz;
	uint32_t ldy_dst = dst_block->ldy;
	uint32_t ldz_dst = dst_block->ldz;

	unsigned y, z;

	if (ldy_src == nx && ldy_dst == nx && ldz_src == ny && ldz_dst == ny)
	{
		/* Optimise non-partitioned and z-partitioned case */
		if (starpu_interface_copy(src_block->dev_handle, src_block->offset, src_node,
		                          dst_block->dev_handle, dst_block->offset, dst_node,
		                          nx*ny*nz*elemsize, async_data))
				ret = -EAGAIN;
	}
	else
	for (z = 0; z < nz; z++)
	{
		if (ldy_src == nx && ldy_dst == nx)
		{
			/* Optimise y-partitioned case */
			uint32_t src_offset = z*ldz_src*elemsize;
			uint32_t dst_offset = z*ldz_dst*elemsize;

			if (starpu_interface_copy(src_block->dev_handle, src_block->offset + src_offset, src_node,
			                          dst_block->dev_handle, dst_block->offset + dst_offset, dst_node,
			                          nx*ny*elemsize, async_data))
				ret = -EAGAIN;
		}
		else
		for (y = 0; y < ny; y++)
		{
			/* Eerf, x-partitioned case */
			uint32_t src_offset = (y*ldy_src + z*ldz_src)*elemsize;
			uint32_t dst_offset = (y*ldy_dst + z*ldz_dst)*elemsize;

			if (starpu_interface_copy(src_block->dev_handle, src_block->offset + src_offset, src_node,
			                          dst_block->dev_handle, dst_block->offset + dst_offset, dst_node,
			                          nx*elemsize, async_data))
				ret = -EAGAIN;
		}
	}

	_STARPU_TRACE_DATA_COPY(src_node, dst_node, nx*ny*nz*elemsize);

	return ret;
}
