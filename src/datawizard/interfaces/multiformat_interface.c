/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static int copy_ram_to_ram(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node);
#ifdef STARPU_USE_CUDA
static int copy_ram_to_cuda(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node);
static int copy_cuda_to_ram(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node);
static int copy_ram_to_cuda_async(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node, cudaStream_t stream);
static int copy_cuda_to_ram_async(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node, cudaStream_t stream);
static int copy_cuda_to_cuda(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED);
static int copy_cuda_to_cuda_async(void *src_interface, unsigned src_node,					void *dst_interface, unsigned dst_node, cudaStream_t stream);
#endif
#ifdef STARPU_USE_OPENCL
static int copy_ram_to_opencl(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node);
static int copy_opencl_to_ram(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node);
static int copy_opencl_to_opencl(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node);
static int copy_ram_to_opencl_async(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node, cl_event *event);
static int copy_opencl_to_ram_async(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node, cl_event *event);
#endif
#ifdef STARPU_USE_MIC
static int copy_ram_to_mic(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
static int copy_mic_to_ram(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
static int copy_ram_to_mic_async(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
static int copy_mic_to_ram_async(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
#endif

static const struct starpu_data_copy_methods multiformat_copy_data_methods_s =
{
	.ram_to_ram = copy_ram_to_ram,
#ifdef STARPU_USE_CUDA
	.ram_to_cuda = copy_ram_to_cuda,
	.cuda_to_ram = copy_cuda_to_ram,
	.ram_to_cuda_async = copy_ram_to_cuda_async,
	.cuda_to_ram_async = copy_cuda_to_ram_async,
	.cuda_to_cuda = copy_cuda_to_cuda,
	.cuda_to_cuda_async = copy_cuda_to_cuda_async,
#else
#ifdef STARPU_SIMGRID
	/* Enable GPU-GPU transfers in simgrid */
	.cuda_to_cuda_async = (void *)1,
#endif
#endif
#ifdef STARPU_USE_OPENCL
	.ram_to_opencl = copy_ram_to_opencl,
	.opencl_to_ram = copy_opencl_to_ram,
	.opencl_to_opencl = copy_opencl_to_opencl,
        .ram_to_opencl_async = copy_ram_to_opencl_async,
	.opencl_to_ram_async = copy_opencl_to_ram_async,
#endif
#ifdef STARPU_USE_MIC
	.ram_to_mic = copy_ram_to_mic,
	.mic_to_ram = copy_mic_to_ram,
	.ram_to_mic_async = copy_ram_to_mic_async,
	.mic_to_ram_async = copy_mic_to_ram_async,
#endif
};

static void register_multiformat_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface);
static starpu_ssize_t allocate_multiformat_buffer_on_node(void *data_interface_, unsigned dst_node);
static void *multiformat_to_pointer(void *data_interface, unsigned node);
static int multiformat_pointer_is_inside(void *data_interface, unsigned node, void *ptr);
static void free_multiformat_buffer_on_node(void *data_interface, unsigned node);
static size_t multiformat_interface_get_size(starpu_data_handle_t handle);
static uint32_t footprint_multiformat_interface_crc32(starpu_data_handle_t handle);
static int multiformat_compare(void *data_interface_a, void *data_interface_b);
static void display_multiformat_interface(starpu_data_handle_t handle, FILE *f);
static uint32_t starpu_multiformat_get_nx(starpu_data_handle_t handle);

static struct starpu_multiformat_data_interface_ops*
get_mf_ops(void *data_interface)
{
	struct starpu_multiformat_interface *mf;
	mf = (struct starpu_multiformat_interface *) data_interface;

	return mf->ops;
}

struct starpu_data_interface_ops starpu_interface_multiformat_ops =
{
	.register_data_handle  = register_multiformat_handle,
	.allocate_data_on_node = allocate_multiformat_buffer_on_node,
	.to_pointer            = multiformat_to_pointer,
	.pointer_is_inside     = multiformat_pointer_is_inside,
	.free_data_on_node     = free_multiformat_buffer_on_node,
	.copy_methods          = &multiformat_copy_data_methods_s,
	.get_size              = multiformat_interface_get_size,
	.footprint             = footprint_multiformat_interface_crc32,
	.compare               = multiformat_compare,
	.interfaceid           = STARPU_MULTIFORMAT_INTERFACE_ID,
	.interface_size        = sizeof(struct starpu_multiformat_interface),
	.display               = display_multiformat_interface,
	.is_multiformat        = 1,
	.get_mf_ops            = get_mf_ops
};

static void *multiformat_to_pointer(void *data_interface, unsigned node)
{
	struct starpu_multiformat_interface *multiformat_interface = data_interface;

	switch(starpu_node_get_kind(node))
	{
		case STARPU_CPU_RAM:
			return multiformat_interface->cpu_ptr;
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_RAM:
			return multiformat_interface->cuda_ptr;
#endif
#ifdef STARPU_USE_OPENCL
		case STARPU_OPENCL_RAM:
			return multiformat_interface->opencl_ptr;
#endif
#ifdef STARPU_USE_MIC
		case STARPU_MIC_RAM:
			return multiformat_interface->mic_ptr;
#endif
		default:
			STARPU_ABORT();
	}
	return NULL;
}

static int multiformat_pointer_is_inside(void *data_interface, unsigned node, void *ptr)
{
	struct starpu_multiformat_interface *multiformat_interface = data_interface;

	switch(starpu_node_get_kind(node))
	{
		case STARPU_CPU_RAM:
			return (char*) ptr >= (char*) multiformat_interface->cpu_ptr &&
				(char*) ptr < (char*) multiformat_interface->cpu_ptr + multiformat_interface->nx * multiformat_interface->ops->cpu_elemsize;
#ifdef STARPU_USE_CUDA
		case STARPU_CUDA_RAM:
			return (char*) ptr >= (char*) multiformat_interface->cuda_ptr &&
				(char*) ptr < (char*) multiformat_interface->cuda_ptr + multiformat_interface->nx * multiformat_interface->ops->cuda_elemsize;
#endif
#ifdef STARPU_USE_OPENCL
		case STARPU_OPENCL_RAM:
			return (char*) ptr >= (char*) multiformat_interface->opencl_ptr &&
				(char*) ptr < (char*) multiformat_interface->opencl_ptr + multiformat_interface->nx * multiformat_interface->ops->opencl_elemsize;
#endif
#ifdef STARPU_USE_MIC
		case STARPU_MIC_RAM:
			return (char*) ptr >= (char*) multiformat_interface->mic_ptr &&
				(char*) ptr < (char*) multiformat_interface->mic_ptr + multiformat_interface->nx * multiformat_interface->ops->mic_elemsize;
#endif
		default:
			STARPU_ABORT();
	}
	return -1;
}

static void register_multiformat_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface)
{
	struct starpu_multiformat_interface *multiformat_interface;
	multiformat_interface = (struct starpu_multiformat_interface *) data_interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_multiformat_interface *local_interface =
			(struct starpu_multiformat_interface *) starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
		{
			local_interface->cpu_ptr    = multiformat_interface->cpu_ptr;
#ifdef STARPU_USE_CUDA
			local_interface->cuda_ptr   = multiformat_interface->cuda_ptr;
#endif
#ifdef STARPU_USE_OPENCL
			local_interface->opencl_ptr = multiformat_interface->opencl_ptr;
#endif
#ifdef STARPU_USE_MIC
			local_interface->mic_ptr    = multiformat_interface->mic_ptr;
#endif
		}
		else
		{
			local_interface->cpu_ptr    = NULL;
#ifdef STARPU_USE_CUDA
			local_interface->cuda_ptr   = NULL;
#endif
#ifdef STARPU_USE_OPENCL
			local_interface->opencl_ptr = NULL;
#endif
#ifdef STARPU_USE_MIC
			local_interface->mic_ptr    = NULL;
#endif
		}
		local_interface->id = multiformat_interface->id;
		local_interface->nx = multiformat_interface->nx;
		local_interface->ops = multiformat_interface->ops;
	}
}

void starpu_multiformat_data_register(starpu_data_handle_t *handleptr,
				      int home_node,
				      void *ptr,
				      uint32_t nobjects,
				      struct starpu_multiformat_data_interface_ops *format_ops)
{
	struct starpu_multiformat_interface multiformat =
	{
		.id         = STARPU_MULTIFORMAT_INTERFACE_ID,
		.cpu_ptr    = ptr,
		.cuda_ptr   = NULL,
		.opencl_ptr = NULL,
		.mic_ptr    = NULL,
		.nx         = nobjects,
		.ops        = format_ops
	};

	starpu_data_register(handleptr, home_node, &multiformat, &starpu_interface_multiformat_ops);
}

static uint32_t footprint_multiformat_interface_crc32(starpu_data_handle_t handle)
{
	return starpu_hash_crc32c_be(starpu_multiformat_get_nx(handle), 0);
}

static int multiformat_compare(void *data_interface_a, void *data_interface_b)
{
	struct starpu_multiformat_interface *multiformat_a = (struct starpu_multiformat_interface *) data_interface_a;
	struct starpu_multiformat_interface *multiformat_b = (struct starpu_multiformat_interface *) data_interface_b;

	return (multiformat_a->nx == multiformat_b->nx)
		&& (multiformat_a->ops->cpu_elemsize == multiformat_b->ops->cpu_elemsize)
#ifdef STARPU_USE_CUDA
		&& (multiformat_a->ops->cuda_elemsize == multiformat_b->ops->cuda_elemsize)
#endif
#ifdef STARPU_USE_OPENCL
		&& (multiformat_a->ops->opencl_elemsize == multiformat_b->ops->opencl_elemsize)
#endif
#ifdef STARPU_USE_MIC
		&& (multiformat_a->ops->mic_elemsize == multiformat_b->ops->mic_elemsize)
#endif
		;
}

static void display_multiformat_interface(starpu_data_handle_t handle, FILE *f)
{
	struct starpu_multiformat_interface *multiformat_interface;
	multiformat_interface = (struct starpu_multiformat_interface *)
		starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);

	fprintf(f, "%u\t", multiformat_interface->nx);
}

/* XXX : returns CPU size */
static size_t multiformat_interface_get_size(starpu_data_handle_t handle)
{
	size_t size;
	struct starpu_multiformat_interface *multiformat_interface;
	multiformat_interface = (struct starpu_multiformat_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	size = multiformat_interface->nx * multiformat_interface->ops->cpu_elemsize;
	return size;
}

uint32_t starpu_multiformat_get_nx(starpu_data_handle_t handle)
{
	struct starpu_multiformat_interface *multiformat_interface;
	multiformat_interface = (struct starpu_multiformat_interface *) starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	return multiformat_interface->nx;
}

static starpu_ssize_t allocate_multiformat_buffer_on_node(void *data_interface_, unsigned dst_node)
{
	struct starpu_multiformat_interface *multiformat_interface;
	multiformat_interface = (struct starpu_multiformat_interface *) data_interface_;
	uintptr_t addr = 0;
	starpu_ssize_t allocated_memory = 0;
	size_t size;

	size = multiformat_interface->nx * multiformat_interface->ops->cpu_elemsize;
	allocated_memory += size;
	addr = starpu_malloc_on_node(dst_node, size);
	if (!addr)
		goto fail_cpu;
	multiformat_interface->cpu_ptr = (void *) addr;
#ifdef STARPU_USE_CUDA
	size = multiformat_interface->nx * multiformat_interface->ops->cuda_elemsize;
	allocated_memory += size;
	addr = starpu_malloc_on_node(dst_node, size);
	if (!addr)
		goto fail_cuda;
	multiformat_interface->cuda_ptr = (void *) addr;
#endif
#ifdef STARPU_USE_OPENCL
	size = multiformat_interface->nx * multiformat_interface->ops->opencl_elemsize;
	allocated_memory += size;
	addr = starpu_malloc_on_node(dst_node, size);
	if (!addr)
		goto fail_opencl;
	multiformat_interface->opencl_ptr = (void *) addr;
#endif
#ifdef STARPU_USE_MIC
	size = multiformat_interface->nx * multiformat_interface->ops->mic_elemsize;
	allocated_memory += size;
	addr = starpu_malloc_on_node(dst_node, size);
	if (!addr)
		goto fail_mic;
	multiformat_interface->mic_ptr = (void *) addr;
#endif

	return allocated_memory;

#ifdef STARPU_USE_MIC
fail_mic:
#endif
#ifdef STARPU_USE_OPENCL
	starpu_free_on_node(dst_node, (uintptr_t) multiformat_interface->opencl_ptr, multiformat_interface->nx * multiformat_interface->ops->opencl_elemsize);
fail_opencl:
#endif
#ifdef STARPU_USE_CUDA
	starpu_free_on_node(dst_node, (uintptr_t) multiformat_interface->cuda_ptr, multiformat_interface->nx * multiformat_interface->ops->cuda_elemsize);
fail_cuda:
#endif
	starpu_free_on_node(dst_node, (uintptr_t) multiformat_interface->cpu_ptr, multiformat_interface->nx * multiformat_interface->ops->cpu_elemsize);
fail_cpu:
	return -ENOMEM;
}

static void free_multiformat_buffer_on_node(void *data_interface, unsigned node)
{
	struct starpu_multiformat_interface *multiformat_interface;
	multiformat_interface = (struct starpu_multiformat_interface *) data_interface;

	starpu_free_on_node(node, (uintptr_t) multiformat_interface->cpu_ptr,
				   multiformat_interface->nx * multiformat_interface->ops->cpu_elemsize);
	multiformat_interface->cpu_ptr = NULL;
#ifdef STARPU_USE_CUDA
	starpu_free_on_node(node, (uintptr_t) multiformat_interface->cuda_ptr,
				   multiformat_interface->nx * multiformat_interface->ops->cuda_elemsize);
	multiformat_interface->cuda_ptr = NULL;
#endif
#ifdef STARPU_USE_OPENCL
	starpu_free_on_node(node, (uintptr_t) multiformat_interface->opencl_ptr,
				   multiformat_interface->nx * multiformat_interface->ops->opencl_elemsize);
	multiformat_interface->opencl_ptr = NULL;
#endif
#ifdef STARPU_USE_MIC
	starpu_free_on_node(node, (uintptr_t) multiformat_interface->mic_ptr,
				   multiformat_interface->nx * multiformat_interface->ops->mic_elemsize);
	multiformat_interface->mic_ptr = NULL;
#endif
}




/*
 * Copy methods
 */
static int copy_ram_to_ram(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED,
			   void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_multiformat_interface *src_multiformat;
	struct starpu_multiformat_interface *dst_multiformat;

	src_multiformat = (struct starpu_multiformat_interface *) src_interface;
	dst_multiformat = (struct starpu_multiformat_interface *) dst_interface;

	STARPU_ASSERT(src_multiformat != NULL);
	STARPU_ASSERT(dst_multiformat != NULL);
	STARPU_ASSERT(dst_multiformat->ops != NULL);

	size_t size = dst_multiformat->nx * dst_multiformat->ops->cpu_elemsize;
	memcpy(dst_multiformat->cpu_ptr, src_multiformat->cpu_ptr, size);

	return 0;
}

#ifdef STARPU_USE_CUDA
static int copy_cuda_common(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED,
			    void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED,
			    enum cudaMemcpyKind kind)
{
	struct starpu_multiformat_interface *src_multiformat;
	struct starpu_multiformat_interface *dst_multiformat;

	src_multiformat = (struct starpu_multiformat_interface *) src_interface;
	dst_multiformat = (struct starpu_multiformat_interface *) dst_interface;

	size_t size;

	cudaError_t status;

	switch (kind)
	{
		case cudaMemcpyHostToDevice:
		{
			size = src_multiformat->nx * src_multiformat->ops->cuda_elemsize;
			if (src_multiformat->cuda_ptr == NULL)
			{
				src_multiformat->cuda_ptr = malloc(size);
				if (src_multiformat->cuda_ptr == NULL)
					return -ENOMEM;
			}
			status = cudaMemcpy(dst_multiformat->cpu_ptr, src_multiformat->cpu_ptr, size, kind);
			if (!status)
				status = cudaDeviceSynchronize();
			if (STARPU_UNLIKELY(status))
				STARPU_CUDA_REPORT_ERROR(status);
			break;
		}
		case cudaMemcpyDeviceToHost:
		{
			size = src_multiformat->nx * src_multiformat->ops->cuda_elemsize;
			status = cudaMemcpy(dst_multiformat->cuda_ptr, src_multiformat->cuda_ptr, size, kind);
			if (!status)
				status = cudaDeviceSynchronize();
			if (STARPU_UNLIKELY(status))
				STARPU_CUDA_REPORT_ERROR(status);

			break;
		}
		case cudaMemcpyDeviceToDevice:
		{
			size = src_multiformat->nx * src_multiformat->ops->cuda_elemsize;
			status = cudaMemcpy(dst_multiformat->cuda_ptr, src_multiformat->cuda_ptr, size, kind);
			if (!status)
				status = cudaDeviceSynchronize();
			if (STARPU_UNLIKELY(status))
				STARPU_CUDA_REPORT_ERROR(status);
			break;
		}
		default:
			STARPU_ABORT();
	}

	return 0;
}

static int copy_ram_to_cuda(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node)
{
	return copy_cuda_common(src_interface, src_node, dst_interface, dst_node, cudaMemcpyHostToDevice);
}

static int copy_cuda_to_ram(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node)
{
	return copy_cuda_common(src_interface, src_node, dst_interface, dst_node, cudaMemcpyDeviceToHost);
}

static int copy_cuda_common_async(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED,
				  void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED,
				  cudaStream_t stream, enum cudaMemcpyKind kind)
{
	struct starpu_multiformat_interface *src_multiformat;
	struct starpu_multiformat_interface *dst_multiformat;

	src_multiformat = (struct starpu_multiformat_interface *) src_interface;
	dst_multiformat = (struct starpu_multiformat_interface *) dst_interface;

	size_t size;
	cudaError_t status;

	switch (kind)
	{
		case cudaMemcpyHostToDevice:
		{
			size = src_multiformat->nx * src_multiformat->ops->cuda_elemsize;
			if (src_multiformat->cuda_ptr == NULL)
			{
				src_multiformat->cuda_ptr = malloc(size);
				if (src_multiformat->cuda_ptr == NULL)
					return -ENOMEM;
			}

			status = cudaMemcpyAsync(dst_multiformat->cpu_ptr, src_multiformat->cpu_ptr, size, kind, stream);
			if (STARPU_UNLIKELY(status))
			{
				STARPU_CUDA_REPORT_ERROR(status);
			}
			break;
		}
		case cudaMemcpyDeviceToHost:
		{
			size = src_multiformat->nx * src_multiformat->ops->cuda_elemsize;
			status = cudaMemcpy(dst_multiformat->cuda_ptr, src_multiformat->cuda_ptr, size, kind);
			if (!status)
				status = cudaDeviceSynchronize();
			if (STARPU_UNLIKELY(status))
				STARPU_CUDA_REPORT_ERROR(status);

			break;
		}
		case cudaMemcpyDeviceToDevice:
		{
			size = src_multiformat->nx * src_multiformat->ops->cuda_elemsize;
			status = cudaMemcpyAsync(dst_multiformat->cuda_ptr, src_multiformat->cuda_ptr, size, kind, stream);
			if (STARPU_UNLIKELY(status))
				STARPU_CUDA_REPORT_ERROR(status);
			break;
		}
		default:
			STARPU_ABORT();
	}

	return 0;
}

static int copy_ram_to_cuda_async(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node, cudaStream_t stream)
{
	return copy_cuda_common_async(src_interface, src_node, dst_interface, dst_node, stream, cudaMemcpyHostToDevice);
}

static int copy_cuda_to_ram_async(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node, cudaStream_t stream)
{
	return copy_cuda_common_async(src_interface, src_node, dst_interface, dst_node, stream, cudaMemcpyDeviceToHost);
}

#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
static int copy_cuda_peer_common(void *src_interface, unsigned src_node,
				void *dst_interface, unsigned dst_node,
				cudaStream_t stream)
{
	struct starpu_multiformat_interface *src_multiformat;
	struct starpu_multiformat_interface *dst_multiformat;

	src_multiformat = (struct starpu_multiformat_interface *) src_interface;
	dst_multiformat = (struct starpu_multiformat_interface *) dst_interface;

	STARPU_ASSERT(src_multiformat != NULL);
	STARPU_ASSERT(dst_multiformat != NULL);
	STARPU_ASSERT(src_multiformat->ops != NULL);

	cudaError_t status;
	int size = src_multiformat->nx * src_multiformat->ops->cuda_elemsize;
	int src_dev = starpu_memory_node_get_devid(src_node);
	int dst_dev = starpu_memory_node_get_devid(dst_node);

	if (stream)
	{
		double start;
		starpu_interface_start_driver_copy_async(src_node, dst_node, &start);
		status = cudaMemcpyPeerAsync(dst_multiformat->cuda_ptr, dst_dev,
					     src_multiformat->cuda_ptr, src_dev,
					     size, stream);
		starpu_interface_end_driver_copy_async(src_node, dst_node, start);
		/* All good ! Still, returning -EAGAIN, because we will need to
                   check the transfert completion later */
		if (status == cudaSuccess)
			return -EAGAIN;
	}

	/* Either a synchronous transfert was requested, or the asynchronous one
           failed. */
	status = cudaMemcpyPeer(dst_multiformat->cuda_ptr, dst_dev,
				src_multiformat->cuda_ptr, src_dev,
				size);
	if (!status)
		status = cudaDeviceSynchronize();
	if (STARPU_UNLIKELY(status != cudaSuccess))
		STARPU_CUDA_REPORT_ERROR(status);

	starpu_interface_data_copy(src_node, dst_node, size);

	return 0;
}
#endif
static int copy_cuda_to_cuda(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED)
{
	if (src_node == dst_node)
	{
		return copy_cuda_common(src_interface, src_node, dst_interface, dst_node, cudaMemcpyDeviceToDevice);
	}
	else
	{
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
		return copy_cuda_peer_common(src_interface, src_node,
					     dst_interface, dst_node,
					     NULL);
#else
		STARPU_ABORT();
#endif
	}
}

static int copy_cuda_to_cuda_async(void *src_interface, unsigned src_node,
				   void *dst_interface, unsigned dst_node,
				   cudaStream_t stream)
{
	if (src_node == dst_node)
	{
		return copy_cuda_common_async(src_interface, src_node,
					      dst_interface, dst_node,
					      stream, cudaMemcpyDeviceToDevice);
	}
	else
	{
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
		return copy_cuda_peer_common(src_interface, src_node,
					     dst_interface, dst_node,
					     stream);
#else
		STARPU_ABORT();
#endif
	}
}
#endif /* STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
static int copy_ram_to_opencl_async(void *src_interface, unsigned src_node,
				    void *dst_interface, unsigned dst_node,
				    cl_event *event)
{
	int err, ret;
	size_t size;
	struct starpu_multiformat_interface *src_multiformat;
	struct starpu_multiformat_interface *dst_multiformat;

	src_multiformat = (struct starpu_multiformat_interface *) src_interface;
	dst_multiformat = (struct starpu_multiformat_interface *) dst_interface;

	STARPU_ASSERT(src_multiformat != NULL);
	STARPU_ASSERT(dst_multiformat != NULL);
	STARPU_ASSERT(src_multiformat->ops != NULL);

	size = src_multiformat->nx * src_multiformat->ops->opencl_elemsize;


	err = starpu_opencl_copy_ram_to_opencl(src_multiformat->cpu_ptr,
					       src_node,
					       (cl_mem) dst_multiformat->cpu_ptr,
					       dst_node,
					       size,
					       0,
					       event,
					       &ret);
        if (STARPU_UNLIKELY(err))
                STARPU_OPENCL_REPORT_ERROR(err);

	starpu_interface_data_copy(src_node, dst_node, size);
	return ret;
}

static int copy_opencl_to_ram_async(void *src_interface, unsigned src_node,
				    void *dst_interface, unsigned dst_node,
				    cl_event *event)
{
	int err, ret;
	size_t size;
	struct starpu_multiformat_interface *src_multiformat;
	struct starpu_multiformat_interface *dst_multiformat;

	src_multiformat = (struct starpu_multiformat_interface *) src_interface;
	dst_multiformat = (struct starpu_multiformat_interface *) dst_interface;

	STARPU_ASSERT(src_multiformat != NULL);
	STARPU_ASSERT(dst_multiformat != NULL);
	STARPU_ASSERT(src_multiformat->ops != NULL);
	STARPU_ASSERT(dst_multiformat->ops != NULL);

	size = src_multiformat->nx * src_multiformat->ops->opencl_elemsize;

	if (dst_multiformat->opencl_ptr == NULL)
	{
		/* XXX : it is weird that we might have to allocate memory here... */
		dst_multiformat->opencl_ptr = malloc(dst_multiformat->nx * dst_multiformat->ops->opencl_elemsize);
		STARPU_ASSERT_MSG(dst_multiformat->opencl_ptr != NULL || dst_multiformat->nx * dst_multiformat->ops->opencl_elemsize == 0, "Cannot allocate %ld bytes\n", (long) (dst_multiformat->nx * dst_multiformat->ops->opencl_elemsize));
	}
	err = starpu_opencl_copy_opencl_to_ram((cl_mem)src_multiformat->opencl_ptr,
					       src_node,
					       dst_multiformat->opencl_ptr,
					       dst_node,
					       size,
					       0,
					       event,
					       &ret);
        if (STARPU_UNLIKELY(err))
                STARPU_OPENCL_REPORT_ERROR(err);

	starpu_interface_data_copy(src_node, dst_node, size);


	return ret;
}

static int copy_ram_to_opencl(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED,
                              void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED)
{
        return copy_ram_to_opencl_async(src_interface, src_node, dst_interface, dst_node, NULL);
}

static int copy_opencl_to_ram(void *src_interface, unsigned src_node STARPU_ATTRIBUTE_UNUSED,
				void *dst_interface, unsigned dst_node STARPU_ATTRIBUTE_UNUSED)
{
        return copy_opencl_to_ram_async(src_interface, src_node, dst_interface, dst_node, NULL);
}

static int copy_opencl_to_opencl(void *src_interface, unsigned src_node,
                                 void *dst_interface, unsigned dst_node)
{
	(void) src_interface;
	(void) dst_interface;
	(void) src_node;
	(void) dst_node;

	STARPU_ASSERT_MSG(0, "XXX multiformat copy OpenCL-OpenCL not supported yet (TODO)");
	return 0;
}
#endif

#ifdef STARPU_USE_MIC
static int copy_mic_common_ram_to_mic(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node,
						   int (*copy_func)(void *, unsigned, void *, unsigned, size_t))
{
	struct starpu_multiformat_interface *src_multiformat = src_interface;
	struct starpu_multiformat_interface *dst_multiformat = dst_interface;

	STARPU_ASSERT(src_multiformat != NULL);
	STARPU_ASSERT(dst_multiformat != NULL);
	STARPU_ASSERT(dst_multiformat->ops != NULL);

	size_t size = dst_multiformat->nx * dst_multiformat->ops->mic_elemsize;
	if (src_multiformat->mic_ptr == NULL)
	{
		src_multiformat->mic_ptr = malloc(size);
		if (src_multiformat->mic_ptr == NULL)
			return -ENOMEM;
	}

	copy_func(src_multiformat->cpu_ptr, src_node, dst_multiformat->cpu_ptr, dst_node, size);

	starpu_interface_data_copy(src_node, dst_node, size);

	return 0;
}

static int copy_mic_common_mic_to_ram(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node,
						   int (*copy_func)(void *, unsigned, void *, unsigned, size_t))
{
	struct starpu_multiformat_interface *src_multiformat = src_interface;
	struct starpu_multiformat_interface *dst_multiformat = dst_interface;

	STARPU_ASSERT(src_multiformat != NULL);
	STARPU_ASSERT(dst_multiformat != NULL);
	STARPU_ASSERT(dst_multiformat->ops != NULL);

	size_t size = src_multiformat->nx * src_multiformat->ops->mic_elemsize;
	copy_func(src_multiformat->mic_ptr, src_node, dst_multiformat->mic_ptr, dst_node, size);

	starpu_interface_data_copy(src_node, dst_node, size);

	return 0;
}

static int copy_ram_to_mic(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node)
{
	return copy_mic_common_ram_to_mic(src_interface, src_node, dst_interface, dst_node, _starpu_mic_copy_ram_to_mic);
}

static int copy_mic_to_ram(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node)
{
	return copy_mic_common_mic_to_ram(src_interface, src_node, dst_interface, dst_node, _starpu_mic_copy_mic_to_ram);
}

static int copy_ram_to_mic_async(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node)
{
	copy_mic_common_ram_to_mic(src_interface, src_node, dst_interface, dst_node, _starpu_mic_copy_ram_to_mic_async);
	return -EAGAIN;
}

static int copy_mic_to_ram_async(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node)
{
	copy_mic_common_mic_to_ram(src_interface, src_node, dst_interface, dst_node, _starpu_mic_copy_mic_to_ram_async);
	return -EAGAIN;
}
#endif
