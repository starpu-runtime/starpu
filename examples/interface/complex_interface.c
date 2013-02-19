/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012, 2013  Centre National de la Recherche Scientifique
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

#include "complex_interface.h"

double *starpu_complex_get_real(starpu_data_handle_t handle)
{
	struct starpu_complex_interface *complex_interface =
		(struct starpu_complex_interface *) starpu_data_get_interface_on_node(handle, 0);

	return complex_interface->real;
}

double *starpu_complex_get_imaginary(starpu_data_handle_t handle)
{
	struct starpu_complex_interface *complex_interface =
		(struct starpu_complex_interface *) starpu_data_get_interface_on_node(handle, 0);

	return complex_interface->imaginary;
}

int starpu_complex_get_nx(starpu_data_handle_t handle)
{
	struct starpu_complex_interface *complex_interface =
		(struct starpu_complex_interface *) starpu_data_get_interface_on_node(handle, 0);

	return complex_interface->nx;
}

static void complex_register_data_handle(starpu_data_handle_t handle, unsigned home_node, void *data_interface)
{
	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *) data_interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_complex_interface *local_interface = (struct starpu_complex_interface *)
			starpu_data_get_interface_on_node(handle, node);

		local_interface->real = complex_interface->real;
		local_interface->imaginary = complex_interface->imaginary;
		local_interface->nx = complex_interface->nx;
	}
}

static starpu_ssize_t complex_allocate_data_on_node(void *data_interface, unsigned node)
{
	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *) data_interface;

	double *addr_real = 0;
	double *addr_imaginary = 0;
	ssize_t requested_memory = complex_interface->nx * sizeof(complex_interface->real[0]);

	addr_real = (double*) starpu_allocate_buffer_on_node(node, requested_memory);
	if (!addr_real)
		goto fail_real;
	addr_imaginary = (double*) starpu_allocate_buffer_on_node(node, requested_memory);
	if (!addr_imaginary)
		goto fail_imaginary;

	/* update the data properly in consequence */
	complex_interface->real = addr_real;
	complex_interface->imaginary = addr_imaginary;

	return 2*requested_memory;

fail_imaginary:
	starpu_free_buffer_on_node(node, (uintptr_t) addr_real, requested_memory);
fail_real:
	return -ENOMEM;
}

static void complex_free_data_on_node(void *data_interface, unsigned node)
{
	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *) data_interface;
	ssize_t requested_memory = complex_interface->nx * sizeof(complex_interface->real[0]);

	starpu_free_buffer_on_node(node, (uintptr_t) complex_interface->real, requested_memory);
	starpu_free_buffer_on_node(node, (uintptr_t) complex_interface->imaginary, requested_memory);
}

static size_t complex_get_size(starpu_data_handle_t handle)
{
	size_t size;
	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *) starpu_data_get_interface_on_node(handle, 0);

	size = complex_interface->nx * 2 * sizeof(double);
	return size;
}

static uint32_t complex_footprint(starpu_data_handle_t handle)
{
	return starpu_crc32_be(starpu_complex_get_nx(handle), 0);
}

static void *complex_handle_to_pointer(starpu_data_handle_t handle, unsigned node)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *)
		starpu_data_get_interface_on_node(handle, node);

	return (void*) complex_interface->real;
}

static int complex_pack_data(starpu_data_handle_t handle, unsigned node, void **ptr, size_t *count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *)
		starpu_data_get_interface_on_node(handle, node);

	*count = complex_get_size(handle);
	*ptr = malloc(*count);
	memcpy(*ptr, complex_interface->real, complex_interface->nx*sizeof(double));
	memcpy(*ptr+complex_interface->nx*sizeof(double), complex_interface->imaginary, complex_interface->nx*sizeof(double));

	return 0;
}

static int complex_unpack_data(starpu_data_handle_t handle, unsigned node, void *ptr, size_t count)
{
	STARPU_ASSERT(starpu_data_test_if_allocated_on_node(handle, node));

	struct starpu_complex_interface *complex_interface = (struct starpu_complex_interface *)
		starpu_data_get_interface_on_node(handle, node);

	memcpy(complex_interface->real, ptr, complex_interface->nx*sizeof(double));
	memcpy(complex_interface->imaginary, ptr+complex_interface->nx*sizeof(double), complex_interface->nx*sizeof(double));

	return 0;
}

#ifdef STARPU_USE_CUDA
static int copy_cuda_async_sync(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	struct starpu_complex_interface *src_complex = src_interface;
	struct starpu_complex_interface *dst_complex = dst_interface;

	cudaStream_t sstream = stream;
	int ret;

	ret = starpu_cuda_copy_async_sync((void *)src_complex->real, src_node, (void *)dst_complex->real, dst_node,
					  src_complex->nx*sizeof(src_complex->real[0]), sstream, kind);
	if (ret == 0) sstream = NULL;

	ret = starpu_cuda_copy_async_sync((char *)src_complex->imaginary, src_node, (char *)dst_complex->imaginary, dst_node,
					  src_complex->nx*sizeof(src_complex->imaginary[0]), sstream, kind);
	return ret;
}

static int copy_ram_to_cuda(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node)
{
     return copy_cuda_async_sync(src_interface, src_node, dst_interface, dst_node, cudaMemcpyHostToDevice, NULL);
}

static int copy_ram_to_cuda_async(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cudaStream_t stream)
{
	return copy_cuda_async_sync(src_interface, src_node, dst_interface, dst_node, cudaMemcpyHostToDevice, stream);
}

static int copy_cuda_to_ram(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node)
{
	return copy_cuda_async_sync(src_interface, src_node, dst_interface, dst_node, cudaMemcpyDeviceToHost, NULL);
}

static int copy_cuda_to_ram_async(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cudaStream_t stream)
{
	return copy_cuda_async_sync(src_interface, src_node, dst_interface, dst_node, cudaMemcpyDeviceToHost, stream);
}
#endif

#ifdef STARPU_USE_OPENCL
static int copy_ram_to_opencl_async(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cl_event *event)
{
	struct starpu_complex_interface *src_complex = src_interface;
	struct starpu_complex_interface *dst_complex = dst_interface;
	cl_int err;
	int ret;

	err = starpu_opencl_copy_ram_to_opencl(src_complex->real,
					       src_node,
					       (cl_mem) dst_complex->real,
					       dst_node,
					       src_complex->nx * sizeof(src_complex->real[0]),
					       0,
					       event,
					       &ret);
	if (STARPU_UNLIKELY(err != CL_SUCCESS))
		STARPU_OPENCL_REPORT_ERROR(err);
	if (ret == 0)
		event = NULL;

	err = starpu_opencl_copy_ram_to_opencl(src_complex->imaginary,
					       src_node,
					       (cl_mem) dst_complex->imaginary,
					       dst_node,
					       src_complex->nx * sizeof(src_complex->imaginary[0]),
					       0,
					       event,
					       &ret);
	if (STARPU_UNLIKELY(err != CL_SUCCESS))
		STARPU_OPENCL_REPORT_ERROR(err);

	return ret;
}

static int copy_ram_to_opencl(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node)
{
        return copy_ram_to_opencl_async(src_interface, src_node, dst_interface, dst_node, NULL);
}

static int copy_opencl_to_ram_async(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cl_event *event)
{
	struct starpu_complex_interface *src_complex = src_interface;
	struct starpu_complex_interface *dst_complex = dst_interface;
	cl_int err;
	int ret;

	err = starpu_opencl_copy_opencl_to_ram((cl_mem) src_complex->real,
					       src_node,
					       dst_complex->real,
					       dst_node,
					       src_complex->nx * sizeof(src_complex->real[0]),
					       0,
					       event,
					       &ret);
	if (STARPU_UNLIKELY(err != CL_SUCCESS))
		STARPU_OPENCL_REPORT_ERROR(err);
	if (ret == 0)
		event = NULL;

	err = starpu_opencl_copy_opencl_to_ram((cl_mem) src_complex->imaginary,
					       src_node,
					       dst_complex->imaginary,
					       dst_node,
					       src_complex->nx * sizeof(src_complex->imaginary[0]),
					       0,
					       event,
					       &ret);
	if (STARPU_UNLIKELY(err != CL_SUCCESS))
		STARPU_OPENCL_REPORT_ERROR(err);

	return ret;
}

static int copy_opencl_to_ram(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node)
{
        return copy_opencl_to_ram_async(src_interface, src_node, dst_interface, dst_node, NULL);
}
#endif

static struct starpu_data_copy_methods complex_copy_methods =
{
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
};

static struct starpu_data_interface_ops interface_complex_ops =
{
	.register_data_handle = complex_register_data_handle,
	.allocate_data_on_node = complex_allocate_data_on_node,
	.free_data_on_node = complex_free_data_on_node,
	.copy_methods = &complex_copy_methods,
	.get_size = complex_get_size,
	.footprint = complex_footprint,
	.interfaceid = -1,
	.interface_size = sizeof(struct starpu_complex_interface),
	.handle_to_pointer = complex_handle_to_pointer,
	.pack_data = complex_pack_data,
	.unpack_data = complex_unpack_data
};

void starpu_complex_data_register(starpu_data_handle_t *handleptr, unsigned home_node, double *real, double *imaginary, int nx)
{
	struct starpu_complex_interface complex =
	{
		.real = real,
		.imaginary = imaginary,
		.nx = nx
	};

	if (interface_complex_ops.interfaceid == -1)
	{
		interface_complex_ops.interfaceid = starpu_data_interface_get_next_id();
	}

	starpu_data_register(handleptr, home_node, &complex, &interface_complex_ops);
}
