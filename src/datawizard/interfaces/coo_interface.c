/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012 inria
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
#include <common/fxt.h>
#include <datawizard/memalloc.h>

static int
copy_ram_to_ram(void *src_interface, STARPU_ATTRIBUTE_UNUSED unsigned src_node,
		void *dst_interface, STARPU_ATTRIBUTE_UNUSED unsigned dst_node)
{
	size_t size = 0;
	struct starpu_coo_interface *src_coo, *dst_coo;

	src_coo = (struct starpu_coo_interface *) src_interface;
	dst_coo = (struct starpu_coo_interface *) dst_interface;

	size = src_coo->n_values * sizeof(src_coo->columns[0]);
	memcpy((void *) dst_coo->columns, (void *) src_coo->columns, size);

	/* sizeof(src_coo->columns[0]) == sizeof(src_coo->rows[0]) */
	memcpy((void *) dst_coo->rows, (void *) src_coo->rows, size);

	size = src_coo->n_values * src_coo->elemsize;
	memcpy((void *) dst_coo->values, (void *) src_coo->values, size);

	_STARPU_TRACE_DATA_COPY(src_node, dst_node,
		src_coo->n_values *
		(2 * sizeof(src_coo->rows[0]) + src_coo->elemsize));

	return 0;
}

#ifdef STARPU_USE_CUDA
static int
copy_cuda_async_sync(void *src_interface, unsigned src_node,
		     void *dst_interface, unsigned dst_node,
		     cudaStream_t stream, enum cudaMemcpyKind kind)
{
	int ret;
	size_t size = 0;
	struct starpu_coo_interface *src_coo, *dst_coo;

	src_coo = (struct starpu_coo_interface *) src_interface;
	dst_coo = (struct starpu_coo_interface *) dst_interface;

	size = src_coo->n_values * sizeof(src_coo->columns[0]);
	ret = starpu_cuda_copy_async_sync(
		(void *) src_coo->columns,
		src_node,
		(void *) dst_coo->columns,
		dst_node,
		size,
		stream,
		kind);
	if (ret == 0)
		stream = NULL;

	/* sizeof(src_coo->columns[0]) == sizeof(src_coo->rows[0]) */
	ret = starpu_cuda_copy_async_sync(
		(void *) src_coo->rows,
		src_node,
		(void *) dst_coo->rows,
		dst_node,
		size,
		stream,
		kind);
	if (ret == 0)
		stream = NULL;

	size = src_coo->n_values * src_coo->elemsize;
	ret = starpu_cuda_copy_async_sync(
		(void *) src_coo->values,
		src_node,
		(void *) dst_coo->values,
		dst_node,
		size,
		stream,
		kind);

	_STARPU_TRACE_DATA_COPY(src_node, dst_node,
		src_coo->n_values *
		(2 * sizeof(src_coo->rows[0]) + src_coo->elemsize));
	return ret;
}

static int
copy_ram_to_cuda(void *src_interface, unsigned src_node,
		 void *dst_interface, unsigned dst_node)
{
	return copy_cuda_async_sync(src_interface, src_node,
				    dst_interface, dst_node,
				    NULL, cudaMemcpyHostToDevice);
}

static int
copy_cuda_to_ram(void *src_interface, unsigned src_node,
		 void *dst_interface, unsigned dst_node)
{
	return copy_cuda_async_sync(src_interface, src_node,
				    dst_interface, dst_node,
				    NULL, cudaMemcpyDeviceToHost);
}

static int
copy_ram_to_cuda_async(void *src_interface, unsigned src_node,
		       void *dst_interface, unsigned dst_node,
		       cudaStream_t stream)
{
	return copy_cuda_async_sync(src_interface, src_node,
				    dst_interface, dst_node,
				    stream, cudaMemcpyHostToDevice);
}

static int
copy_cuda_to_ram_async(void *src_interface, unsigned src_node,
		       void *dst_interface, unsigned dst_node,
		       cudaStream_t stream)
{
	return copy_cuda_async_sync(src_interface, src_node,
				    dst_interface, dst_node,
				    stream, cudaMemcpyDeviceToHost);
}

static int
copy_cuda_to_cuda(void *src_interface, unsigned src_node,
		  void *dst_interface, unsigned dst_node)
{
	return copy_cuda_async_sync(src_interface, src_node,
				    dst_interface, dst_node,
				    NULL, cudaMemcpyDeviceToDevice);
}

#ifdef NO_STRIDE
static int
copy_cuda_to_cuda_async(void *src_interface, unsigned src_node,
			void *dst_interface, unsigned dst_node,
			cudaStream_t stream)
{
	return copy_cuda_async_sync(src_interface, src_node,
				    dst_interface, dst_node,
				    stream, cudaMemcpyDeviceToDevice);
}
#endif /* !NO_STRIDE */
#endif /* !STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
static int
copy_ram_to_opencl_async(void *src_interface, unsigned src_node,
			 void *dst_interface, unsigned dst_node,
			 cl_event *event)
{
	int ret = 0;
	cl_int err;
	size_t size = 0;
	struct starpu_coo_interface *src_coo, *dst_coo;

	src_coo = (struct starpu_coo_interface *) src_interface;
	dst_coo = (struct starpu_coo_interface *) dst_interface;


	size = src_coo->n_values * sizeof(src_coo->columns[0]);
	err = starpu_opencl_copy_ram_to_opencl(
		(void *) src_coo->columns,
		src_node,
		(cl_mem) dst_coo->columns,
		dst_node,
		size,
		0,
		event,
		NULL);
	if (STARPU_UNLIKELY(err))
		STARPU_OPENCL_REPORT_ERROR(err);

	/* sizeof(src_coo->columns[0]) == sizeof(src_coo->rows[0]) */
	err = starpu_opencl_copy_ram_to_opencl(
		(void *) src_coo->rows,
		src_node,
		(cl_mem) dst_coo->rows,
		dst_node,
		size,
		0,
		event,
		NULL);
	if (STARPU_UNLIKELY(err))
		STARPU_OPENCL_REPORT_ERROR(err);

	size = src_coo->n_values * src_coo->elemsize;
	err = starpu_opencl_copy_ram_to_opencl(
		(void *) src_coo->values,
		src_node,
		(cl_mem) dst_coo->values,
		dst_node,
		size,
		0,
		event,
		&ret);
	if (STARPU_UNLIKELY(err))
		STARPU_OPENCL_REPORT_ERROR(err);

	_STARPU_TRACE_DATA_COPY(src_node, dst_node,
		src_coo->n_values *
		(2 * sizeof(src_coo->rows[0]) + src_coo->elemsize));

	return ret;
}

static int
copy_opencl_to_ram_async(void *src_interface, unsigned src_node,
			 void *dst_interface, unsigned dst_node,
			 cl_event *event)
{
	int ret = 0;
	cl_int err;
	size_t size = 0;
	struct starpu_coo_interface *src_coo, *dst_coo;

	src_coo = (struct starpu_coo_interface *) src_interface;
	dst_coo = (struct starpu_coo_interface *) dst_interface;

	size = src_coo->n_values * sizeof(src_coo->columns[0]);
	err = starpu_opencl_copy_opencl_to_ram(
		(void *) src_coo->columns,
		src_node,
		(cl_mem) dst_coo->columns,
		dst_node,
		size,
		0,
		event,
		NULL);
	if (STARPU_UNLIKELY(err))
		STARPU_OPENCL_REPORT_ERROR(err);

	/* sizeof(src_coo->columns[0]) == sizeof(src_coo->rows[0]) */
	err = starpu_opencl_copy_opencl_to_ram(
		(void *) src_coo->rows,
		src_node,
		(cl_mem) dst_coo->rows,
		dst_node,
		size,
		0,
		event,
		NULL);
	if (STARPU_UNLIKELY(err))
		STARPU_OPENCL_REPORT_ERROR(err);

	size = src_coo->n_values * src_coo->elemsize;
	err = starpu_opencl_copy_opencl_to_ram(
		(void *) src_coo->values,
		src_node,
		(cl_mem) dst_coo->values,
		dst_node,
		size,
		0,
		event,
		&ret);
	if (STARPU_UNLIKELY(err))
		STARPU_OPENCL_REPORT_ERROR(err);

	_STARPU_TRACE_DATA_COPY(src_node, dst_node,
		src_coo->n_values *
		(2 * sizeof(src_coo->rows[0]) + src_coo->elemsize));

	return ret;
}

static int
copy_ram_to_opencl(void *src_interface, unsigned src_node,
		   void *dst_interface, unsigned dst_node)
{
	return copy_ram_to_opencl_async(src_interface, src_node,
					dst_interface, dst_node,
					NULL);
}
static int
copy_opencl_to_ram(void *src_interface, unsigned src_node,
		   void *dst_interface, unsigned dst_node)
{
	return copy_opencl_to_ram_async(src_interface, src_node,
					dst_interface, dst_node,
					NULL);
}
#endif /* !STARPU_USE_OPENCL */

static struct starpu_data_copy_methods coo_copy_data_methods =
{
	.ram_to_ram          = copy_ram_to_ram,
#ifdef STARPU_USE_CUDA
	.ram_to_cuda         = copy_ram_to_cuda,
	.cuda_to_ram         = copy_cuda_to_ram,
	.ram_to_cuda_async   = copy_ram_to_cuda_async,
	.cuda_to_ram_async   = copy_cuda_to_ram_async,
	.cuda_to_cuda        = copy_cuda_to_cuda,
#ifdef NO_STRIDE
	.cuda_to_cuda_async  = copy_cuda_to_cuda_async,
#endif
#else
#ifdef STARPU_SIMGRID
#ifdef NO_STRIDE
	/* Enable GPU-GPU transfers in simgrid */
	.cuda_to_cuda_async = 1,
#endif
#endif
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
	.ram_to_opencl       = copy_ram_to_opencl,
	.opencl_to_ram       = copy_opencl_to_ram,
	.ram_to_opencl_async = copy_ram_to_opencl_async,
	.opencl_to_ram_async = copy_opencl_to_ram_async,
#endif /* !STARPU_USE_OPENCL */
};

static void
register_coo_handle(starpu_data_handle_t handle, uint32_t home_node,
		    void *data_interface)
{
	struct starpu_coo_interface *coo_interface =
		(struct starpu_coo_interface *) data_interface;

	unsigned node;
	for (node = 0; node < STARPU_MAXNODES; node++)
	{
		struct starpu_coo_interface *local_interface;
		local_interface = (struct starpu_coo_interface *)
			starpu_data_get_interface_on_node(handle, node);

		if (node == home_node)
		{
			local_interface->values = coo_interface->values;
			local_interface->columns = coo_interface->columns;
			local_interface->rows = coo_interface->rows;
		}
		else
		{
			local_interface->values = 0;
			local_interface->columns = 0;
			local_interface->rows = 0;
		}

		local_interface->nx = coo_interface->nx;
		local_interface->ny = coo_interface->ny;
		local_interface->n_values = coo_interface->n_values;
		local_interface->elemsize = coo_interface->elemsize;
	}
}

static ssize_t
allocate_coo_buffer_on_node(void *data_interface, uint32_t dst_node)
{
	uint32_t *addr_columns = NULL;
	uint32_t *addr_rows = NULL;
	uintptr_t addr_values = 0;

	struct starpu_coo_interface *coo_interface =
		(struct starpu_coo_interface *) data_interface;

	uint32_t n_values = coo_interface->n_values;
	size_t elemsize = coo_interface->elemsize;

	addr_columns = (void*) starpu_allocate_buffer_on_node(dst_node, n_values * sizeof(coo_interface->columns[0]));
	if (STARPU_UNLIKELY(addr_columns == NULL))
		goto fail_columns;
	addr_rows = (void*) starpu_allocate_buffer_on_node(dst_node, n_values * sizeof(coo_interface->rows[0]));
	if (STARPU_UNLIKELY(addr_rows == NULL))
		goto fail_rows;
	addr_values = starpu_allocate_buffer_on_node(dst_node, n_values * elemsize);
	if (STARPU_UNLIKELY(addr_values == (uintptr_t) NULL))
		goto fail_values;

	coo_interface->columns = addr_columns;
	coo_interface->rows = addr_rows;
	coo_interface->values = addr_values;

	return n_values * (sizeof(coo_interface->columns[0]) + sizeof(coo_interface->rows[0]) + elemsize);

fail_values:
	starpu_free_buffer_on_node(dst_node, (uintptr_t) addr_rows, n_values * sizeof(coo_interface->rows[0]));
fail_rows:
	starpu_free_buffer_on_node(dst_node, (uintptr_t) addr_columns, n_values * sizeof(coo_interface->columns[0]));
fail_columns:
	return -ENOMEM;
}

static void
free_coo_buffer_on_node(void *data_interface, uint32_t node)
{
	struct starpu_coo_interface *coo_interface = (struct starpu_coo_interface *) data_interface;
	uint32_t n_values = coo_interface->n_values;
	size_t elemsize = coo_interface->elemsize;

	starpu_free_buffer_on_node(node, (uintptr_t) coo_interface->columns, n_values * sizeof(coo_interface->columns[0]));
	starpu_free_buffer_on_node(node, (uintptr_t) coo_interface->rows, n_values * sizeof(coo_interface->rows[0]));
	starpu_free_buffer_on_node(node, coo_interface->values, n_values * elemsize);
}

static size_t
coo_interface_get_size(starpu_data_handle_t handle)
{
	struct starpu_coo_interface *coo_interface;
	coo_interface = (struct starpu_coo_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	return coo_interface->nx * coo_interface->ny * coo_interface->elemsize;
}

static uint32_t
coo_interface_footprint(starpu_data_handle_t handle)
{
	struct starpu_coo_interface *coo_interface;
	coo_interface = (struct starpu_coo_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	return starpu_crc32_be(coo_interface->nx * coo_interface->ny, 0);
}

static int
coo_compare(void *a, void *b)
{
	struct starpu_coo_interface *coo_a, *coo_b;

	coo_a = (struct starpu_coo_interface *) a;
	coo_b = (struct starpu_coo_interface *) b;

	return (coo_a->nx == coo_b->nx &&
		coo_a->ny == coo_b->ny &&
		coo_a->n_values == coo_b->n_values &&
		coo_a->elemsize == coo_b->elemsize);
}

static void
display_coo_interface(starpu_data_handle_t handle, FILE *f)
{
	struct starpu_coo_interface *coo_interface =
	coo_interface = (struct starpu_coo_interface *)
		starpu_data_get_interface_on_node(handle, 0);

	fprintf(f, "%u\t%u", coo_interface->nx, coo_interface->ny);
}

struct starpu_data_interface_ops _starpu_interface_coo_ops =
{
	.register_data_handle  = register_coo_handle,
	.allocate_data_on_node = allocate_coo_buffer_on_node,
	.handle_to_pointer     = NULL,
	.free_data_on_node     = free_coo_buffer_on_node,
	.copy_methods          = &coo_copy_data_methods,
	.get_size              = coo_interface_get_size,
	.footprint             = coo_interface_footprint,
	.compare               = coo_compare,
#ifdef STARPU_USE_GORDON
	.convert_to_gordon     = NULL,
#endif
	.interfaceid           = STARPU_COO_INTERFACE_ID,
	.interface_size        = sizeof(struct starpu_coo_interface),
	.display               = display_coo_interface
};

void
starpu_coo_data_register(starpu_data_handle_t *handleptr, uint32_t home_node,
			 uint32_t nx, uint32_t ny, uint32_t n_values,
			 uint32_t *columns, uint32_t *rows,
			 uintptr_t values, size_t elemsize)
{
	struct starpu_coo_interface coo_interface =
	{
		.values = values,
		.columns = columns,
		.rows = rows,
		.nx = nx,
		.ny = ny,
		.n_values = n_values,
		.elemsize = elemsize,
	};

	starpu_data_register(handleptr, home_node, &coo_interface,
			     &_starpu_interface_coo_ops);
}
