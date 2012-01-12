/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012 INRIA
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
#ifdef STARPU_USE_OPENCL
#include <starpu_opencl.h>
#endif

#include "multiformat_types.h"

#if STARPU_USE_CUDA
static int copy_cuda_common_async(void *src_interface, unsigned src_node,
				  void *dst_interface, unsigned dst_node,
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
		/*
		 * XXX : Should we do that ? It is a mix between copy and conversion...
		 */
		/*
		 * Copying the data to the CUDA device.
		 */
		status = cudaMemcpyAsync(dst_multiformat->cpu_ptr,
					 src_multiformat->cpu_ptr,
					 src_multiformat->nx * src_multiformat->ops->cpu_elemsize,
					 kind, stream);
		assert(status == cudaSuccess);

		/*
		 * Copying the real data (that is pointed to).
		 */
		float *x = malloc(src_multiformat->nx * sizeof(float));
		float *y = malloc(src_multiformat->nx * sizeof(float));
		assert(x && y);

		int i;
		for (i = 0; i < src_multiformat->nx; i++)
		{
			struct point *p;
			p = (struct point *) src_multiformat->cpu_ptr;
			x[i] = p[i].x;
			y[i] = p[i].y;
		}

		void *rets[2];
		unsigned size = src_multiformat->nx * sizeof(float);

		status = cudaMalloc(&rets[0], sizeof(float)*src_multiformat->nx);
		assert(status == cudaSuccess);
		status = cudaMemcpyAsync(rets[0], x, size, kind, stream);
		assert(status == cudaSuccess);

		status = cudaMalloc(&rets[1], sizeof(float)*src_multiformat->nx);
		assert(status == cudaSuccess);
		status = cudaMemcpyAsync(rets[1], y, size, kind, stream);
		assert(status == cudaSuccess);

		status = cudaMemcpyAsync(dst_multiformat->cuda_ptr, rets,
				2*sizeof(void*), kind, stream);
		assert(status == cudaSuccess);

		free(x);
		free(y);
		break;
	}
	case cudaMemcpyDeviceToHost:
	{
		/*
		 * Copying the cuda_ptr from the cuda device to the RAM.
		 */
		size = sizeof(struct struct_of_arrays);
		if (!dst_multiformat->cuda_ptr)
		{
			dst_multiformat->cuda_ptr = calloc(1, size);
			assert(dst_multiformat->cuda_ptr != NULL);
		}


		/* Getting the addresses of our data on the CUDA device. */
		void *addrs[2];
		status = cudaMemcpyAsync(addrs, src_multiformat->cuda_ptr,
					 2 * sizeof(void*), kind, stream);
		assert(status == cudaSuccess);


		/*
		 * Getting the real data.
		 */
		struct struct_of_arrays *soa;
		soa = (struct struct_of_arrays *) dst_multiformat->cuda_ptr;
		size = src_multiformat->nx * sizeof(float);

		if (!soa->x)
			soa->x = malloc(size);
		status = cudaMemcpyAsync(soa->x, addrs[0], size, kind, stream);
		assert(status == cudaSuccess);
		

		if (!soa->y)
			soa->y = malloc(size);
		status = cudaMemcpyAsync(soa->y, addrs[1], size, kind, stream);
		assert(status == cudaSuccess);

		/* Let's free this. */
		status = cudaFree(addrs[0]);
		assert(status == cudaSuccess);
		status = cudaFree(addrs[1]);
		assert(status == cudaSuccess);
		break;
	}
	default:
		assert(0);
	}

	return 0;
}

static int
copy_ram_to_cuda_async(void *src_interface, unsigned src_node,
		       void *dst_interface, unsigned dst_node,
		       cudaStream_t stream)
{
	fprintf(stderr, "ENTER %s\n", __func__);
	copy_cuda_common_async(src_interface, src_node,
				dst_interface, dst_node,
				stream, cudaMemcpyHostToDevice);
	return 0;
}

static int
copy_cuda_to_ram_async(void *src_interface, unsigned src_node,
		       void *dst_interface, unsigned dst_node,
		       cudaStream_t stream)
{
	fprintf(stderr, "ENTER %s\n", __func__);
	copy_cuda_common_async(src_interface, src_node,
				dst_interface, dst_node,
				stream, cudaMemcpyDeviceToHost);
	return 0;
}

static int
copy_ram_to_cuda(void *src_interface, unsigned src_node,
		 void *dst_interface, unsigned dst_node)
{
	/* TODO */
	fprintf(stderr, "ENTER %s\n", __func__);
	return 1;
}

static int
copy_cuda_to_ram(void *src_interface, unsigned src_node,
		 void *dst_interface, unsigned dst_node)
{
	/* TODO */
	fprintf(stderr, "ENTER %s\n", __func__);
	return 1;
}
#endif /* !STARPU_USE_CUDA */


#ifdef STARPU_USE_OPENCL
static cl_int
_opencl_malloc(cl_context context, cl_mem *mem, size_t size, cl_mem_flags flags)
{
	cl_int err;
        cl_mem memory;

	memory = clCreateBuffer(context, flags, size, NULL, &err);
	if (err != CL_SUCCESS)
		return err;

        *mem = memory;
        return CL_SUCCESS;
}

static cl_int
_opencl_copy_ram_to_opencl_async_sync(void *ptr, unsigned src_node,
				      cl_mem buffer, unsigned dst_node,
				      size_t size, size_t offset,
				      cl_event *event, int *ret,
				      cl_command_queue queue)
{
        cl_int err;
        cl_bool blocking;

        blocking = (event == NULL) ? CL_TRUE : CL_FALSE;

        err = clEnqueueWriteBuffer(queue, buffer, blocking, offset, size, ptr, 0, NULL, event);

        if (err == CL_SUCCESS)
                *ret = (event == NULL) ? 0 : -EAGAIN;

	return err;
}

static cl_int
_opencl_copy_opencl_to_ram(cl_mem buffer, unsigned src_node,
			   void *ptr, unsigned dst_node,
			   size_t size, size_t offset, cl_event *event,
			   cl_command_queue queue)

{
        cl_int err;
        cl_bool blocking;

        blocking = (event == NULL) ? CL_TRUE : CL_FALSE;
        err = clEnqueueReadBuffer(queue, buffer, blocking, offset, size, ptr, 0, NULL, event);

        return err;
}

static int
copy_ram_to_opencl(void *src_interface, unsigned src_node,
		   void *dst_interface, unsigned dst_node)
{
	return 1;
}

static int
copy_opencl_to_ram(void *src_interface, unsigned src_node,
		   void *dst_interface, unsigned dst_node)
{
	return 1;
}

cl_mem xy[2];
static int
copy_ram_to_opencl_async(void *src_interface, unsigned src_node,
			 void *dst_interface, unsigned dst_node,
			 void *event)
{
	(void) event;
	FPRINTF(stderr, "Enter %s\n", __func__);
	struct starpu_multiformat_interface *src_mf;
	struct starpu_multiformat_interface *dst_mf;

	src_mf = (struct starpu_multiformat_interface *) src_interface;
	dst_mf = (struct starpu_multiformat_interface *) dst_interface;

	/*
	 * Opencl stuff.
	 */
	cl_context context;
	cl_command_queue queue;
	int id = starpu_worker_get_id();
	int devid = starpu_worker_get_devid(id);
	starpu_opencl_get_queue(devid, &queue);
	starpu_opencl_get_context(devid, &context);

	/*
	 * Copying the cpu pointer to the OpenCL device.
	 */
	int err;
	cl_int ret;
	size_t cpu_size = src_mf->nx * src_mf->ops->cpu_elemsize;
	err = _opencl_copy_ram_to_opencl_async_sync(src_mf->cpu_ptr,
						    src_node,
						    dst_mf->cpu_ptr,
						    dst_node,
						    cpu_size,
						    0,
						    (cl_event *) event,
						    &ret,
						    queue);
	assert(err == 0);

	/*
	 * Copying the real data.
	 */
	float *x = malloc(src_mf->nx * sizeof(float));
	float *y = malloc(src_mf->nx * sizeof(float));
	assert(x && y);

	int i;
	for (i = 0; i < src_mf->nx; i++)
	{
		struct point *p;
		p = (struct point *) src_mf->cpu_ptr;
		x[i] = p[i].x;
		y[i] = p[i].y;
	}

	ret = _opencl_malloc(context, xy, src_mf->nx*sizeof(*x), CL_MEM_READ_WRITE);
	assert(ret == CL_SUCCESS);
	ret = _opencl_malloc(context, xy+1, src_mf->nx*sizeof(*y), CL_MEM_READ_ONLY);
	assert(ret == CL_SUCCESS);

	err = _opencl_copy_ram_to_opencl_async_sync(x,
						    src_node,
						    xy[0],
						    dst_node,
						    src_mf->nx*sizeof(*x),
						    0,
						    NULL,
						    &ret,
						    queue);
	assert(err == CL_SUCCESS);
	err = _opencl_copy_ram_to_opencl_async_sync(y,
						    src_node,
						    xy[1],
						    dst_node,
						    src_mf->nx * sizeof(*y),
						    0,
						    NULL,
						    &ret,
						    queue);
	assert(err == CL_SUCCESS);


	struct struct_of_arrays *soa;
	soa = (struct struct_of_arrays *) dst_mf->opencl_ptr;
	soa->x = (void *) xy[0];
	soa->y = (void *) xy[1];

	/* Not needed anymore */
	free(x);
	free(y);
	return 0;

}

static int
copy_opencl_to_ram_async(void *src_interface, unsigned src_node,
			 void *dst_interface, unsigned dst_node,
			 void *event)
{
	FPRINTF(stderr, "Enter %s\n", __func__);
	struct starpu_multiformat_interface *src_mf;
	struct starpu_multiformat_interface *dst_mf;

	src_mf = (struct starpu_multiformat_interface *) src_interface;
	dst_mf = (struct starpu_multiformat_interface *) dst_interface;

	/*
	 * OpenCL stuff.
	 */
	int id = starpu_worker_get_id();
	int devid = starpu_worker_get_devid(id);
	cl_command_queue queue;
	starpu_opencl_get_queue(devid, &queue);
	cl_int ret;
	if (dst_mf->opencl_ptr == NULL)
	{
		dst_mf->opencl_ptr = malloc(sizeof(struct struct_of_arrays));
		assert(dst_mf->opencl_ptr);
	}

	float *x = malloc(src_mf->nx * sizeof(float));
	float *y = malloc(src_mf->nx * sizeof(float));
	assert(x && y);

	struct struct_of_arrays *soa;
	soa = (struct struct_of_arrays *) dst_mf->opencl_ptr;
	ret = _opencl_copy_opencl_to_ram(
		xy[0],
		src_node,
		x,
		dst_node,
		src_mf->nx * sizeof(float),
		0,
		NULL,
		queue);
	assert(ret == CL_SUCCESS);


	ret = _opencl_copy_opencl_to_ram(
		xy[1],
		src_node,
		y,
		dst_node,
		src_mf->nx * sizeof(float),
		0,
		NULL,
		queue);
	assert(ret == CL_SUCCESS);
	

	soa->x = x;
	soa->y = y;
	return 0;
}
#endif /* STARPU_USE_OPENCL */

 const struct starpu_data_copy_methods my_multiformat_copy_data_methods_s =
{
	.ram_to_ram = NULL,
	.ram_to_spu = NULL,
#ifdef STARPU_USE_CUDA
	.ram_to_cuda        = copy_ram_to_cuda,
	.cuda_to_ram        = copy_cuda_to_ram,
	.ram_to_cuda_async  = copy_ram_to_cuda_async,
	.cuda_to_ram_async  = copy_cuda_to_ram_async,
	.cuda_to_cuda       = NULL,
	.cuda_to_cuda_async = NULL,
#endif
#if STARPU_USE_OPENCL
	.ram_to_opencl       = copy_ram_to_opencl,
	.opencl_to_ram       = copy_opencl_to_ram,
	.opencl_to_opencl    = NULL,
        .ram_to_opencl_async = copy_ram_to_opencl_async,
	.opencl_to_ram_async = copy_opencl_to_ram_async,
#endif
	.cuda_to_spu = NULL,
	.spu_to_ram  = NULL,
	.spu_to_cuda = NULL,
	.spu_to_spu  = NULL
};
