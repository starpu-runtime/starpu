/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This example complements vector_scal.c: here we implement a OpenCL version.
 */

#include <starpu.h>

extern struct starpu_opencl_program opencl_program;

void scal_opencl_func(void *buffers[], void *_args)
{
	float *factor = _args;
	int id, devid;
	cl_int err;
	cl_kernel kernel;
	cl_command_queue queue;

	/* length of the vector */
	unsigned int n = STARPU_VECTOR_GET_NX(buffers[0]);
	/* OpenCL copy of the vector pointer */
	cl_mem val = (cl_mem)STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);

	id = starpu_worker_get_id_check();
	devid = starpu_worker_get_devid(id);

	err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_program, "vector_mult_opencl", devid);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	err = clSetKernelArg(kernel, 0, sizeof(n), &n);
	err |= clSetKernelArg(kernel, 1, sizeof(val), &val);
	err |= clSetKernelArg(kernel, 2, sizeof(*factor), factor);
	if (err) STARPU_OPENCL_REPORT_ERROR(err);

	{
		size_t global=n;
		size_t local;
		size_t s;
		cl_device_id device;

		starpu_opencl_get_device(devid, &device);

		err = clGetKernelWorkGroupInfo (kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, &s);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
		if (local > global) local=global;
		else global = (global + local-1) / local * local;

		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	}
	starpu_opencl_release_kernel(kernel);
}
