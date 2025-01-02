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

#include <starpu.h>
#include "complex_dev_handle_interface.h"

extern struct starpu_opencl_program opencl_program;

void copy_complex_dev_handle_codelet_opencl(void *buffers[], void *_args)
{
	(void) _args;

	int id, devid;
	cl_int err;
	cl_kernel kernel;
	cl_command_queue queue;

	/* length of the vector */
	unsigned n = STARPU_COMPLEX_DEV_HANDLE_GET_NX(buffers[0]);
	/* OpenCL copy of the vector pointer */
	cl_mem i_real			= (cl_mem) STARPU_COMPLEX_DEV_HANDLE_GET_DEV_HANDLE_REAL(buffers[0]);
	unsigned i_real_offset		= STARPU_COMPLEX_DEV_HANDLE_GET_OFFSET_REAL(buffers[0]);
	cl_mem i_imaginary		= (cl_mem) STARPU_COMPLEX_DEV_HANDLE_GET_DEV_HANDLE_IMAGINARY(buffers[0]);
	unsigned i_imaginary_offset	= STARPU_COMPLEX_DEV_HANDLE_GET_OFFSET_IMAGINARY(buffers[0]);
	cl_mem o_real			= (cl_mem) STARPU_COMPLEX_DEV_HANDLE_GET_DEV_HANDLE_REAL(buffers[1]);
	unsigned o_real_offset		= STARPU_COMPLEX_DEV_HANDLE_GET_OFFSET_REAL(buffers[1]);
	cl_mem o_imaginary		= (cl_mem) STARPU_COMPLEX_DEV_HANDLE_GET_DEV_HANDLE_IMAGINARY(buffers[1]);
	unsigned o_imaginary_offset	= STARPU_COMPLEX_DEV_HANDLE_GET_OFFSET_IMAGINARY(buffers[1]);

	id = starpu_worker_get_id_check();
	devid = starpu_worker_get_devid(id);

	err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_program, "complex_copy_opencl", devid);
	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);

	err = clSetKernelArg(kernel, 0, sizeof(o_real), &o_real);
	err|= clSetKernelArg(kernel, 1, sizeof(o_real_offset), &o_real_offset);
	err|= clSetKernelArg(kernel, 2, sizeof(o_imaginary), &o_imaginary);
	err|= clSetKernelArg(kernel, 3, sizeof(o_imaginary_offset), &o_imaginary_offset);
	err|= clSetKernelArg(kernel, 4, sizeof(i_real), &i_real);
	err|= clSetKernelArg(kernel, 5, sizeof(i_real_offset), &i_real_offset);
	err|= clSetKernelArg(kernel, 6, sizeof(i_imaginary), &i_imaginary);
	err|= clSetKernelArg(kernel, 7, sizeof(i_imaginary_offset), &i_imaginary_offset);
	err|= clSetKernelArg(kernel, 8, sizeof(n), &n);
	if (err)
		STARPU_OPENCL_REPORT_ERROR(err);

	{
		size_t global=n;
		size_t local;
		size_t s;
		cl_device_id device;

		starpu_opencl_get_device(devid, &device);

		err = clGetKernelWorkGroupInfo (kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, &s);
		if (err != CL_SUCCESS)
			STARPU_OPENCL_REPORT_ERROR(err);
		if (local > global)
			local=global;
		else
			global = (global + local-1) / local * local;

		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
		if (err != CL_SUCCESS)
			STARPU_OPENCL_REPORT_ERROR(err);
	}
	starpu_opencl_release_kernel(kernel);
}
