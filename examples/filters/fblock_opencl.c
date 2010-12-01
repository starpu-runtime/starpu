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
#include <starpu_opencl.h>

extern struct starpu_opencl_program opencl_program;

void opencl_func(void *buffers[], void *cl_arg)
{
	int id, devid, err;
	cl_kernel kernel;
	cl_command_queue queue;

        int *factor = cl_arg;
	int *block = (int *)STARPU_BLOCK_GET_PTR(buffers[0]);
	int nx = (int)STARPU_BLOCK_GET_NX(buffers[0]);
	int ny = (int)STARPU_BLOCK_GET_NY(buffers[0]);
	int nz = (int)STARPU_BLOCK_GET_NZ(buffers[0]);
        unsigned ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
        unsigned ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);

	id = starpu_worker_get_id();
	devid = starpu_worker_get_devid(id);

	err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_program, "fblock_opencl", devid);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	err = 0;
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &block);
	err = clSetKernelArg(kernel, 1, sizeof(nx), &nx);
	err = clSetKernelArg(kernel, 2, sizeof(ny), &ny);
	err = clSetKernelArg(kernel, 3, sizeof(nz), &nz);
	err = clSetKernelArg(kernel, 4, sizeof(ldy), &ldy);
	err = clSetKernelArg(kernel, 5, sizeof(ldz), &ldz);
	err |= clSetKernelArg(kernel, 6, sizeof(*factor), factor);
	if (err) STARPU_OPENCL_REPORT_ERROR(err);

	{
		size_t global=nx*ny*nz;
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	}

	clFinish(queue);

	starpu_opencl_release_kernel(kernel);
}

