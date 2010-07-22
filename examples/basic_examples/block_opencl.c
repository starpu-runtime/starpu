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
#include <starpu_opencl.h>

extern struct starpu_opencl_program opencl_code;

void opencl_codelet(void *descr[], void *_args)
{
	cl_kernel kernel;
	cl_command_queue queue;
	int id, devid, err, n;
	float *block = (float *)STARPU_BLOCK_GET_PTR(descr[0]);
	int nx = (int)STARPU_BLOCK_GET_NX(descr[0]);
	int ny = (int)STARPU_BLOCK_GET_NY(descr[0]);
	int nz = (int)STARPU_BLOCK_GET_NZ(descr[0]);
        unsigned ldy = STARPU_BLOCK_GET_LDY(descr[0]);
        unsigned ldz = STARPU_BLOCK_GET_LDZ(descr[0]);
        float *multiplier = (float *)_args;

        id = starpu_worker_get_id();
        devid = starpu_worker_get_devid(id);

        err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_code, "block", devid);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	err = 0;
        n=0;
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &block);
	err = clSetKernelArg(kernel, 1, sizeof(int), &nx);
	err = clSetKernelArg(kernel, 2, sizeof(int), &ny);
	err = clSetKernelArg(kernel, 3, sizeof(int), &nz);
	err = clSetKernelArg(kernel, 4, sizeof(ldy), &ldy);
	err = clSetKernelArg(kernel, 5, sizeof(ldz), &ldz);
	err = clSetKernelArg(kernel, 6, sizeof(float), multiplier);
        if (err) STARPU_OPENCL_REPORT_ERROR(err);

	{
                size_t global=nx*ny*nz;
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	}

	clFinish(queue);

        starpu_opencl_release_kernel(kernel);
}

