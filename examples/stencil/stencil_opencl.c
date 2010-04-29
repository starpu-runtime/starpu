/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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
#include <starpu_util.h>
#include "stencil.h"

void opencl_codelet(void *descr[], void *_args)
{
	float *data = (float *)STARPU_GET_VECTOR_PTR(descr[0]);
	float *results = (float *)STARPU_GET_VECTOR_PTR(descr[1]);
	float *C0 = (float *)STARPU_GET_VECTOR_PTR(descr[2]);
	float *C1 = (float *)STARPU_GET_VECTOR_PTR(descr[3]);

	cl_kernel kernel;
	cl_command_queue queue;
	int id, devid, err;

        id = starpu_get_worker_id();
        devid = starpu_get_worker_devid(id);

        err = starpu_opencl_load_kernel(&kernel, &queue, "examples/stencil/stencil_opencl_codelet.cl", "stencil", devid);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	err = 0;
	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &data);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &results);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &C0);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &C1);
	if (err) STARPU_OPENCL_REPORT_ERROR(err);

	{
		size_t global[3];
		size_t local[3];

                // Execute the kernel over the entire range of our 3d input data set
                local[0] = XBLOCK / X_PER_THREAD;   // threads along the X axis
                local[1] = YBLOCK / Y_PER_THREAD;   // threads along the Y axis
                local[2] = ZBLOCK / Z_PER_THREAD;   // threads along the Z axis

                global[0] = DIM / X_PER_THREAD;  // virtual size of global X axis
                global[1] = DIM / Y_PER_THREAD;  // virtual size of global Y axis
                global[2] = DIM / Z_PER_THREAD;  // virtual size of global Z axis

                err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 0, NULL, NULL);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	}

	clFinish(queue);

	starpu_opencl_release(kernel);
}
