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

/*
 * This example complements vector_scale.c: here we implement a OpenCL version.
 */

#include <starpu.h>
#include <starpu_opencl.h>

void scal_opencl_func(void *buffers[], void *_args)
{
	float *factor = (float *)_args;
	struct starpu_vector_interface_s *vector = (struct starpu_vector_interface_s *) buffers[0];
	int id, devid, err;
	cl_kernel kernel;
	cl_command_queue queue;

	/* length of the vector */
	unsigned n = STARPU_GET_VECTOR_NX(vector);
	/* local copy of the vector pointer */
	float *val = (float *)STARPU_GET_VECTOR_PTR(vector);

	id = starpu_worker_get_id();
	devid = starpu_worker_get_devid(id);

	err = starpu_opencl_load_kernel(&kernel, &queue,
					"examples/basic_examples/vector_scal_opencl_codelet.cl",
					"vector_mult_opencl", devid);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	err = 0;
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &val);
	err = clSetKernelArg(kernel, 1, sizeof(int), &n);
	err |= clSetKernelArg(kernel, 2, sizeof(float), (void*)factor);
	if (err) STARPU_OPENCL_REPORT_ERROR(err);

	{
		size_t global=1;
		size_t local=1;
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	}

	clFinish(queue);

	starpu_opencl_release(kernel);
}
