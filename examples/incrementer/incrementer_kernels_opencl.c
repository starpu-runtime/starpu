/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* OpenCL codelet for incrementation */

#include <starpu.h>

extern struct starpu_opencl_program opencl_program;
void opencl_codelet(void *descr[], void *_args)
{
	(void)_args;
	cl_mem val = (cl_mem)STARPU_VECTOR_GET_DEV_HANDLE(descr[0]);
	cl_kernel kernel;
	cl_command_queue queue;
	int id, devid, err;

        id = starpu_worker_get_id_check();
        devid = starpu_worker_get_devid(id);

	err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_program, "incrementer", devid);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	err = clSetKernelArg(kernel, 0, sizeof(val), &val);
	if (err) STARPU_OPENCL_REPORT_ERROR(err);

	{
		size_t global=4;
		size_t local, s;
                cl_device_id device;

                starpu_opencl_get_device(devid, &device);
                err = clGetKernelWorkGroupInfo (kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, &s);
                if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
                if (local > global) local=global;

		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	}

	starpu_opencl_release_kernel(kernel);
}
