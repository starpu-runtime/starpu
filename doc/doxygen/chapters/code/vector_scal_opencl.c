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

//! [To be included. You should update doxygen if you see this text.]
#include <starpu.h>

extern struct starpu_opencl_program programs;

void scal_opencl_func(void *buffers[], void *_args)
{
    float *factor = _args;
    int id, devid, err;                   /* OpenCL specific code */
    cl_kernel kernel;                     /* OpenCL specific code */
    cl_command_queue queue;               /* OpenCL specific code */
    cl_event event;                       /* OpenCL specific code */

    /* length of the vector */
    unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
    /* OpenCL copy of the vector pointer */
    cl_mem val = (cl_mem)STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);

    {  /* OpenCL specific code */
	 id = starpu_worker_get_id();
	 devid = starpu_worker_get_devid(id);

	 err = starpu_opencl_load_kernel(&kernel, &queue, &programs,
					 "vector_mult_opencl", /* Name of the codelet */
					 devid);
	 if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	 err = clSetKernelArg(kernel, 0, sizeof(n), &n);
	 err |= clSetKernelArg(kernel, 1, sizeof(val), &val);
	 err |= clSetKernelArg(kernel, 2, sizeof(*factor), factor);
	 if (err) STARPU_OPENCL_REPORT_ERROR(err);
    }

    {   /* OpenCL specific code */
        size_t global=n;
        size_t local;
        size_t s;
        cl_device_id device;

        starpu_opencl_get_device(devid, &device);
        err = clGetKernelWorkGroupInfo (kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, &s);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
        if (local > global) local=global;
        else global = (global + local-1) / local * local;

        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, &event);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
    }

    {  /* OpenCL specific code */
	 clFinish(queue);
	 starpu_opencl_collect_stats(event);
	 clReleaseEvent(event);

	 starpu_opencl_release_kernel(kernel);
    }
}
//! [To be included. You should update doxygen if you see this text.]
