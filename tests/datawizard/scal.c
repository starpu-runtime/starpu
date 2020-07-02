/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "scal.h"
#include "helper.h"

/*
 * Implement a kernel that just multiplies a vector by 2
 */

void scal_func_cpu(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
	unsigned i;

	struct starpu_vector_interface *vector = (struct starpu_vector_interface *) buffers[0];
	unsigned *val = (unsigned *) STARPU_VECTOR_GET_PTR(vector);
	unsigned n = STARPU_VECTOR_GET_NX(vector);

	/* scale the vector */
	for (i = 0; i < n; i++)
		val[i] *= 2;
}

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program opencl_program;

void scal_func_opencl(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
	int id, devid;
        cl_int err;
	cl_kernel kernel;
	cl_command_queue queue;

	unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
	cl_mem val = (cl_mem)STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);
	unsigned offset = STARPU_VECTOR_GET_OFFSET(buffers[0]);

	id = starpu_worker_get_id_check();
	devid = starpu_worker_get_devid(id);

	err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_program, "vector_mult_opencl", devid);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	err = clSetKernelArg(kernel, 0, sizeof(val), &val);
	err |= clSetKernelArg(kernel, 1, sizeof(offset), &offset);
	err |= clSetKernelArg(kernel, 2, sizeof(n), &n);
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
#endif

struct starpu_codelet scal_codelet =
{
		
	.cpu_funcs = { scal_func_cpu },
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = { scal_func_opencl },
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
#ifdef STARPU_USE_CUDA
	.cuda_funcs = { scal_func_cuda },
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.cpu_funcs_name = {"scal_func_cpu"},
	.modes = { STARPU_RW },
        .model = NULL,
        .nbuffers = 1
};

