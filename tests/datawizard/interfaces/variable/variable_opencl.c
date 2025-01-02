/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../test_interfaces.h"

#define KERNEL_LOCATION "tests/datawizard/interfaces/variable/variable_opencl_kernel.cl"

extern struct test_config variable_config;
static struct starpu_opencl_program opencl_program;

void test_variable_opencl_func(void *buffers[], void *args)
{
	STARPU_SKIP_IF_VALGRIND;

	int id, devid, ret;
	int factor = *(int *) args;

        cl_int             err;
	cl_kernel          kernel;
	cl_command_queue   queue;
	cl_event           event;

	ret = starpu_opencl_load_opencl_from_file(KERNEL_LOCATION, &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");

	cl_mem val = (cl_mem)STARPU_VARIABLE_GET_PTR(buffers[0]);

	cl_context context;
	id = starpu_worker_get_id_check();
	devid = starpu_worker_get_devid(id);
	starpu_opencl_get_context(devid, &context);

	cl_mem fail = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
		sizeof(int), &variable_config.copy_failed, &err);

	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);


	err = starpu_opencl_load_kernel(&kernel,
					&queue,
					&opencl_program,
					"variable_opencl",
					devid);
	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);

	err  = clSetKernelArg(kernel, 0, sizeof(val), &val);
	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);

	err = clSetKernelArg(kernel, 1, sizeof(fail), &fail);
	if (err)
		STARPU_OPENCL_REPORT_ERROR(err);

	err = clSetKernelArg(kernel, 2, sizeof(factor), &factor);
	if (err)
		STARPU_OPENCL_REPORT_ERROR(err);

	{
		size_t global = 1;
                size_t local = 1;
                cl_device_id device;

                starpu_opencl_get_device(devid, &device);

		err = clEnqueueNDRangeKernel(queue,
					kernel,
					1,
					NULL,
					&global,
					&local,
					0,
					NULL,
					&event);

		if (err != CL_SUCCESS)
			STARPU_OPENCL_REPORT_ERROR(err);
	}

	err = clEnqueueReadBuffer(queue,
				  fail,
				  CL_TRUE,
				  0,
				  sizeof(int),
				  &variable_config.copy_failed,
				  0,
				  NULL,
				  NULL);
	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);

	clFinish(queue);
	starpu_opencl_collect_stats(event);
	clReleaseEvent(event);

	starpu_opencl_release_kernel(kernel);
        ret = starpu_opencl_unload_opencl(&opencl_program);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
	return;
}
