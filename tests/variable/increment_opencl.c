/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../helper.h"

/*
 * Queue an OpenCL kernel that just increments a variable
 */

struct starpu_opencl_program opencl_increment_program;
struct starpu_opencl_program opencl_redux_program;
struct starpu_opencl_program opencl_neutral_program;

void increment_load_opencl()
{
	int ret = starpu_opencl_load_opencl_from_file("tests/variable/increment_opencl_kernel.cl", &opencl_increment_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
	ret = starpu_opencl_load_opencl_from_file("tests/variable/redux_opencl_kernel.cl", &opencl_redux_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
	ret = starpu_opencl_load_opencl_from_file("tests/variable/neutral_opencl_kernel.cl", &opencl_neutral_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
}

void increment_unload_opencl()
{
	int ret = starpu_opencl_unload_opencl(&opencl_increment_program);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
	ret = starpu_opencl_unload_opencl(&opencl_redux_program);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
	ret = starpu_opencl_unload_opencl(&opencl_neutral_program);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
}

void increment_opencl(void *buffers[], void *args)
{
	(void) args;
	int id, devid;
	cl_int err;
	cl_kernel kernel;
	cl_command_queue queue;

	cl_mem val = (cl_mem)STARPU_VARIABLE_GET_PTR(buffers[0]);

	id = starpu_worker_get_id_check();
	devid = starpu_worker_get_devid(id);

	err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_increment_program, "_increment_opencl", devid);
	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);

	err = clSetKernelArg(kernel, 0, sizeof(val), &val);
	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);

	{
		size_t global=1;
		size_t local=1;

		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
		if (err != CL_SUCCESS)
			STARPU_OPENCL_REPORT_ERROR(err);
	}
	starpu_opencl_release_kernel(kernel);
}

void redux_opencl(void *buffers[], void *args)
{
	(void) args;
	int id, devid;
	cl_int err;
	cl_kernel kernel;
	cl_command_queue queue;

	cl_mem dst = (cl_mem)STARPU_VARIABLE_GET_PTR(buffers[0]);
	cl_mem src = (cl_mem)STARPU_VARIABLE_GET_PTR(buffers[1]);

	id = starpu_worker_get_id_check();
	devid = starpu_worker_get_devid(id);

	err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_redux_program, "_redux_opencl", devid);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	err = clSetKernelArg(kernel, 0, sizeof(dst), &dst);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	err = clSetKernelArg(kernel, 1, sizeof(src), &src);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	{
		size_t global=1;
		size_t local=1;

		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	}
	starpu_opencl_release_kernel(kernel);
}

void neutral_opencl(void *buffers[], void *args)
{
	(void) args;
	int id, devid;
	cl_int err;
	cl_kernel kernel;
	cl_command_queue queue;

	cl_mem dst = (cl_mem)STARPU_VARIABLE_GET_PTR(buffers[0]);

	id = starpu_worker_get_id_check();
	devid = starpu_worker_get_devid(id);

	err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_neutral_program, "_neutral_opencl", devid);
	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);

	err = clSetKernelArg(kernel, 0, sizeof(dst), &dst);
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	{
		size_t global=1;
		size_t local=1;

		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	}
	starpu_opencl_release_kernel(kernel);
}
