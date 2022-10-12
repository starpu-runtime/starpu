/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define KERNEL_LOCATION "tests/datawizard/interfaces/ndim/ndim_opencl_kernel.cl"
extern struct test_config arr4d_config;
static struct starpu_opencl_program opencl_program;

void
test_arr4d_opencl_func(void *buffers[], void *args)
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

	int *nn = (int *)STARPU_NDIM_GET_NN(buffers[0]);
    unsigned *ldn = STARPU_NDIM_GET_LDN(buffers[0]);
    int nx = nn[0];
    int ny = nn[1];
    int nz = nn[2];
    int nt = nn[3];
    unsigned ldy = ldn[1];
    unsigned ldz = ldn[2];
    unsigned ldt = ldn[3];
	cl_mem arr4d = (cl_mem) STARPU_NDIM_GET_DEV_HANDLE(buffers[0]);

	cl_context context;
	id = starpu_worker_get_id_check();
	devid = starpu_worker_get_devid(id);
	starpu_opencl_get_context(devid, &context);

	cl_mem fail = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
		sizeof(int), &arr4d_config.copy_failed, &err);

	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);


	err = starpu_opencl_load_kernel(&kernel,
					&queue,
					&opencl_program,
					"arr4d_opencl",
					devid);
	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);

	int nargs;
	nargs = starpu_opencl_set_kernel_args(&err, &kernel,
					      sizeof(arr4d), &arr4d,
					      sizeof(nx), &nx,
					      sizeof(ny), &ny,
					      sizeof(nz), &nz,
					      sizeof(nt), &nt,
					      sizeof(ldy), &ldy,
					      sizeof(ldz), &ldz,
					      sizeof(ldt), &ldt,
					      sizeof(factor), &factor,
					      sizeof(fail), &fail,
					      0);

	if (nargs != 10)
	{
		fprintf(stderr, "Failed to set argument #%d\n", nargs);
		STARPU_OPENCL_REPORT_ERROR(err);
	}

	{
		size_t global[3] = {nx, ny, nz*nt};
		err = clEnqueueNDRangeKernel(queue,
					     kernel,
					     3,
					     NULL,
					     global,
					     NULL,
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
				  &arr4d_config.copy_failed,
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
}
