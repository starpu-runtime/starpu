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
#include "custom_types.h"
#include "custom_interface.h"

extern struct starpu_opencl_program _opencl_conversion_program;

void cpu_to_opencl_opencl_func(void *buffers[], void *args)
{
	(void) args;
	int id, devid;
	cl_int err;
	cl_kernel kernel;
	cl_command_queue queue;

	unsigned n = CUSTOM_GET_NX(buffers[0]);
	n*=2;
	struct point *aop;
	aop = (struct point *) CUSTOM_GET_CPU_PTR(buffers[0]);

	id = starpu_worker_get_id_check();
	devid = starpu_worker_get_devid(id);

	err = starpu_opencl_load_kernel(&kernel,
					&queue,
					&_opencl_conversion_program,
					"custom_opencl_conversion",
					devid);
	if (err != CL_SUCCESS)
		STARPU_OPENCL_REPORT_ERROR(err);


	void *x = CUSTOM_GET_OPENCL_X_PTR(buffers[0]);
	if (starpu_opencl_set_kernel_args(&err, &kernel,
					  sizeof(aop), &aop,
					  sizeof(x), &x,
					  sizeof(n), &n,
					  0) != 3)
	{
		STARPU_OPENCL_REPORT_ERROR(err);
		assert(0);
	}


	{
		size_t global=n;
		size_t local;
		size_t s;
		cl_device_id device;

		starpu_opencl_get_device(devid, &device);

		err = clGetKernelWorkGroupInfo (kernel,
						device,
						CL_KERNEL_WORK_GROUP_SIZE,
						sizeof(local),
						&local,
						&s);
		if (err != CL_SUCCESS)
			STARPU_OPENCL_REPORT_ERROR(err);

		if (local > global)
			local = global;
		else
			global = (global + local-1) / local * local;

		err = clEnqueueNDRangeKernel(
				queue,
				kernel,
				1,	 /* work_dim */
				NULL,	 /* global_work_offset */
				&global, /* global_work_size */
				&local,	 /* local_work_size */
				0,	 /* num_events_in_wait_list */
				NULL,	 /* event_wait_list */
				NULL);

		if (err != CL_SUCCESS)
			STARPU_OPENCL_REPORT_ERROR(err);
	}
	starpu_opencl_release_kernel(kernel);
}
