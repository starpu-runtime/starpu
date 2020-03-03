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

#include "socl.h"

CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY cl_int CL_API_CALL
soclGetKernelWorkGroupInfo(cl_kernel                kernel,
			   cl_device_id               device,
			   cl_kernel_work_group_info  param_name,
			   size_t                     param_value_size,
			   void *                     param_value,
			   size_t *                   param_value_size_ret)
{
	int range = starpu_worker_get_range_by_id(device->worker_id);
	cl_device_id dev;
	starpu_opencl_get_device(device->device_id, &dev);

	return clGetKernelWorkGroupInfo(kernel->cl_kernels[range], dev,
					param_name, param_value_size, param_value, param_value_size_ret);
}
