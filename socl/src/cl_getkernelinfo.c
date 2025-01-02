/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "getinfo.h"

CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY cl_int CL_API_CALL
soclGetKernelInfo(cl_kernel       kernel,
		  cl_kernel_info  param_name,
		  size_t          param_value_size,
		  void *          param_value,
		  size_t *        param_value_size_ret)
{
	if (kernel == NULL)
		return CL_INVALID_KERNEL;

	switch (param_name)
	{
		INFO_CASE_EX(CL_KERNEL_FUNCTION_NAME, kernel->kernel_name, strlen(kernel->kernel_name)+1);
		INFO_CASE(CL_KERNEL_NUM_ARGS, kernel->num_args);
		INFO_CASE(CL_KERNEL_REFERENCE_COUNT, kernel->_entity.refs);
		INFO_CASE(CL_KERNEL_PROGRAM, kernel->program);
		INFO_CASE(CL_KERNEL_CONTEXT, kernel->program->context);
	default:
		return CL_INVALID_VALUE;
	}

	return CL_SUCCESS;
}
