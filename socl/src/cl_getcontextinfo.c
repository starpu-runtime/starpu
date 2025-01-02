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
soclGetContextInfo(cl_context       context,
		   cl_context_info    param_name,
		   size_t             param_value_size,
		   void *             param_value,
		   size_t *           param_value_size_ret)
{
	if (context == NULL)
		return CL_INVALID_CONTEXT;

	switch (param_name)
	{
		INFO_CASE(CL_CONTEXT_REFERENCE_COUNT, context->_entity.refs);
		INFO_CASE_EX(CL_CONTEXT_DEVICES, context->devices, context->num_devices * sizeof(cl_device_id));
		INFO_CASE_EX(CL_CONTEXT_PROPERTIES, context->properties, context->num_properties * sizeof(cl_context_properties));
	default:
		return CL_INVALID_VALUE;
	}

	return CL_SUCCESS;
}
