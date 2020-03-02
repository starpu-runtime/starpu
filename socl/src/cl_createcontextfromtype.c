/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2012       Vincent Danjean
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
#include "init.h"

CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY cl_context CL_API_CALL
soclCreateContextFromType(const cl_context_properties * properties,
			  cl_device_type                device_type,
			  void (*pfn_notify)(const char *, const void *, size_t, void *),
			  void *                        user_data,
			  cl_int *                      errcode_ret)
{
	if (socl_init_starpu() < 0)
		return NULL;

	//TODO: appropriate error messages

	cl_uint num_devices;
	soclGetDeviceIDs(&socl_platform, device_type, 0, NULL, &num_devices);

	cl_device_id devices[num_devices];
	soclGetDeviceIDs(&socl_platform, device_type, num_devices, devices, NULL);

	return soclCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}
