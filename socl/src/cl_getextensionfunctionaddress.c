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

#include <string.h>
#include "socl.h"
#include "init.h"

CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY void * CL_API_CALL
soclGetExtensionFunctionAddress(const char * func_name)
{
	if (func_name != NULL && strcmp(func_name, "clShutdown") == 0)
	{
		return (void*)soclShutdown;
	}

	return NULL;
}

CL_API_ENTRY void * CL_API_CALL
soclGetExtensionFunctionAddressForPlatform(cl_platform_id p, const char * func_name) CL_API_SUFFIX__VERSION_1_2
{
	if (p != &socl_platform)
		return NULL;

	return soclGetExtensionFunctionAddress(func_name);
}

CL_API_ENTRY void * CL_API_CALL clGetExtensionFunctionAddress(const char * func_name) CL_API_SUFFIX__VERSION_1_0
{
	if( func_name != NULL &&  strcmp("clIcdGetPlatformIDsKHR", func_name) == 0 )
		return (void *)soclIcdGetPlatformIDsKHR;
	return NULL;
}
