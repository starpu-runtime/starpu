/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2012 University of Bordeaux
 * Copyright (C) 2012 CNRS
 * Copyright (C) 2012 Vincent Danjean <Vincent.Danjean@ens-lyon.org>
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

CL_API_ENTRY void * CL_API_CALL
soclGetExtensionFunctionAddress(const char * UNUSED(func_name)) CL_API_SUFFIX__VERSION_1_0
{
   //TODO
   return NULL;
}

CL_API_ENTRY void * CL_API_CALL clGetExtensionFunctionAddress(
             const char *   func_name) CL_API_SUFFIX__VERSION_1_0 {
  if( func_name != NULL &&  strcmp("clIcdGetPlatformIDsKHR", func_name) == 0 )
    return (void *)soclIcdGetPlatformIDsKHR;
  return NULL;
}
