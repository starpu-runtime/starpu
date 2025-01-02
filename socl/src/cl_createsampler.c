/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
CL_API_ENTRY cl_sampler CL_API_CALL
soclCreateSampler(cl_context          UNUSED(context),
		  cl_bool             UNUSED(normalized_coords),
		  cl_addressing_mode  UNUSED(addressing_mode),
		  cl_filter_mode      UNUSED(filter_mode),
		  cl_int *            errcode_ret)
{
	if (errcode_ret != NULL)
		*errcode_ret = CL_INVALID_OPERATION;
	return NULL;
}
