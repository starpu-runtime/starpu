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

CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY cl_int CL_API_CALL
soclEnqueueNativeKernel(cl_command_queue  UNUSED(command_queue),
			__attribute__((unused)) void (*user_func)(void *),
			void *            UNUSED(args),
			size_t            UNUSED(cb_args),
			cl_uint           UNUSED(num_mem_objects),
			const cl_mem *    UNUSED(mem_list),
			const void **     UNUSED(args_mem_loc),
			cl_uint           UNUSED(num_events_in_wait_list),
			const cl_event *  UNUSED(event_wait_list),
			cl_event *        UNUSED(event))
{
	return CL_INVALID_OPERATION;
}
