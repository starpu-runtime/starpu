/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010,2011 University of Bordeaux
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

static cl_uint et_work_dim = 3;
static const size_t et_global_work_offset[3] = {0,0,0};
static const size_t et_global_work_size[3] = {1,1,1};
static const size_t * et_local_work_size = NULL;

CL_API_ENTRY cl_int CL_API_CALL
soclEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
soclEnqueueTask(cl_command_queue cq,
              cl_kernel         kernel,
              cl_uint           num_events,
              const cl_event *  events,
              cl_event *        event) CL_API_SUFFIX__VERSION_1_0
{
   return soclEnqueueNDRangeKernel(cq, kernel, et_work_dim, et_global_work_offset, et_global_work_size, et_local_work_size, num_events, events, event); 
}
