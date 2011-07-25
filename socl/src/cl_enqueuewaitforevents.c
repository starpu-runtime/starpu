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

#include "socl.h"

CL_API_ENTRY cl_int CL_API_CALL
soclEnqueueWaitForEvents(cl_command_queue cq,
                       cl_uint          num_events,
                       const cl_event * events) CL_API_SUFFIX__VERSION_1_0
{

	cl_int ndeps;
	cl_event *deps;

	//CL_COMMAND_MARKER has been chosen as CL_COMMAND_WAIT_FOR_EVENTS doesn't exist
	starpu_task * task = task_create(CL_COMMAND_MARKER);

	DEBUG_MSG("Submitting WAIT_FOR_EVENTS task (event %d)\n", task->tag_id);
	command_queue_enqueue(cq, task_event(task), 1, num_events, events, &ndeps, &deps);

	task_submit(task, ndeps, deps);

	return CL_SUCCESS;
}
