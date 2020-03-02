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
soclEnqueueMarker(cl_command_queue  cq,
                cl_event *          event)
{
	if (event == NULL)
		return CL_INVALID_VALUE;

	command_marker cmd = command_marker_create();

	cl_event ev = command_event_get(cmd);

	command_queue_enqueue(cq, cmd, 0, NULL);

	RETURN_EVENT(ev, event);

	return CL_SUCCESS;
}

cl_int command_marker_submit(command_marker cmd)
{
	struct starpu_task *task;
	task = task_create(CL_COMMAND_MARKER);

	return task_submit(task, cmd);
}
