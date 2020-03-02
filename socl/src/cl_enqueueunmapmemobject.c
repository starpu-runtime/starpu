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

cl_int command_unmap_mem_object_submit(command_unmap_mem_object cmd)
{
	/* Aliases */
	cl_mem buffer = cmd->buffer;

	static struct starpu_codelet codelet =
	{
		.name = "SOCL_UNMAP_MEM_OBJECT"
	};
	cpu_task_submit(cmd, (void(*)(void*))starpu_data_release, buffer->handle, 0, 1, &codelet, 0, NULL);

	return CL_SUCCESS;
}

CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY cl_int CL_API_CALL
soclEnqueueUnmapMemObject(cl_command_queue cq,
			  cl_mem            buffer,
			  void *            ptr,
			  cl_uint           num_events,
			  const cl_event *  events,
			  cl_event *        event)
{
	command_unmap_mem_object cmd = command_unmap_mem_object_create(buffer, ptr);

	cl_event ev = command_event_get(cmd);

	command_queue_enqueue(cq, cmd, num_events, events);

	RETURN_EVENT(ev, event);

	return CL_SUCCESS;
}
