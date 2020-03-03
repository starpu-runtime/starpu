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

static void mapbuffer_task(void *args)
{
	command_map_buffer cmd = (command_map_buffer)args;

	cl_event ev = command_event_get(cmd);
	ev->prof_start = _socl_nanotime();
	gc_entity_release(ev);

	enum starpu_data_access_mode mode = (cmd->map_flags == CL_MAP_READ ? STARPU_R : STARPU_RW);

	starpu_data_acquire_cb(cmd->buffer->handle, mode, command_completed_task_callback, cmd);
}

static struct starpu_codelet codelet_mapbuffer =
{
	.name = "SOCL_MAP_BUFFER"
};

cl_int command_map_buffer_submit(command_map_buffer cmd)
{
	gc_entity_retain(cmd);

	cpu_task_submit(cmd, mapbuffer_task, cmd, 0, 0, &codelet_mapbuffer, 0, NULL);

	return CL_SUCCESS;
}

CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY void * CL_API_CALL
soclEnqueueMapBuffer(cl_command_queue cq,
		     cl_mem           buffer,
		     cl_bool          blocking,
		     cl_map_flags     map_flags,
		     size_t           offset,
		     size_t           cb,
		     cl_uint          num_events,
		     const cl_event * events,
		     cl_event *       event,
		     cl_int *         errcode_ret)
{
	command_map_buffer cmd = command_map_buffer_create(buffer, map_flags, offset, cb);

	cl_event ev = command_event_get(cmd);

	command_queue_enqueue(cq, cmd, num_events, events);

	if (errcode_ret != NULL)
		*errcode_ret = CL_SUCCESS;

	MAY_BLOCK_THEN_RETURN_EVENT(ev,blocking,event);

	return (void*)(starpu_variable_get_local_ptr(buffer->handle) + offset);
}
