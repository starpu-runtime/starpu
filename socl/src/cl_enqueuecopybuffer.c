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

static void soclEnqueueCopyBuffer_opencl_task(void *descr[], void *args)
{
	int wid;
	cl_command_queue cq;
	cl_event ev;
	command_copy_buffer cmd = (command_copy_buffer)args;

	cl_event event = command_event_get(cmd);
	event->prof_start = _socl_nanotime();
	gc_entity_release(event);

	wid = starpu_worker_get_id_check();
	starpu_opencl_get_queue(wid, &cq);

	cl_mem src = (cl_mem)STARPU_VARIABLE_GET_PTR(descr[0]);
	cl_mem dst = (cl_mem)STARPU_VARIABLE_GET_PTR(descr[1]);

	clEnqueueCopyBuffer(cq, src,dst, cmd->src_offset, cmd->dst_offset, cmd->cb, 0, NULL, &ev);
	clWaitForEvents(1, &ev);
	clReleaseEvent(ev);

	gc_entity_release_cmd(cmd);
}

static void soclEnqueueCopyBuffer_cpu_task(void *descr[], void *args)
{
	command_copy_buffer cmd = (command_copy_buffer)args;

	cl_event ev = command_event_get(cmd);
	ev->prof_start = _socl_nanotime();
	gc_entity_release(ev);

	char * src = (void*)STARPU_VARIABLE_GET_PTR(descr[0]);
	char * dst = (void*)STARPU_VARIABLE_GET_PTR(descr[1]);

	memcpy(dst+cmd->dst_offset, src+cmd->src_offset, cmd->cb);

	gc_entity_release_cmd(cmd);
}

static struct starpu_perfmodel copy_buffer_perfmodel =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "SOCL_COPY_BUFFER"
};

static struct starpu_codelet codelet_copybuffer =
{
	.where = STARPU_CPU | STARPU_OPENCL,
	.model = &copy_buffer_perfmodel,
	.cpu_funcs = { &soclEnqueueCopyBuffer_cpu_task },
	.opencl_funcs = { &soclEnqueueCopyBuffer_opencl_task },
	.modes = {STARPU_R, STARPU_RW},
	.nbuffers = 2
};

cl_int command_copy_buffer_submit(command_copy_buffer cmd)
{
	struct starpu_task * task = task_create(CL_COMMAND_COPY_BUFFER);

	task->handles[0] = cmd->src_buffer->handle;
	task->handles[1] = cmd->dst_buffer->handle;
	task->cl = &codelet_copybuffer;

	/* Execute the task on a specific worker? */
	if (cmd->_command.event->cq->device != NULL)
	{
		task->execute_on_a_specific_worker = 1;
		task->workerid = cmd->_command.event->cq->device->worker_id;
	}

	gc_entity_store_cmd(&task->cl_arg, cmd);
	task->cl_arg_size = sizeof(*cmd);

	cmd->dst_buffer->scratch = 0;

	task_submit(task, cmd);

	return CL_SUCCESS;
}

CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY cl_int CL_API_CALL
soclEnqueueCopyBuffer(cl_command_queue  cq,
		      cl_mem              src_buffer,
		      cl_mem              dst_buffer,
		      size_t              src_offset,
		      size_t              dst_offset,
		      size_t              cb,
		      cl_uint             num_events,
		      const cl_event *    events,
		      cl_event *          event)
{
	command_copy_buffer cmd = command_copy_buffer_create(src_buffer, dst_buffer, src_offset, dst_offset, cb);

	cl_event ev = command_event_get(cmd);

	command_queue_enqueue(cq, cmd, num_events, events);

	RETURN_EVENT(ev, event);

	return CL_SUCCESS;
}
