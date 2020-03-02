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

static void soclEnqueueReadBuffer_cpu_task(void *descr[], void *args)
{
	command_read_buffer cmd = (command_read_buffer)args;

	cl_event ev = command_event_get(cmd);
	ev->prof_start = _socl_nanotime();
	gc_entity_release(ev);

	char * ptr = (void*)STARPU_VARIABLE_GET_PTR(descr[0]);
	DEBUG_MSG("[Buffer %d] Reading %ld bytes from %p to %p\n", cmd->buffer->id, (long)cmd->cb, ptr+cmd->offset, cmd->ptr);

	//This fix is for people who use USE_HOST_PTR and still use ReadBuffer to sync the buffer in host mem at host_ptr.
	//They should use buffer mapping facilities instead.
	if (ptr+cmd->offset != cmd->ptr)
		memcpy(cmd->ptr, ptr+cmd->offset, cmd->cb);

	gc_entity_release_cmd(cmd);
}

static void soclEnqueueReadBuffer_opencl_task(void *descr[], void *args)
{
	command_read_buffer cmd = (command_read_buffer)args;

	cl_event event = command_event_get(cmd);
	event->prof_start = _socl_nanotime();
	gc_entity_release(event);

	cl_mem mem = (cl_mem)STARPU_VARIABLE_GET_PTR(descr[0]);

	DEBUG_MSG("[Buffer %d] Reading %ld bytes from offset %ld into %p\n", cmd->buffer->id, (long)cmd->cb, (long)cmd->offset, cmd->ptr);

	int wid = starpu_worker_get_id_check();
	cl_command_queue cq;
	starpu_opencl_get_queue(wid, &cq);

	cl_event ev;
	cl_int ret = clEnqueueReadBuffer(cq, mem, CL_TRUE, cmd->offset, cmd->cb, cmd->ptr, 0, NULL, &ev);
	if (ret != CL_SUCCESS)
		ERROR_CL("clEnqueueReadBuffer", ret);

	clWaitForEvents(1, &ev);
	clReleaseEvent(ev);

	gc_entity_release_cmd(cmd);
}

static struct starpu_perfmodel read_buffer_perfmodel =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "SOCL_READ_BUFFER"
};

static struct starpu_codelet codelet_readbuffer =
{
	.where = STARPU_OPENCL,
	.model = &read_buffer_perfmodel,
	.cpu_funcs = { &soclEnqueueReadBuffer_cpu_task },
	.opencl_funcs = { &soclEnqueueReadBuffer_opencl_task },
	.modes = {STARPU_R},
	.nbuffers = 1
};

cl_int command_read_buffer_submit(command_read_buffer cmd)
{
	struct starpu_task * task = task_create(CL_COMMAND_READ_BUFFER);

	task->handles[0] = cmd->buffer->handle;
	task->cl = &codelet_readbuffer;

	/* Execute the task on a specific worker? */
	if (cmd->_command.event->cq->device != NULL)
	{
		task->execute_on_a_specific_worker = 1;
		task->workerid = cmd->_command.event->cq->device->worker_id;
	}

	gc_entity_store_cmd(&task->cl_arg, cmd);
	task->cl_arg_size = sizeof(*cmd);

	task_submit(task, cmd);

	return CL_SUCCESS;
}

CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY cl_int CL_API_CALL
soclEnqueueReadBuffer(cl_command_queue  cq,
		      cl_mem              buffer,
		      cl_bool             blocking,
		      size_t              offset,
		      size_t              cb,
		      void *              ptr,
		      cl_uint             num_events,
		      const cl_event *    events,
		      cl_event *          event)
{
	command_read_buffer cmd = command_read_buffer_create(buffer, offset, cb, ptr);

	cl_event ev = command_event_get(cmd);

	command_queue_enqueue(cq, cmd, num_events, events);

	MAY_BLOCK_THEN_RETURN_EVENT(ev, blocking, event);

	return CL_SUCCESS;
}
