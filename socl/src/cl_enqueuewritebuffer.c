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

static void soclEnqueueWriteBuffer_cpu_task(void *descr[], void *args)
{
	command_write_buffer cmd = (command_write_buffer)args;

	cl_event ev = command_event_get(cmd);
	ev->prof_start = _socl_nanotime();
	gc_entity_release(ev);

	char * ptr = (void*)STARPU_VARIABLE_GET_PTR(descr[0]);
	DEBUG_MSG("[Buffer %d] Writing %ld bytes from %p to %p\n", cmd->buffer->id, (long)cmd->cb, cmd->ptr, ptr+cmd->offset);

	//FIXME: Fix for people who use USE_HOST_PTR, modify data at host_ptr and use WriteBuffer to commit the change.
	// StarPU may have erased host mem at host_ptr (for instance by retrieving current buffer data at host_ptr)
	// Buffer mapping facilities should be used instead
	// Maybe we should report the bug here... for now, we just avoid memcpy crash due to overlapping regions...
	if (ptr+cmd->offset != cmd->ptr)
		memcpy(ptr+cmd->offset, cmd->ptr, cmd->cb);

	gc_entity_release_cmd(cmd);
}

static void soclEnqueueWriteBuffer_opencl_task(void *descr[], void *args)
{
	command_write_buffer cmd = (command_write_buffer)args;

	cl_event event = command_event_get(cmd);
	event->prof_start = _socl_nanotime();
	gc_entity_release(event);

	cl_mem mem = (cl_mem)STARPU_VARIABLE_GET_PTR(descr[0]);

	DEBUG_MSG("[Buffer %d] Writing %ld bytes to offset %ld from %p\n", cmd->buffer->id, (long)cmd->cb, (long)cmd->offset, cmd->ptr);

	int wid = starpu_worker_get_id_check();
	cl_command_queue cq;
	starpu_opencl_get_queue(wid, &cq);

	cl_event ev;

	cl_int err = clEnqueueWriteBuffer(cq, mem, CL_TRUE, cmd->offset, cmd->cb, cmd->ptr, 0, NULL, &ev);
	if (err != CL_SUCCESS)
		ERROR_CL("clEnqueueWriteBuffer", err);

	clWaitForEvents(1, &ev);
	clReleaseEvent(ev);

	gc_entity_release_cmd(cmd);
}

static struct starpu_perfmodel write_buffer_perfmodel =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "SOCL_WRITE_BUFFER"
};

static struct starpu_codelet codelet_writebuffer =
{
	.where = STARPU_OPENCL,
	.model = &write_buffer_perfmodel,
	.cpu_funcs = { &soclEnqueueWriteBuffer_cpu_task },
	.opencl_funcs = { &soclEnqueueWriteBuffer_opencl_task },
	.modes = {STARPU_W},
	.nbuffers = 1
};

static struct starpu_codelet codelet_writebuffer_partial =
{
	.where = STARPU_OPENCL,
	.model = &write_buffer_perfmodel,
	.cpu_funcs = { &soclEnqueueWriteBuffer_cpu_task },
	.opencl_funcs = { &soclEnqueueWriteBuffer_opencl_task },
	.modes = {STARPU_RW},
	.nbuffers = 1
};

cl_int command_write_buffer_submit(command_write_buffer cmd)
{
	/* Aliases */
	cl_mem buffer = cmd->buffer;
	size_t cb = cmd->cb;

	struct starpu_task *task;
	task = task_create(CL_COMMAND_WRITE_BUFFER);

	task->handles[0] = buffer->handle;
	//If only a subpart of the buffer is written, RW access mode is required
	if (cb != buffer->size)
		task->cl = &codelet_writebuffer_partial;
	else
		task->cl = &codelet_writebuffer;

	gc_entity_store_cmd(&task->cl_arg, cmd);
	task->cl_arg_size = sizeof(*cmd);

	/* Execute the task on a specific worker? */
	if (cmd->_command.event->cq->device != NULL)
	{
		task->execute_on_a_specific_worker = 1;
		task->workerid = cmd->_command.event->cq->device->worker_id;
	}

	//The buffer now contains meaningful data
	cmd->buffer->scratch = 0;

	task_submit(task, cmd);

	return CL_SUCCESS;
}

CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY cl_int CL_API_CALL
soclEnqueueWriteBuffer(cl_command_queue cq,
		       cl_mem             buffer,
		       cl_bool            blocking,
		       size_t             offset,
		       size_t             cb,
		       const void *       ptr,
		       cl_uint            num_events,
		       const cl_event *   events,
		       cl_event *         event)
{
	command_write_buffer cmd = command_write_buffer_create(buffer, offset, cb, ptr);

	cl_event ev = command_event_get(cmd);

	command_queue_enqueue(cq, cmd, num_events, events);

	MAY_BLOCK_THEN_RETURN_EVENT(ev, blocking, event);

	return CL_SUCCESS;
}
