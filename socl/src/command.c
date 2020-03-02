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
#include <string.h>

/* Forward extern declaration */
extern void soclEnqueueNDRangeKernel_task(void *descr[], void *args);

cl_event command_event_get_ex(cl_command cmd)
{
	cl_event ev = cmd->event;
	gc_entity_retain(ev);
	return ev;
}

static void command_release_callback(void *a)
{
	cl_command cmd = (cl_command)a;

	// Call command specific release callback
	if (cmd->release_callback != NULL)
		cmd->release_callback(cmd);

	// Generic command destructor
	cl_uint i;
	for (i=0; i<cmd->num_events; i++)
	{
		gc_entity_unstore(&cmd->events[i]);
	}
	cmd->num_events = 0;
	free(cmd->events);

	/* Remove from command queue */
	cl_command_queue cq = cmd->event->cq;
	if (cq != NULL)
	{
		/* Lock command queue */
		STARPU_PTHREAD_MUTEX_LOCK(&cq->mutex);

		/* Remove barrier if applicable */
		if (cq->barrier == cmd)
			cq->barrier = NULL;

		/* Remove from the list of out-of-order commands */
		cq->commands = command_list_remove(cq->commands, cmd);

		/* Unlock command queue */
		STARPU_PTHREAD_MUTEX_UNLOCK(&cq->mutex);
	}

	// Events may survive to commands that created them
	cmd->event->command = NULL;
	gc_entity_unstore(&cmd->event);
}

void command_init_ex(cl_command cmd, cl_command_type typ, void (*cb)(void*))
{
	gc_entity_init(&cmd->_entity, command_release_callback, "command");
	cmd->release_callback = cb;
	cmd->typ = typ;
	cmd->num_events = 0;
	cmd->events = NULL;
	cmd->event = event_create(); // we do not use gc_entity_store here because if nobody requires the event, it should be destroyed with the command
	cmd->event->command = cmd;
	cmd->task = NULL;
	cmd->submitted = 0;
}

void command_submit_ex(cl_command cmd)
{
#define SUBMIT(typ,name) case typ:		\
	name##_submit((name)cmd);		\
	break;

	assert(cmd->submitted == 0);

	switch(cmd->typ)
	{
		SUBMIT(CL_COMMAND_NDRANGE_KERNEL, command_ndrange_kernel);
		SUBMIT(CL_COMMAND_TASK, command_ndrange_kernel);
		SUBMIT(CL_COMMAND_READ_BUFFER, command_read_buffer);
		SUBMIT(CL_COMMAND_WRITE_BUFFER, command_write_buffer);
		SUBMIT(CL_COMMAND_COPY_BUFFER, command_copy_buffer);
		SUBMIT(CL_COMMAND_MAP_BUFFER, command_map_buffer);
		SUBMIT(CL_COMMAND_UNMAP_MEM_OBJECT, command_unmap_mem_object);
		SUBMIT(CL_COMMAND_MARKER, command_marker);
		SUBMIT(CL_COMMAND_BARRIER, command_barrier);
	default:
		ERROR_STOP("Trying to submit unknown command (type %x)", cmd->typ);
	}

	cmd->submitted = 1;
#undef SUBMIT
}

cl_int command_submit_deep_ex(cl_command cmd)
{
	if (cmd->submitted == 1)
		return CL_SUCCESS;

	/* We set this in order to avoid cyclic dependencies */
	cmd->submitted = 1;

	unsigned int i;
	for (i=0; i<cmd->num_events; i++)
		command_submit_deep(cmd->events[i]->command);

	cmd->submitted = 0;

	command_submit_ex(cmd);

	return CL_SUCCESS;
}

void command_graph_dump_ex(cl_command cmd)
{
	unsigned int i;
	for (i=0; i<cmd->num_events; i++)
		command_graph_dump_ex(cmd->events[i]->command);

	const char * typ_str = (cmd->typ == CL_COMMAND_NDRANGE_KERNEL ? "ndrange_kernel" :
				cmd->typ == CL_COMMAND_TASK           ? "task"           :
				cmd->typ == CL_COMMAND_READ_BUFFER    ? "read_buffer"    :
				cmd->typ == CL_COMMAND_WRITE_BUFFER   ? "write_buffer"   :
				cmd->typ == CL_COMMAND_COPY_BUFFER    ? "copy_buffer"    :
				cmd->typ == CL_COMMAND_MAP_BUFFER     ? "map_buffer"     :
				cmd->typ == CL_COMMAND_UNMAP_MEM_OBJECT ? "unmap_mem_object" :
				cmd->typ == CL_COMMAND_MARKER         ? "marker"         :
				cmd->typ == CL_COMMAND_BARRIER        ? "barrier"        : "unknown");

	printf("CMD %p TYPE %s DEPS", cmd, typ_str);
	for (i=0; i<cmd->num_events; i++)
		printf(" %p", cmd->events[i]->command);
	printf("\n");
}

#define nullOrDup(name,size) cmd->name = memdup_safe(name,size)
#define nullOrFree(name) if (cmd->name != NULL) free((void*)cmd->name)
#define dup(name) cmd->name = name

void command_ndrange_kernel_release(void * arg)
{
	command_ndrange_kernel cmd = (command_ndrange_kernel)arg;

	gc_entity_unstore(&cmd->kernel);
	nullOrFree(global_work_offset);
	nullOrFree(global_work_size);
	nullOrFree(local_work_size);
	free(cmd->arg_sizes);
	free(cmd->arg_types);
	unsigned int i;
	for (i=0; i<cmd->num_args; i++)
	{
		free(cmd->args[i]);
		cmd->args[i] = NULL;
	}
	free(cmd->args);

	for (i=0; i<cmd->num_buffers; i++)
		gc_entity_unstore(&cmd->buffers[i]);

	free(cmd->buffers);
}

command_ndrange_kernel command_ndrange_kernel_create(cl_kernel        kernel,
						     cl_uint          work_dim,
						     const size_t *   global_work_offset,
						     const size_t *   global_work_size,
						     const size_t *   local_work_size)
{
	command_ndrange_kernel cmd = calloc(1, sizeof(struct command_ndrange_kernel_t));
	command_init(cmd, CL_COMMAND_NDRANGE_KERNEL, command_ndrange_kernel_release);

	gc_entity_store(&cmd->kernel, kernel);

	dup(work_dim);
	nullOrDup(global_work_offset, work_dim*sizeof(size_t));
	nullOrDup(global_work_size, work_dim*sizeof(size_t));
	nullOrDup(local_work_size, work_dim*sizeof(size_t));

	starpu_codelet_init(&cmd->codelet);
	cmd->codelet.where = STARPU_OPENCL;
	cmd->codelet.energy_model = NULL;
	cmd->codelet.opencl_funcs[0] = &soclEnqueueNDRangeKernel_task;

	/* Kernel is mutable, so we duplicate its parameters... */
	cmd->num_args = kernel->num_args;
	cmd->arg_sizes = memdup(kernel->arg_size, sizeof(size_t) * kernel->num_args);
	cmd->arg_types = memdup(kernel->arg_type, sizeof(enum kernel_arg_type) * kernel->num_args);
	cmd->args = memdup_deep_varsize_safe(kernel->arg_value, kernel->num_args, kernel->arg_size);

	return cmd;
}

command_ndrange_kernel command_task_create (cl_kernel kernel)
{
	static cl_uint task_work_dim = 3;
	static const size_t task_global_work_offset[3] = {0,0,0};
	static const size_t task_global_work_size[3] = {1,1,1};
	static const size_t * task_local_work_size = NULL;

	command_ndrange_kernel cmd = command_ndrange_kernel_create(kernel, task_work_dim, task_global_work_offset,
								   task_global_work_size, task_local_work_size);

	/* This is the only difference with command_ndrange_kernel_create */
	cmd->_command.typ = CL_COMMAND_TASK;

	return cmd;
}

command_barrier command_barrier_create ()
{
	command_barrier cmd = malloc(sizeof(struct command_barrier_t));
	command_init(cmd, CL_COMMAND_BARRIER, NULL);

	return cmd;
}

command_marker command_marker_create ()
{
	command_marker cmd = malloc(sizeof(struct command_marker_t));
	command_init(cmd, CL_COMMAND_MARKER, NULL);

	return cmd;
}

void command_map_buffer_release(void * UNUSED(arg))
{
	/* We DO NOT unstore (release) the buffer as unmap will do it
	   gc_entity_unstore(&cmd->buffer); */
}

command_map_buffer command_map_buffer_create(cl_mem buffer,
					     cl_map_flags map_flags,
					     size_t offset,
					     size_t cb
					     )
{
	command_map_buffer cmd = malloc(sizeof(struct command_map_buffer_t));
	command_init(cmd, CL_COMMAND_MAP_BUFFER, command_map_buffer_release);

	gc_entity_store(&cmd->buffer, buffer);
	dup(map_flags);
	dup(offset);
	dup(cb);

	return cmd;
}

void command_unmap_mem_object_release(void * arg)
{
	command_unmap_mem_object cmd = (command_unmap_mem_object)arg;

	/* We release the buffer twice because map buffer command did not */
	gc_entity_release(cmd->buffer);
	gc_entity_unstore(&cmd->buffer);
}

command_unmap_mem_object command_unmap_mem_object_create(cl_mem buffer, void * ptr)
{
	command_unmap_mem_object cmd = malloc(sizeof(struct command_unmap_mem_object_t));
	command_init(cmd, CL_COMMAND_UNMAP_MEM_OBJECT, command_unmap_mem_object_release);

	gc_entity_store(&cmd->buffer, buffer);
	dup(ptr);

	return cmd;
}

void command_read_buffer_release(void *arg)
{
	command_read_buffer cmd = (command_read_buffer)arg;
	gc_entity_unstore(&cmd->buffer);
}

command_read_buffer command_read_buffer_create(cl_mem buffer, size_t offset, size_t cb, void * ptr)
{
	command_read_buffer cmd = malloc(sizeof(struct command_read_buffer_t));
	command_init(cmd, CL_COMMAND_READ_BUFFER, command_read_buffer_release);

	gc_entity_store(&cmd->buffer, buffer);
	dup(offset);
	dup(cb);
	dup(ptr);

	return cmd;
}

void command_write_buffer_release(void *arg)
{
	command_write_buffer cmd = (command_write_buffer)arg;
	gc_entity_unstore(&cmd->buffer);
}

command_write_buffer command_write_buffer_create(cl_mem buffer, size_t offset, size_t cb, const void * ptr)
{
	command_write_buffer cmd = malloc(sizeof(struct command_write_buffer_t));
	command_init(cmd, CL_COMMAND_WRITE_BUFFER, command_write_buffer_release);

	gc_entity_store(&cmd->buffer, buffer);
	dup(offset);
	dup(cb);
	dup(ptr);

	return cmd;
}

void command_copy_buffer_release(void *arg)
{
	command_copy_buffer cmd = (command_copy_buffer)arg;
	gc_entity_unstore(&cmd->src_buffer);
	gc_entity_unstore(&cmd->dst_buffer);
}

command_copy_buffer command_copy_buffer_create( cl_mem src_buffer, cl_mem dst_buffer,
						size_t src_offset, size_t dst_offset, size_t cb)
{
	command_copy_buffer cmd = malloc(sizeof(struct command_copy_buffer_t));
	command_init(cmd, CL_COMMAND_COPY_BUFFER, command_copy_buffer_release);

	gc_entity_store(&cmd->src_buffer, src_buffer);
	gc_entity_store(&cmd->dst_buffer, dst_buffer);
	dup(src_offset);
	dup(dst_offset);
	dup(cb);

	return cmd;
}

#undef nullOrDup
#undef nodeNullOrDup
#undef dup
#undef nodeDup
#undef memdup
