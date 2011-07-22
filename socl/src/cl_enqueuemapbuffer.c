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

struct mb_data {
  cl_event ev;
  cl_mem buffer;
  cl_map_flags map_flags;
};

static void mapbuffer_callback(void *args) {
  struct mb_data *arg = (struct mb_data*)args;

  starpu_tag_notify_from_apps(arg->ev->id);
  arg->ev->status = CL_COMPLETE;

  gc_entity_unstore(&arg->ev);
  gc_entity_unstore(&arg->buffer);
  free(args);
}

static void mapbuffer_task(void *args) {
  struct mb_data *arg = (struct mb_data*)args;

  starpu_access_mode mode = (arg->map_flags == CL_MAP_READ ? STARPU_R : STARPU_RW);

  starpu_data_acquire_cb(arg->buffer->handle, mode, mapbuffer_callback, arg);
}

CL_API_ENTRY void * CL_API_CALL
soclEnqueueMapBuffer(cl_command_queue cq,
                   cl_mem           buffer,
                   cl_bool          blocking_map, 
                   cl_map_flags     map_flags,
                   size_t           offset, 
                   size_t           UNUSED(cb),
                   cl_uint          num_events,
                   const cl_event * events,
                   cl_event *       event,
                   cl_int *         errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
   struct starpu_task *task;
   struct mb_data *arg;
   cl_event ev;
   cl_int err;

   /* Create custom event that will be triggered when map is complete */
   ev = event_create();

   /* Store arguments */
   arg = (struct mb_data*)malloc(sizeof(struct mb_data));
   arg->map_flags = map_flags;
   gc_entity_store(&arg->ev, ev);
   gc_entity_store(&arg->buffer, buffer);

   /* Create StarPU task */
   task = task_create_cpu(CL_COMMAND_MAP_BUFFER, mapbuffer_task, arg, 0);
   cl_event map_event = task_event(task);

   /* Enqueue task */
   DEBUG_MSG("Submitting MapBuffer task (event %d)\n", ev->id);
   err = command_queue_enqueue_fakeevent(cq, task, 0, num_events, events, ev);
   gc_entity_release(map_event);

   if (errcode_ret != NULL)
      *errcode_ret = err;

   if (err != CL_SUCCESS)
      return NULL;

   if (blocking_map == CL_TRUE)
      soclWaitForEvents(1, &ev);

   RETURN_EVENT(ev, event);

   return (void*)(starpu_variable_get_local_ptr(buffer->handle) + offset);
}
