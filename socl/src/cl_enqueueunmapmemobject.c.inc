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

CL_API_ENTRY cl_int CL_API_CALL
soclEnqueueUnmapMemObject(cl_command_queue cq,
                        cl_mem            memobj,
                        void *            UNUSED(mapped_ptr),
                        cl_uint           num_events,
                        const cl_event *  events,
                        cl_event *        event) CL_API_SUFFIX__VERSION_1_0
{
   struct starpu_task *task;
   cl_int err;
   cl_event ev;

   /* Create StarPU task */
   task = task_create_cpu(CL_COMMAND_UNMAP_MEM_OBJECT, (void(*)(void*))starpu_data_release, memobj->handle, 0);
   ev = task_event(task);

   DEBUG_MSG("Submitting UnmapBuffer task (event %d)\n", task->tag_id);

   err = command_queue_enqueue(cq, task, 0, num_events, events);

   RETURN_EVENT(ev, event);

   return err;
}
