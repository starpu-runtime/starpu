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

struct arg_copybuffer {
   size_t src_offset, dst_offset;
   cl_mem src_buffer, dst_buffer;
   size_t cb;
};

static void soclEnqueueCopyBuffer_opencl_task(void *descr[], void *args) {
   struct arg_copybuffer *arg;
   int wid;
   cl_command_queue cq;
   cl_event ev;

   arg = (struct arg_copybuffer*)args;
   wid = starpu_worker_get_id();
   starpu_opencl_get_queue(wid, &cq);

   cl_mem src = (cl_mem)STARPU_VARIABLE_GET_PTR(descr[0]);
   cl_mem dst = (cl_mem)STARPU_VARIABLE_GET_PTR(descr[1]);

   clEnqueueCopyBuffer(cq, src,dst, arg->src_offset, arg->dst_offset, arg->cb, 0, NULL, &ev);
   clWaitForEvents(1, &ev);
   clReleaseEvent(ev);

   gc_entity_unstore(&arg->src_buffer);
   gc_entity_unstore(&arg->dst_buffer);

   free(arg);
}

static void soclEnqueueCopyBuffer_cpu_task(void *descr[], void *args) {
   struct arg_copybuffer *arg;
   arg = (struct arg_copybuffer*)args;
   void * src = (void*)STARPU_VARIABLE_GET_PTR(descr[0]);
   void * dst = (void*)STARPU_VARIABLE_GET_PTR(descr[1]);
   memcpy(dst+arg->dst_offset, src+arg->src_offset, arg->cb);

   gc_entity_unstore(&arg->src_buffer);
   gc_entity_unstore(&arg->dst_buffer);

   free(arg);
}

static starpu_codelet codelet_copybuffer = {
   .where = STARPU_CPU | STARPU_OPENCL,
   .model = NULL,
   .cpu_func = &soclEnqueueCopyBuffer_cpu_task,
   .opencl_func = &soclEnqueueCopyBuffer_opencl_task,
   .nbuffers = 2
};

CL_API_ENTRY cl_int CL_API_CALL
soclEnqueueCopyBuffer(cl_command_queue  cq, 
                    cl_mem              src_buffer,
                    cl_mem              dst_buffer, 
                    size_t              src_offset,
                    size_t              dst_offset,
                    size_t              cb, 
                    cl_uint             num_events,
                    const cl_event *    events,
                    cl_event *          event) CL_API_SUFFIX__VERSION_1_0
{
   struct starpu_task *task;
   struct arg_copybuffer *arg;
   cl_event ev;

   task = task_create(CL_COMMAND_COPY_BUFFER);
   ev = task_event(task);

   task->buffers[0].handle = src_buffer->handle;
   task->buffers[0].mode = STARPU_R;
   task->buffers[1].handle = dst_buffer->handle;
   task->buffers[1].mode = STARPU_RW;
   task->cl = &codelet_copybuffer;

   arg = (struct arg_copybuffer*)malloc(sizeof(struct arg_copybuffer));
   arg->src_offset = src_offset;
   arg->dst_offset = dst_offset;
   arg->cb = cb;
   gc_entity_store(&arg->src_buffer, src_buffer);
   gc_entity_store(&arg->dst_buffer, dst_buffer);
   task->cl_arg = arg;
   task->cl_arg_size = sizeof(struct arg_copybuffer);

   dst_buffer->scratch = 0;

   DEBUG_MSG("Submitting CopyBuffer task (event %d)\n", ev->id);

   cl_int ret = command_queue_enqueue(cq, task, 0, num_events, events);

   RETURN_EVENT(ev, event);

   return ret;
}
