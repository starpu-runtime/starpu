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

typedef struct running_kernel * running_kernel;

struct running_kernel {
  cl_kernel kernel;
  cl_mem *buffers;
  unsigned int buffer_count;
  starpu_codelet *codelet;
  cl_uint work_dim;
  size_t * global_work_offset;
  size_t * global_work_size;
  size_t * local_work_size;
  /* Arguments */
  unsigned int arg_count;
  size_t *arg_size;
  enum kernel_arg_type  *arg_type;
  void  **arg_value;
};

static void soclEnqueueNDRangeKernel_task(void *descr[], void *args) {
   running_kernel d;
   cl_command_queue cq;
   int wid;
   cl_int err;

   d = (running_kernel)args;
   wid = starpu_worker_get_id();
   starpu_opencl_get_queue(wid, &cq);

   DEBUG_MSG("[worker %d] [kernel %d] Executing kernel...\n", wid, d->kernel->id);

   int range = starpu_worker_get_range();

   /* Set arguments */
   {
      unsigned int i;
      int buf = 0;
      for (i=0; i<d->arg_count; i++) {
         switch (d->arg_type[i]) {
            case Null:
               err = clSetKernelArg(d->kernel->cl_kernels[range], i, d->arg_size[i], NULL);
               break;
            case Buffer: {
                  cl_mem mem;  
                  mem = (cl_mem)STARPU_VARIABLE_GET_PTR(descr[buf]);
                  err = clSetKernelArg(d->kernel->cl_kernels[range], i, d->arg_size[i], &mem);
                  buf++;
               }
               break;
            case Immediate:
               err = clSetKernelArg(d->kernel->cl_kernels[range], i, d->arg_size[i], d->arg_value[i]);
               break;
         }
         if (err != CL_SUCCESS) {
            DEBUG_CL("clSetKernelArg", err);
            DEBUG_ERROR("Aborting\n");
         }
      }
   }

   /* Calling Kernel */
   cl_event event;
   err = clEnqueueNDRangeKernel(cq, d->kernel->cl_kernels[range], d->work_dim, d->global_work_offset, d->global_work_size, d->local_work_size, 0, NULL, &event);

   if (err != CL_SUCCESS) {
      ERROR_MSG("Worker[%d] Unable to Enqueue kernel (error %d)\n", wid, err);
      DEBUG_CL("clEnqueueNDRangeKernel", err);
      DEBUG_MSG("Workdim %d, global_work_offset %p, global_work_size %p, local_work_size %p\n",
                d->work_dim, d->global_work_offset, d->global_work_size, d->local_work_size);
      DEBUG_MSG("Global work size: %ld %ld %ld\n", d->global_work_size[0],
            (d->work_dim > 1 ? d->global_work_size[1] : 1), (d->work_dim > 2 ? d->global_work_size[2] : 1)); 
      if (d->local_work_size != NULL)
         DEBUG_MSG("Local work size: %ld %ld %ld\n", d->local_work_size[0],
               (d->work_dim > 1 ? d->local_work_size[1] : 1), (d->work_dim > 2 ? d->local_work_size[2] : 1)); 
      ERROR_MSG("Aborting.\n");
      exit(1);
   }

   /* Waiting for kernel to terminate */
   clWaitForEvents(1, &event);
   clReleaseEvent(event);
}

static void cleaning_task_callback(void *args) {
   running_kernel arg = (running_kernel)args;

   free(arg->arg_size);
   free(arg->arg_type);

   unsigned int i;
   for (i=0; i<arg->arg_count; i++) {
      free(arg->arg_value[i]);
   }
   free(arg->arg_value);

   for (i=0; i<arg->buffer_count; i++)
      gc_entity_unstore(&arg->buffers[i]);

   gc_entity_unstore(&arg->kernel);

   free(arg->buffers);
   free(arg->global_work_offset);
   free(arg->global_work_size);
   free(arg->local_work_size);
   void * co = arg->codelet;
   arg->codelet = NULL;
   free(co);
}

static struct starpu_perfmodel_t perf_model = {
  .type = STARPU_HISTORY_BASED,
  .symbol = "perf_model"
};

/**
 * Real kernel enqueuing command
 */
cl_int graph_play_enqueue_kernel(node_enqueue_kernel n) {

   struct starpu_task *task;
   running_kernel arg;
   starpu_codelet *codelet;
   cl_event ev;
   
   /* Alias struc fields */
   cl_command_queue cq = n->cq;
   cl_kernel        kernel = n->kernel;
   cl_uint          work_dim = n->work_dim;
   size_t *	    global_work_offset = (size_t*)n->global_work_offset;
   size_t *   	    global_work_size = (size_t*)n->global_work_size;
   size_t *   	    local_work_size = (size_t*)n->local_work_size;
   cl_uint          num_events = n->node.num_events;
   const cl_event * events = n->node.events;
   cl_event         event = n->node.event;
   char 	    is_task = n->is_task;
   cl_int ndeps;
   cl_event *deps;


   /* Allocate structures */

   /* Codelet */
   codelet = (starpu_codelet*)malloc(sizeof(starpu_codelet));
   if (codelet == NULL)
      return CL_OUT_OF_HOST_MEMORY;

   /* Codelet arguments */
   arg = (running_kernel)malloc(sizeof(struct running_kernel));
   if (arg == NULL) {
      free(codelet);
      return CL_OUT_OF_HOST_MEMORY;
   }

	/* StarPU task */
	if (event != NULL) {
		task = task_create_with_event(is_task ? CL_COMMAND_TASK : CL_COMMAND_NDRANGE_KERNEL, event);
	}
	else {
		
		task = task_create(is_task ? CL_COMMAND_TASK : CL_COMMAND_NDRANGE_KERNEL);
	}
	ev = task_event(task);

   /*******************
    * Initializations *
    *******************/

   /* ------- *
    * Codelet *
    * ------- */
   codelet->where = STARPU_OPENCL;
   codelet->power_model = NULL;
   codelet->opencl_func = &soclEnqueueNDRangeKernel_task;
   //codelet->model = NULL;
   codelet->model = &perf_model;

   /* ---------------- *
    * Codelet argument *
    * ---------------- */
   gc_entity_store(&arg->kernel, kernel);
   arg->work_dim = work_dim;
   arg->codelet = codelet;

   arg->global_work_offset = memdup_safe(global_work_offset, sizeof(size_t)*work_dim);
   arg->global_work_size = memdup_safe(global_work_size, sizeof(size_t)*work_dim);
   arg->local_work_size = memdup_safe(local_work_size, sizeof(size_t)*work_dim);

   /* ----------- *
    * StarPU task *
    * ----------- */
   task->cl = codelet;
   task->cl_arg = arg;
   task->cl_arg_size = sizeof(struct running_kernel);

   /* Convert OpenCL's memory objects to StarPU buffers */
   codelet->nbuffers = 0;
   {
      arg->buffers = malloc(sizeof(cl_mem) * kernel->arg_count);
      arg->buffer_count = 0;

      unsigned int i;
      for (i=0; i<kernel->arg_count; i++) {
         if (kernel->arg_type[i] == Buffer) {

            cl_mem buf = (cl_mem)kernel->arg_value[i];

            /* We save cl_mem references in order to properly release them after kernel termination */
            gc_entity_store(&arg->buffers[arg->buffer_count], buf);
            arg->buffer_count += 1;

            codelet->nbuffers++;
            task->buffers[codelet->nbuffers-1].handle = buf->handle;

            /* Determine best StarPU buffer access mode */
            int mode;
            if (buf->mode == CL_MEM_READ_ONLY)
               mode = STARPU_R;
            else if (buf->mode == CL_MEM_WRITE_ONLY) {
               mode = STARPU_W;
               buf->scratch = 0;
            }
            else if (buf->scratch) { //RW but never accessed in RW or W mode
               mode = STARPU_W;
               buf->scratch = 0;
            }
            else {
               mode = STARPU_RW;
               buf->scratch = 0;
            }
            task->buffers[codelet->nbuffers-1].mode = mode; 
         }
      }
   }

   /* Copy arguments as kernel args can be modified by the time we launch the kernel */
   arg->arg_count = kernel->arg_count;
   arg->arg_size = memdup(kernel->arg_size, sizeof(size_t) * kernel->arg_count);
   arg->arg_type = memdup(kernel->arg_type, sizeof(enum kernel_arg_type) * kernel->arg_count);
   arg->arg_value = memdup_deep_varsize_safe(kernel->arg_value, kernel->arg_count, kernel->arg_size);

   DEBUG_MSG("Submitting NDRange task (event %d)\n", ev->id);

   command_queue_enqueue(cq, task_event(task), 0, num_events, events, &ndeps, &deps);

   task_submit(task, ndeps, deps);

   /* Enqueue a cleaning task */
   starpu_task * cleaning_task = task_create_cpu(0, cleaning_task_callback, arg,1);
   task_submit(cleaning_task, 1, &ev);
  
   return CL_SUCCESS;
}

/**
 * Virtual kernel enqueueing command
 */
CL_API_ENTRY cl_int CL_API_CALL
soclEnqueueNDRangeKernel(cl_command_queue cq,
                       cl_kernel        kernel,
                       cl_uint          work_dim,
                       const size_t *   global_work_offset,
                       const size_t *   global_work_size,
                       const size_t *   local_work_size,
                       cl_uint          num_events,
                       const cl_event * events,
                       cl_event *       event) CL_API_SUFFIX__VERSION_1_1
{
	node_enqueue_kernel n;

	n = graph_create_enqueue_kernel(0, cq, kernel, work_dim, global_work_offset, global_work_size,
		local_work_size, num_events, events, kernel->arg_count, kernel->arg_size,
		kernel->arg_type, kernel->arg_value);
	
	//FIXME: temporarily, we execute the node directly. In the future, we will postpone this.
	graph_play_enqueue_kernel(n);
	graph_free(n);

	//graph_store(n);

	RETURN_OR_RELEASE_EVENT(n->node.event, event);

	return CL_SUCCESS;
}
