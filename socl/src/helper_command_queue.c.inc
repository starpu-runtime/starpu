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


/**
 * WARNING: command queues do NOT hold references on events. Only events hold references
 * on command queues. This way, event release will automatically remove the event from
 * its command queue.
 */

/**
 * Enqueue the given task but put fake_event into the command queue.
 * This is used when a tag notified by application is used (cf clEnqueueMapBuffer, etc.)
 */
static cl_int command_queue_enqueue_fakeevent(cl_command_queue cq, starpu_task *task, cl_int barrier, cl_int num_events, const cl_event * events, cl_event fake_event) {

  int in_order = !(cq->properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

  /* Set explicit task dependencies */
  task_dependency_add(task, num_events, events);

  /* Lock command queue */
  pthread_spin_lock(&cq->spin);

  /* Add dependency to last barrier if applicable */
  if (cq->barrier != NULL)
    task_dependency_add(task, 1, &cq->barrier);

  /* Add dependencies to out-of-order events (if any) */
  if (barrier) {
    while (cq->events != NULL) {
      task_dependency_add(task, 1, &cq->events);
      cq->events = cq->events->next;
    }
  }

  cl_event ev = (fake_event == NULL ? task_event(task) : fake_event);

  /* Add event to the list of out-of-order events */
  if (!in_order) {
    ev->next = cq->events;
    ev->prev = NULL;
    if (cq->events != NULL)
      cq->events->prev = ev;
    cq->events = ev;
  }

  /* Register this event as last barrier */
  if (barrier || in_order)
    cq->barrier = ev;

   /* Unlock command queue */
   pthread_spin_unlock(&cq->spin);

   /* Add reference to the command queue */
   gc_entity_store(&ev->cq, cq);

   /* Submit task */
   gc_entity_retain(task_event(task));
   int ret = starpu_task_submit(task);
   if (ret != 0)
      DEBUG_ERROR("Unable to submit a task. Error %d\n", ret);

   return CL_SUCCESS;
}

static cl_int command_queue_enqueue(cl_command_queue cq, starpu_task *task, cl_int barrier, cl_int num_events, const cl_event * events) {
  return command_queue_enqueue_fakeevent(cq, task, barrier, num_events, events, NULL);
}


static cl_event enqueueBarrier(cl_command_queue cq) {

   //CL_COMMAND_MARKER has been chosen as CL_COMMAND_BARRIER doesn't exist
   starpu_task * task = task_create(CL_COMMAND_MARKER);

   DEBUG_MSG("Submitting barrier task (event %d)\n", task->tag_id);
   command_queue_enqueue(cq, task, 1, 0, NULL);

   return task_event(task);
}
