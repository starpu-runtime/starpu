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
#include "gc.h"
#include "event.h"

cl_event task_event(starpu_task *task) {
  return (cl_event)task->callback_arg;
}

static void task_release_callback(void *arg) {
  starpu_task *task = starpu_get_current_task();
  cl_event ev = (cl_event)arg;
  
  ev->status = CL_COMPLETE;

  if (task->profiling_info != NULL && (intptr_t)task->profiling_info != -ENOSYS) {
    ev->profiling_info = malloc(sizeof(*task->profiling_info));
    memcpy(ev->profiling_info, task->profiling_info, sizeof(*task->profiling_info));
  }

  gc_entity_release(ev);
}


/*
 * Create a StarPU task
 *
 * Task's callback_arg is event
 * Task's tag is set to event ID
 */
starpu_task * task_create(cl_command_type type) {
   cl_event event;

   /* Create event */
   event = event_create();

   return task_create_with_event(type, event);
}


starpu_task * task_create_with_event(cl_command_type type, cl_event event) {
   struct starpu_task * task;

   event->type = type;

   /* Create StarPU task */
   task = starpu_task_create();

   /* Task tag is set to event id */
   task->use_tag = 1;
   task->tag_id = event->id;

   /* Set task common settings */
   task->destroy = 1;
   task->detach = 1;
   task->callback_func = task_release_callback;
   task->callback_arg = event;

   return task;
}


void task_dependency_add(starpu_task * task, cl_uint num, const cl_event *events) {
   unsigned int i;

   for (i=0; i<num; i++) {
      starpu_tag_t tag = events[i]->id;
      DEBUG_MSG("Event %d depends on event %d\n", task->tag_id, events[i]->id);
      starpu_tag_declare_deps_array(task->tag_id, 1, &tag);
   }
}

cl_int task_submit(starpu_task * task, cl_int num_events, cl_event * events) {

	task_dependency_add(task, num_events, events);

	/* Submit task */
	int ret = starpu_task_submit(task);
	gc_entity_retain(task_event(task));
	if (ret != 0)
		DEBUG_ERROR("Unable to submit a task. Error %d\n", ret);

	return CL_SUCCESS;
}


/*********************************
 * CPU task helper
 *********************************/

struct cputask_arg {
  void (*callback)(void*);
  void * arg;
  int free_arg;
};

static void cputask_task(__attribute__((unused)) void *descr[], void *args) {
  struct cputask_arg * arg = (struct cputask_arg*)args;

  arg->callback(arg->arg);

  if (arg->free_arg)
    free(arg->arg);

  free(arg);
}

static starpu_codelet cputask_codelet = {
   .where = STARPU_CPU,
   .model = NULL,
   .cpu_func = &cputask_task
};

starpu_task * task_create_cpu(cl_command_type type, void (*callback)(void*), void *arg, int free_arg) {
  
  struct cputask_arg * a = malloc(sizeof(struct cputask_arg));
  a->callback = callback;
  a->arg = arg;
  a->free_arg = free_arg;

  starpu_task *task = task_create(type);
  task->cl = &cputask_codelet;
  task->cl_arg = a;

  return task;
}

