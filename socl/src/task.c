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
#include "gc.h"
#include "event.h"

void command_completed(cl_command cmd)
{
	starpu_task task = cmd->task;

	cl_event ev = command_event_get_ex(cmd);
	ev->status = CL_COMPLETE;

	ev->prof_end = _socl_nanotime();

	/* Commands without codelets (marker, barrier, unmap...) take no time */
	if (task->cl == NULL)
		ev->prof_start = ev->prof_end;

	/* Trigger the tag associated to the command event */
	DEBUG_MSG("Trigger event %d\n", ev->id);
	starpu_tag_notify_from_apps(ev->id);

	gc_entity_release(ev);
}

void command_completed_task_callback(void *arg)
{
	cl_command cmd = (cl_command)arg;

	command_completed(cmd);

	/* Release the command stored task callback parameter */
	gc_entity_release(cmd);
}

/*
 * Create a StarPU task
 */
starpu_task task_create()
{
	struct starpu_task * task;

	/* Create StarPU task */
	task = starpu_task_create();

	/* Set task common settings */
	task->destroy = 0;
	task->detach = 0;

	task->use_tag = 1;
	task->tag_id = event_unique_id();

	return task;
}

void task_depends_on(starpu_task task, cl_uint num_events, cl_event *events)
{
	if (num_events != 0)
	{
		cl_uint i;

		starpu_tag_t * tags = malloc(num_events * sizeof(starpu_tag_t));

		DEBUG_MSG("Task %p depends on events:", task);
		for (i=0; i<num_events; i++)
		{
			tags[i] = events[i]->id;
			DEBUG_MSG_NOHEAD(" %d", events[i]->id);
		}
		DEBUG_MSG_NOHEAD("\n");

		starpu_tag_declare_deps_array(task->tag_id, num_events, tags);

		free(tags);
	}
}

cl_int task_submit_ex(starpu_task task, cl_command cmd)
{
	/* Associated the task to the command */
	cmd->task = task;

	cl_uint num_events = command_num_events_get_ex(cmd);
	cl_event * events = command_events_get_ex(cmd);

	task_depends_on(task, num_events, events);

	task->callback_func = command_completed_task_callback;
	gc_entity_store(&task->callback_arg, cmd);

	cl_event ev = command_event_get_ex(cmd);
	ev->prof_submit = _socl_nanotime();
	gc_entity_release(ev);

	/* Submit task */
	int ret = (task->cl != NULL && task->where == STARPU_OPENCL ?
		   starpu_task_submit_to_ctx(task, cmd->event->cq->context->sched_ctx) :
		   starpu_task_submit(task));

	if (ret != 0)
		DEBUG_ERROR("Unable to submit a task. Error %d\n", ret);

	return CL_SUCCESS;
}


/*********************************
 * CPU task helper
 *********************************/

struct cputask_arg
{
	void (*callback)(void*);
	void * arg;
	int free_arg;
	cl_command cmd;
	int complete_cmd;
};

static void cputask_task(void *args)
{
	struct cputask_arg * arg = (struct cputask_arg*)args;

	arg->callback(arg->arg);

	if (arg->complete_cmd)
		command_completed(arg->cmd);

	if (arg->free_arg)
	{
		assert(arg->arg != NULL);
		free(arg->arg);
		arg->arg = NULL;
	}

	gc_entity_unstore(&arg->cmd);
	free(arg);
}

void cpu_task_submit_ex(cl_command cmd, void (*callback)(void*), void *arg, int free_arg, int complete_cmd, struct starpu_codelet * codelet, unsigned num_events, cl_event * events)
{
	struct cputask_arg * a = malloc(sizeof(struct cputask_arg));
	a->callback = callback;
	a->arg = arg;
	a->free_arg = free_arg;
	gc_entity_store(&a->cmd, cmd);
	a->complete_cmd = complete_cmd;

	codelet->where = STARPU_OPENCL | STARPU_CPU | STARPU_CUDA;

	starpu_task task = task_create();
	if (num_events != 0)
	{
		task_depends_on(task, num_events, events);
	}

	task->callback_func = cputask_task;
	task->callback_arg = a;

	cmd->task = task;

	int ret = starpu_task_submit(task);
	if (ret != 0)
		DEBUG_ERROR("Unable to submit a task. Error %d\n", ret);
}
