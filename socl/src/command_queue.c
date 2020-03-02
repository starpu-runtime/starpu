/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2012       Vincent Danjean
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
#include "task.h"
#include "gc.h"

/**
 * WARNING: command queues do NOT hold references on events. Only events hold references
 * on command queues. This way, event release will automatically remove the event from
 * its command queue.
 */

void command_queue_enqueue_ex(cl_command_queue cq, cl_command cmd, cl_uint num_events, const cl_event * events)
{
	cl_event ev = command_event_get_ex(cmd);
	ev->prof_queued = _socl_nanotime();
	gc_entity_release(ev);

	/* Check if the command is a barrier */
	int is_barrier = (cmd->typ == CL_COMMAND_BARRIER || !(cq->properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE));

	/* Add references to the command queue */
	gc_entity_store(&cmd->event->cq, cq);

	/* Lock command queue */
	STARPU_PTHREAD_MUTEX_LOCK(&cq->mutex);

	/*** Number of dependencies ***/
	int ndeps = num_events;

	/* Add dependency to last barrier if applicable */
	if (cq->barrier != NULL)
		ndeps++;

	/* Add dependencies to out-of-order events (if any) */
	if (is_barrier)
	{
		command_list cl = cq->commands;
		while (cl != NULL)
		{
			ndeps++;
			cl = cl->next;
		}
	}

	/*** Dependencies ***/
	cl_event * deps = malloc(ndeps * sizeof(cl_event));

	int n = 0;

	/* Add dependency to last barrier if applicable */
	if (cq->barrier != NULL)
		gc_entity_store(&deps[n++], cq->barrier->event);

	/* Add dependencies to out-of-order events (if any) */
	if (is_barrier)
	{
		command_list cl = cq->commands;
		while (cl != NULL)
		{
			gc_entity_store(&deps[n++], cl->cmd->event);
			cl = cl->next;
		}
	}

	/* Add explicit dependencies */
	unsigned i;
	for (i=0; i<num_events; i++)
	{
		gc_entity_store(&deps[n++], events[i]);
	}

	/* Make all dependencies explicit for the command */
	cmd->num_events = ndeps;
	cmd->events = deps;

	/* Insert command in the queue */
	if (is_barrier)
	{
		/* Remove out-of-order commands */
		cq->commands = NULL;
		/* Register the command as the last barrier */
		cq->barrier = cmd;
	}
	else
	{
		/* Add command to the list of out-of-order commands */
		cq->commands = command_list_cons(cmd, cq->commands);
	}

	/* Submit command
	 * We need to do it before unlocking because we don't want events to get
	 * released while we use them to set dependencies
	 */
	command_submit_ex(cmd);

	/* Unlock command queue */
	STARPU_PTHREAD_MUTEX_UNLOCK(&cq->mutex);

	gc_entity_release(cmd);
}
