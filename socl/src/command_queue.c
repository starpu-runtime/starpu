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
#include "task.h"
#include "gc.h"

/**
 * WARNING: command queues do NOT hold references on events. Only events hold references
 * on command queues. This way, event release will automatically remove the event from
 * its command queue.
 */


/**
 * Returned implicit dependencies for a task
 * Command queue must be locked!
 */
void command_queue_dependencies_implicit(
	cl_command_queue cq, 	/* Command queue */
	char is_barrier,	/* Is the task a barrier */
	cl_int * ret_num_events,	/* Returned number of dependencies */
	cl_event ** ret_events	/* Returned dependencies */
) {

	/*********************
	 * Count dependencies
	 *********************/
	int ndeps = 0;

	/* Add dependency to last barrier if applicable */
	if (cq->barrier != NULL)
		ndeps++;

	/* Add dependencies to out-of-order events (if any) */
	if (is_barrier) {
		cl_event ev = cq->events;
		while (ev != NULL) {
			ndeps++;
			ev = ev->next;
		}
	}

	/*********************
	 * Return dependencies
	 *********************/

	cl_event * evs = malloc(ndeps * sizeof(cl_event));
	int n = 0;

	/* Add dependency to last barrier if applicable */
	if (cq->barrier != NULL)
		evs[n++] = cq->barrier;

	/* Add dependencies to out-of-order events (if any) */
	if (is_barrier) {
		cl_event ev = cq->events;
		while (ev != NULL) {
			evs[n++] = ev;
			ev = ev->next;
		}
	}

	*ret_num_events = ndeps;
	*ret_events = evs;
}
	
/**
 * Insert a task in the command queue
 * The command queue must be locked!
 */
void command_queue_insert(
	cl_command_queue cq, 	/* Command queue */
	cl_event task_event,	/* Event for the task */
	char is_barrier		/* Is the task a barrier */
) {

	int in_order = !(cq->properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

	/*********************
	 * Insert event
	 *********************/

	if (is_barrier)
		cq->events = NULL;

	/* Add event to the list of out-of-order events */
	if (!in_order) {
		task_event->next = cq->events;
		task_event->prev = NULL;
		if (cq->events != NULL)
			cq->events->prev = task_event;
		cq->events = task_event;
	}

	/* Register this event as last barrier */
	if (is_barrier || in_order)
		cq->barrier = task_event;

	/* Add reference to the command queue */
	gc_entity_store(&task_event->cq, cq);
}

/**
 * Return implicit and explicit dependencies for a task
 * The command queue must be locked!
 */
void command_queue_dependencies(
	cl_command_queue cq,	/* Command queue */
	char is_barrier,	/* Is the task a barrier */
	cl_int num_events,	/* Number of explicit dependencies */
	const cl_event events,	/* Explicit dependencies */
	cl_int * ret_num_events,	/* Returned number of dependencies */
	cl_event ** ret_events	/* Returned dependencies */
) {
	cl_int implicit_num_events;
	cl_event * implicit_events;

	/* Implicit dependencies */
	command_queue_dependencies_implicit(cq, is_barrier, &implicit_num_events, &implicit_events);

	/* Explicit dependencies */
	cl_int ndeps = implicit_num_events + num_events;
	cl_event * evs = malloc(sizeof(cl_event) * ndeps);
	memcpy(evs, implicit_events, sizeof(cl_event) * implicit_num_events);
	memcpy(&evs[implicit_num_events], events, sizeof(cl_event) * num_events);

	*ret_num_events = ndeps;
	*ret_events = evs;
}

/**
 * Enqueue the given task and put ev into the command queue.
 */
void command_queue_enqueue(
	cl_command_queue cq, 		/* Command queue */
	cl_event ev,			/* Event triggered on task completion (can be NULL if task event should be used)*/
	cl_int is_barrier,			/* True if the task acts as a barrier */
	cl_int num_events,		/* Number of dependencies */
	const cl_event * events,	/* Dependencies */
	cl_int * ret_num_events,	/* Returned number of events */
	cl_event ** ret_events		/* Returned events */
	) {

	/* Lock command queue */
	pthread_spin_lock(&cq->spin);

	command_queue_dependencies(cq, is_barrier, num_events, events, ret_num_events, ret_events);

	command_queue_insert(cq, ev, is_barrier);

	/* Unlock command queue */
	pthread_spin_unlock(&cq->spin);
}


cl_event command_queue_barrier(cl_command_queue cq) {

	cl_int ndeps;
	cl_event *deps;

	//CL_COMMAND_MARKER has been chosen as CL_COMMAND_BARRIER doesn't exist
	starpu_task * task = task_create(CL_COMMAND_MARKER);

	DEBUG_MSG("Submitting barrier task (event %d)\n", task->tag_id);
	command_queue_enqueue(cq, task_event(task), 1, 0, NULL, &ndeps, &deps);

	task_submit(task, ndeps, deps);

	return task_event(task);
}
