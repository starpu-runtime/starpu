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

#ifndef SOCL_COMMAND_QUEUE_H
#define SOCl_COMMAND_QUEUE_H

void command_queue_enqueue(
	cl_command_queue cq, 		/* Command queue */
	cl_event ev,			/* Event triggered on task completion (can be NULL if task event should be used)*/
	cl_int is_barrier,			/* True if the task acts as a barrier */
	cl_int num_events,		/* Number of dependencies */
	const cl_event * events,	/* Dependencies */
	cl_int * ret_num_events,	/* Returned number of events */
	cl_event ** ret_events		/* Returned events */
	);

cl_event command_queue_barrier(cl_command_queue cq);

#endif /* SOCl_COMMAND_QUEUE_H */
