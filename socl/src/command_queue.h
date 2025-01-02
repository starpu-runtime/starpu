/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#define SOCL_COMMAND_QUEUE_H

void command_queue_enqueue_ex(cl_command_queue 	cq,		/* Command queue */
			      cl_command		cmd,		/* Command to enqueue */
			      cl_uint			num_events,	/* Number of explicit dependencies */
			      const cl_event *	events		/* Explicit dependencies */
			      );

#define command_queue_enqueue(cq, cmd, num_events, events)\
	command_queue_enqueue_ex(cq, (cl_command)cmd, num_events, events)

#endif /* SOCL_COMMAND_QUEUE_H */
