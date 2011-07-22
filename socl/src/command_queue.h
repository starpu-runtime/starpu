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

cl_int command_queue_enqueue(cl_command_queue cq, starpu_task *task, cl_int barrier, cl_int num_events, const cl_event * events);

cl_int command_queue_enqueue_fakeevent(cl_command_queue cq, starpu_task *task, cl_int barrier, cl_int num_events, const cl_event * events, cl_event fake_event);

cl_event enqueueBarrier(cl_command_queue cq);

#endif /* SOCl_COMMAND_QUEUE_H */
