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

CL_API_SUFFIX__VERSION_1_0
CL_API_ENTRY cl_int CL_API_CALL
soclEnqueueBarrier(cl_command_queue cq)
{
	command_barrier cmd = command_barrier_create();

	command_queue_enqueue(cq, cmd, 0, NULL);

	return CL_SUCCESS;
}

cl_int command_barrier_submit(command_barrier cmd)
{
	struct starpu_task *task;
	task = task_create(CL_COMMAND_BARRIER);

	return task_submit(task, cmd);
}
