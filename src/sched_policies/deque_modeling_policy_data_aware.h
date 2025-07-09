/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2024  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DEQUE_MODELING_POLICY_DATA_AWARE_H__
#define __DEQUE_MODELING_POLICY_DATA_AWARE_H__

#include <starpu.h>
#include <sched_policies/fifo_queues.h>

#pragma GCC visibility push(hidden)

struct _starpu_dmda_data
{
	double alpha;
	double beta;
	double _gamma;
	double idle_power;

	struct starpu_st_fifo_taskq queue_array[STARPU_NMAXWORKERS];

	long int total_task_cnt;
	long int ready_task_cnt;
	long int eager_task_cnt; /* number of tasks scheduled without model */
	int num_priorities;
	int num_levels_of_tasks;
};

#pragma GCC visibility pop

#endif
