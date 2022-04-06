/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2016       Uppsala University
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

#ifndef __PRIO_DEQUE_H__
#define __PRIO_DEQUE_H__

#include <core/task.h>

/** @file */

struct starpu_st_prio_deque
{
	struct starpu_task_prio_list list;
	unsigned ntasks;
	unsigned nprocessed;
	// Assumptions:
	// exp_len is the sum of predicted_length + predicted_tansfer of all tasks in list
	// exp_start is the time at which the first task of list can start
	// exp_end = exp_start + exp_end
	// Careful: those are NOT maintained by the prio_queue operations
	double exp_start, exp_end, exp_len;
};



#endif /* __PRIO_DEQUE_H__ */
