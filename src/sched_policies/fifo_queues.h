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

#ifndef __FIFO_QUEUES_H__
#define __FIFO_QUEUES_H__

#include <core/task.h>

/** @file */

struct starpu_st_fifo_taskq
{
	/** the actual list */
	struct starpu_task_list taskq;

	/** the number of tasks currently in the queue */
	unsigned ntasks;

	/** the number of tasks already pushed to the worker */
	unsigned pipeline_ntasks;

	/** the number of tasks currently in the queue corresponding to each priority */
	unsigned *ntasks_per_priority;

	/** the number of tasks that were processed */
	unsigned nprocessed;

	/** only meaningful if the queue is only used by a single worker */
	double exp_start; /** Expected start date of next item to do in the
			   * queue (i.e. not started yet). This is thus updated
			   * when we start it. */
	double exp_end; /** Expected end date of last task in the queue */
	double exp_len; /** Expected duration of the set of tasks in the queue */
	double *exp_len_per_priority; /** Expected duration of the set of tasks in the queue corresponding to each priority */
	double pipeline_len; /** the expected duration of what is already pushed to the worker */
};


#endif /* __FIFO_QUEUES_H__ */
