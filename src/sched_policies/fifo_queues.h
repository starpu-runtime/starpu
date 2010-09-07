/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

/* FIFO queues, ready for use by schedulers */

#ifndef __FIFO_QUEUES_H__
#define __FIFO_QUEUES_H__

#include <starpu.h>
#include <common/config.h>

struct starpu_fifo_taskq_s {
	/* the actual list */
	struct starpu_task_list taskq;

	/* the number of tasks currently in the queue */
	unsigned ntasks;

	/* the number of tasks that were processed */
	unsigned nprocessed;

	/* only meaningful if the queue is only used by a single worker */
	double exp_start;
	double exp_end;
	double exp_len;
};

struct starpu_fifo_taskq_s*_starpu_create_fifo(void);
void _starpu_destroy_fifo(struct starpu_fifo_taskq_s *fifo);

int _starpu_fifo_push_task(struct starpu_fifo_taskq_s *fifo, pthread_mutex_t *sched_mutex, pthread_cond_t *sched_cond, struct starpu_task *task);
int _starpu_fifo_push_prio_task(struct starpu_fifo_taskq_s *fifo, pthread_mutex_t *sched_mutex, pthread_cond_t *sched_cond, struct starpu_task *task);

struct starpu_task *_starpu_fifo_pop_task(struct starpu_fifo_taskq_s *fifo);
struct starpu_task *_starpu_fifo_pop_every_task(struct starpu_fifo_taskq_s *fifo, pthread_mutex_t *sched_mutex, uint32_t where);

#endif // __FIFO_QUEUES_H__
