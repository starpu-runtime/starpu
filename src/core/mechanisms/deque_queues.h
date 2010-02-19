/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#ifndef __DEQUE_QUEUES_H__
#define __DEQUE_QUEUES_H__

#include <core/mechanisms/queues.h>

struct deque_jobq_s {
	/* the actual list */
	starpu_job_list_t jobq;

	/* the number of tasks currently in the queue */
	unsigned njobs;

	/* the number of tasks that were processed */
	unsigned nprocessed;

	/* only meaningful if the queue is only used by a single worker */
	double exp_start;
	double exp_end;
	double exp_len;
};

struct jobq_s *create_deque(void);

int deque_push_task(struct jobq_s *q, starpu_job_t task);
int deque_push_prio_task(struct jobq_s *q, starpu_job_t task);

starpu_job_t deque_pop_task(struct jobq_s *q);

void init_deque_queues_mechanisms(void);

unsigned get_total_njobs_deques(void);

unsigned get_deque_njobs(struct jobq_s *q);
unsigned get_deque_nprocessed(struct jobq_s *q);


#endif // __DEQUE_QUEUES_H__
