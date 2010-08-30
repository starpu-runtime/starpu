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

#include <starpu.h>
#include <common/config.h>
#include <core/mechanisms/priority_queues.h>
#include <common/utils.h>

/*
 * Centralized queue with priorities 
 */

struct starpu_jobq_s *_starpu_create_priority_jobq(void)
{
	struct starpu_jobq_s *q;

	q = malloc(sizeof(struct starpu_jobq_s));

	struct starpu_priority_jobq_s *central_queue;
	
	central_queue = malloc(sizeof(struct starpu_priority_jobq_s));
	q->queue = central_queue;

	central_queue->total_njobs = 0;

	unsigned prio;
	for (prio = 0; prio < NPRIO_LEVELS; prio++)
	{
		central_queue->jobq[prio] = starpu_job_list_new();
		central_queue->njobs[prio] = 0;
	}

	return q;
}

void _starpu_destroy_priority_jobq(struct starpu_jobq_s *jobq)
{
	struct starpu_priority_jobq_s *central_queue;

	central_queue = jobq->queue;

	unsigned prio;
	for (prio = 0; prio < NPRIO_LEVELS; prio++)
		starpu_job_list_delete(central_queue->jobq[prio]);

	free(central_queue);

	free(jobq);
}
