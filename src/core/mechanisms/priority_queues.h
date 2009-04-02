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

#ifndef __PRIORITY_QUEUES_H__
#define __PRIORITY_QUEUES_H__

#define MIN_PRIO	(-4)
#define MAX_PRIO	5

#define NPRIO_LEVELS	((MAX_PRIO) - (MIN_PRIO) + 1)

#include <core/mechanisms/queues.h>

struct priority_jobq_s {
	/* the actual lists 
	 *	jobq[p] is for priority [p - MIN_PRIO] */
	job_list_t jobq[NPRIO_LEVELS];
	unsigned njobs[NPRIO_LEVELS];

	unsigned total_njobs;
};

struct jobq_s *create_priority_jobq(void);
void init_priority_queues_mechanisms(void);

int priority_push_task(struct jobq_s *q, job_t task);

job_t priority_pop_task(struct jobq_s *q);

#endif // __PRIORITY_QUEUES_H__
