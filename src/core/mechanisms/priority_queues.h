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

#ifndef __PRIORITY_QUEUES_H__
#define __PRIORITY_QUEUES_H__

#include <starpu.h>
#include <common/config.h>
#include <core/mechanisms/queues.h>

#define NPRIO_LEVELS	((STARPU_MAX_PRIO) - (STARPU_MIN_PRIO) + 1)

struct starpu_priority_jobq_s {
	/* the actual lists 
	 *	jobq[p] is for priority [p - STARPU_MIN_PRIO] */
	starpu_job_list_t jobq[NPRIO_LEVELS];
	unsigned njobs[NPRIO_LEVELS];

	unsigned total_njobs;
};

struct starpu_jobq_s *_starpu_create_priority_jobq(void);
void _starpu_destroy_priority_jobq(struct starpu_jobq_s *jobq);

void _starpu_init_priority_queues_mechanisms(void);

#endif // __PRIORITY_QUEUES_H__
