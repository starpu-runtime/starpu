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

struct starpu_priority_taskq_s *_starpu_create_priority_taskq(void)
{
	struct starpu_priority_taskq_s *central_queue;
	
	central_queue = malloc(sizeof(struct starpu_priority_taskq_s));
	central_queue->total_ntasks = 0;

	unsigned prio;
	for (prio = 0; prio < NPRIO_LEVELS; prio++)
	{
		starpu_task_list_init(&central_queue->taskq[prio]);
		central_queue->ntasks[prio] = 0;
	}

	return central_queue;
}

void _starpu_destroy_priority_taskq(struct starpu_priority_taskq_s *priority_queue)
{
	free(priority_queue);
}
