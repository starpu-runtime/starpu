/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Marc Sergent
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

#include <starpu_sched_node.h>
#include <starpu_scheduler.h>

static int eager_push_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(node && task && starpu_sched_node_is_eager(node));
	STARPU_ASSERT(starpu_sched_node_can_execute_task(node,task));
	
	int workerid;
	for(workerid = starpu_bitmap_first(node->workers_in_ctx);
	    workerid != -1;
	    workerid = starpu_bitmap_next(node->workers_in_ctx, workerid))
	{
		int nimpl;
		for(nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if(starpu_worker_can_execute_task(workerid,task,nimpl)
			   || starpu_combined_worker_can_execute_task(workerid, task, nimpl))
			{
				int i;
				for (i = 0; i < node->nchilds; i++)
				{
					int idworker;
					for(idworker = starpu_bitmap_first(node->childs[i]->workers);
						idworker != -1;
						idworker = starpu_bitmap_next(node->childs[i]->workers, idworker))
					{
						if (idworker == workerid)
						{
							if(starpu_sched_node_is_worker(node->childs[i]))
							{
								node->childs[i]->avail(node->childs[i]);
								return 1;
							}
							else
							{
								int ret = node->childs[i]->push_task(node->childs[i],task);
								if(!ret)
									return ret;
							}
						}
					}
				}
			}
		}
	}
	return 1;
}

int starpu_sched_node_is_eager(struct starpu_sched_node * node)
{
	return node->push_task == eager_push_task;
}

struct starpu_sched_node * starpu_sched_node_eager_create(void * ARG STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_node * node = starpu_sched_node_create();
	node->push_task = eager_push_task;

	return node;
}
