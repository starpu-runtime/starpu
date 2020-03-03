/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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


#include "sc_hypervisor_policy.h"

void sc_hypervisor_policy_add_task_to_pool(struct starpu_codelet *cl, unsigned sched_ctx, uint32_t footprint, struct sc_hypervisor_policy_task_pool **task_pools, size_t data_size)
{
	struct sc_hypervisor_policy_task_pool *tp = NULL;

	for (tp = *task_pools; tp; tp = tp->next)
	{
		if (tp && tp->cl == cl && tp->footprint == footprint && tp->sched_ctx_id == sched_ctx)
			break;
	}

	if (!tp)
	{
		tp = (struct sc_hypervisor_policy_task_pool *) malloc(sizeof(struct sc_hypervisor_policy_task_pool));
		tp->cl = cl;
		tp->footprint = footprint;
		tp->sched_ctx_id = sched_ctx;
		tp->n = 0;
		tp->next = *task_pools;
		tp->data_size = data_size;
		*task_pools = tp;
	}

	/* One more task of this kind */
	tp->n++;
}

void sc_hypervisor_policy_remove_task_from_pool(struct starpu_task *task, uint32_t footprint, struct sc_hypervisor_policy_task_pool **task_pools)
{
	/* count the tasks of the same type */
	struct sc_hypervisor_policy_task_pool *tp = NULL;

	for (tp = *task_pools; tp; tp = tp->next)
	{
		if (tp && tp->cl == task->cl && tp->footprint == footprint && tp->sched_ctx_id == task->sched_ctx)
			break;
	}

	if (tp)
	{
		if(tp->n > 1)
			tp->n--;
		else
		{
			if(tp == *task_pools)
			{
				struct sc_hypervisor_policy_task_pool *next_tp = NULL;
				if((*task_pools)->next)
					next_tp = (*task_pools)->next;

				free(tp);
				tp = NULL;
				
				*task_pools = next_tp;
				
			}
			else
			{
				struct sc_hypervisor_policy_task_pool *prev_tp = NULL;
				for (prev_tp = *task_pools; prev_tp; prev_tp = prev_tp->next)
				{
					if (prev_tp->next == tp)
						prev_tp->next = tp->next;
				}
				
				free(tp);
				tp = NULL;
			}
		}
	}
}

struct sc_hypervisor_policy_task_pool* sc_hypervisor_policy_clone_task_pool(struct sc_hypervisor_policy_task_pool *tp)
{
	if(tp == NULL) return NULL;

	struct sc_hypervisor_policy_task_pool *tmp_tp = (struct sc_hypervisor_policy_task_pool*)malloc(sizeof(struct sc_hypervisor_policy_task_pool));
	memcpy(tmp_tp, tp, sizeof(struct sc_hypervisor_policy_task_pool));
	tmp_tp->next = sc_hypervisor_policy_clone_task_pool(tp->next);
	return tmp_tp;
}
