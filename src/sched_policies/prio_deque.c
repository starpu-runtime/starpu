/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Simon Archipoff
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

#include <core/workers.h>

#include "prio_deque.h"
#include "fifo_queues.h"


/* a little dirty code factorization */

static inline int pred_true(struct starpu_task * t STARPU_ATTRIBUTE_UNUSED, void * v STARPU_ATTRIBUTE_UNUSED)
{
	return 1;
}

static inline int pred_can_execute(struct starpu_task * t, void * pworkerid)
{
	int i;
	for(i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
		if(starpu_worker_can_execute_task(*(int*)pworkerid, t,i))
		{
			starpu_task_set_implementation(t, i);
			return 1;
		}
	return 0;
}

#define REMOVE_TASK(pdeque, first_task_field, next_task_field, predicate, parg)	\
	{									\
		struct starpu_task * t;						\
		for (t  = starpu_task_prio_list_begin(&pdeque->list);		\
		     t != starpu_task_prio_list_end(&pdeque->list);		\
		     t  = starpu_task_prio_list_next(&pdeque->list, t))		\
		{								\
			if (predicate(t, parg))					\
			{							\
				starpu_task_prio_list_erase(&pdeque->list, t);	\
				pdeque->ntasks--;				\
				return t;					\
			}							\
			else							\
				if (skipped)					\
					*skipped = 1;				\
		}								\
		return NULL;							\
	}

/* deque a task of the higher priority available */

/* From the front of the list for the highest priority */
struct starpu_task * _starpu_prio_deque_pop_task_for_worker(struct _starpu_prio_deque * pdeque, int workerid, int *skipped)
{
	STARPU_ASSERT(pdeque);
	STARPU_ASSERT(workerid >= 0 && (unsigned) workerid < starpu_worker_get_count());
	REMOVE_TASK(pdeque, _head, prev, pred_can_execute, &workerid);
}

/* From the back of the list for the highest priority */
struct starpu_task * _starpu_prio_deque_deque_task_for_worker(struct _starpu_prio_deque * pdeque, int workerid, int *skipped)
{
	STARPU_ASSERT(pdeque);
	STARPU_ASSERT(workerid >= 0 && (unsigned) workerid < starpu_worker_get_count());
	REMOVE_TASK(pdeque, _tail, next, pred_can_execute, &workerid);
}

struct starpu_task *_starpu_prio_deque_deque_first_ready_task(struct _starpu_prio_deque * pdeque, unsigned workerid)
{
	struct starpu_task *task = NULL, *current;

	if (starpu_task_prio_list_empty(&pdeque->list))
		return NULL;

	if (pdeque->ntasks > 0)
	{
		pdeque->ntasks--;

		task = starpu_task_prio_list_front_highest(&pdeque->list);
		if (STARPU_UNLIKELY(!task))
			return NULL;

		int first_task_priority = task->priority;
		int non_ready_best = INT_MAX;

		for (current = starpu_task_prio_list_begin(&pdeque->list);
		     current != starpu_task_prio_list_end(&pdeque->list);
		     current = starpu_task_prio_list_next(&pdeque->list, current))
		{
			int priority = current->priority;

			if (priority >= first_task_priority)
			{
				int non_ready = _starpu_count_non_ready_buffers(current, workerid);
				if (non_ready < non_ready_best)
				{
					non_ready_best = non_ready;
					task = current;

					if (non_ready == 0)
						break;
				}
			}
		}

		starpu_task_prio_list_erase(&pdeque->list, task);
	}

	return task;
}

