/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <schedulers/starpu_scheduler_toolbox.h>
#include <core/workers.h>
#include <sched_policies/prio_deque.h>
#include <sched_policies/fifo_queues.h>

void starpu_st_prio_deque_init(struct starpu_st_prio_deque *pdeque)
{
	memset(pdeque,0,sizeof(*pdeque));
	starpu_task_prio_list_init(&pdeque->list);
	STARPU_HG_DISABLE_CHECKING(pdeque->exp_start);
	STARPU_HG_DISABLE_CHECKING(pdeque->exp_end);
	STARPU_HG_DISABLE_CHECKING(pdeque->exp_len);
}

void starpu_st_prio_deque_destroy(struct starpu_st_prio_deque *pdeque)
{
	starpu_task_prio_list_deinit(&pdeque->list);
}

int starpu_st_prio_deque_is_empty(struct starpu_st_prio_deque *pdeque)
{
	return pdeque->ntasks == 0;
}

void starpu_st_prio_deque_erase(struct starpu_st_prio_deque *pdeque, struct starpu_task *task)
{
	starpu_task_prio_list_erase(&pdeque->list, task);
}

int starpu_st_prio_deque_push_front_task(struct starpu_st_prio_deque *pdeque, struct starpu_task *task)
{
	starpu_task_prio_list_push_front(&pdeque->list, task);
	pdeque->ntasks++;
	return 0;
}

int starpu_st_prio_deque_push_back_task(struct starpu_st_prio_deque *pdeque, struct starpu_task *task)
{
	starpu_task_prio_list_push_back(&pdeque->list, task);
	pdeque->ntasks++;
	return 0;
}

struct starpu_task *starpu_st_prio_deque_highest_task(struct starpu_st_prio_deque *pdeque)
{
	struct starpu_task *task;
	if (starpu_task_prio_list_empty(&pdeque->list))
		return NULL;
	task = starpu_task_prio_list_front_highest(&pdeque->list);
	return task;
}

struct starpu_task *starpu_st_prio_deque_pop_task(struct starpu_st_prio_deque *pdeque)
{
	struct starpu_task *task;
	if (starpu_task_prio_list_empty(&pdeque->list))
		return NULL;
	task = starpu_task_prio_list_pop_front_highest(&pdeque->list);
	pdeque->ntasks--;
	return task;
}

struct starpu_task *starpu_st_prio_deque_pop_back_task(struct starpu_st_prio_deque *pdeque)
{
	struct starpu_task *task;
	if (starpu_task_prio_list_empty(&pdeque->list))
		return NULL;
	task = starpu_task_prio_list_pop_back_lowest(&pdeque->list);
	pdeque->ntasks--;
	return task;
}

int starpu_st_prio_deque_pop_this_task(struct starpu_st_prio_deque *pdeque, int workerid, struct starpu_task *task)
{
	unsigned nimpl = 0;
#ifdef STARPU_DEBUG
	STARPU_ASSERT(starpu_task_prio_list_ismember(&pdeque->list, task));
#endif

	if (workerid < 0 || starpu_worker_can_execute_task_first_impl(workerid, task, &nimpl))
	{
		starpu_task_set_implementation(task, nimpl);
		starpu_task_prio_list_erase(&pdeque->list, task);
		pdeque->ntasks--;
		return 1;
	}

	return 0;
}

static inline int pred_true(struct starpu_task *t STARPU_ATTRIBUTE_UNUSED, void *v STARPU_ATTRIBUTE_UNUSED)
{
	(void)t;
	(void)v;
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

#define REMOVE_TASK(pdeque, first_task, next_task, predicate, parg)		\
	{									\
		struct starpu_task * t;						\
		if (skipped)							\
			*skipped = NULL;					\
		for (t  = starpu_task_prio_##first_task(&pdeque->list);		\
		     t != starpu_task_prio_list_end(&pdeque->list);		\
		     t  = starpu_task_prio_##next_task(&pdeque->list, t))	\
		{								\
			if (predicate(t, parg))					\
			{							\
				starpu_task_prio_list_erase(&pdeque->list, t);	\
				pdeque->ntasks--;				\
				return t;					\
			}							\
			else							\
				if (skipped)					\
					*skipped = t;				\
		}								\
		return NULL;							\
	}

struct starpu_task *starpu_st_prio_deque_pop_task_for_worker(struct starpu_st_prio_deque * pdeque, int workerid, struct starpu_task * *skipped)
{
	STARPU_ASSERT(pdeque);
	STARPU_ASSERT(workerid >= 0 && (unsigned) workerid < starpu_worker_get_count());
	REMOVE_TASK(pdeque, list_begin, list_next, pred_can_execute, &workerid);
}

struct starpu_task *starpu_st_prio_deque_deque_task_for_worker(struct starpu_st_prio_deque * pdeque, int workerid, struct starpu_task * *skipped)
{
	STARPU_ASSERT(pdeque);
	STARPU_ASSERT(workerid >= 0 && (unsigned) workerid < starpu_worker_get_count());
	REMOVE_TASK(pdeque, list_back_highest, list_prev_highest, pred_can_execute, &workerid);
}

struct starpu_task *starpu_st_prio_deque_deque_first_ready_task(struct starpu_st_prio_deque * pdeque, unsigned workerid)
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

		size_t non_ready_best = SIZE_MAX;
		size_t non_loading_best = SIZE_MAX;
		size_t non_allocated_best = SIZE_MAX;

		for (current = starpu_task_prio_list_begin(&pdeque->list);
		     current != starpu_task_prio_list_end(&pdeque->list);
		     current = starpu_task_prio_list_next(&pdeque->list, current))
		{
			int priority = current->priority;

			if (priority >= first_task_priority)
			{
				size_t non_ready, non_loading, non_allocated;
				starpu_st_non_ready_buffers_size(current, workerid, &non_ready, &non_loading, &non_allocated);
				if (non_ready < non_ready_best)
				{
					non_ready_best = non_ready;
					non_loading_best = non_loading;
					non_allocated_best = non_allocated;
					task = current;

					if (non_ready == 0 && non_allocated == 0)
						break;
				}
				else if (non_ready == non_ready_best)
				{
					if (non_loading < non_loading_best)
					{
						non_loading_best = non_loading;
						non_allocated_best = non_allocated;
						task = current;
					}
					else if (non_loading == non_loading_best)
					{
						if (non_allocated < non_allocated_best)
						{
							non_allocated_best = non_allocated;
							task = current;
						}
					}
				}
			}
		}

		starpu_task_prio_list_erase(&pdeque->list, task);
	}

	return task;
}

