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
#ifndef __PRIO_DEQUE_H__
#define __PRIO_DEQUE_H__
#include <starpu.h>
#include <starpu_scheduler.h>
#include <core/task.h>

/** @file */

struct _starpu_prio_deque
{
	struct starpu_task_prio_list list;
	unsigned ntasks;
	unsigned nprocessed;
	// Assumptions:
	// exp_len is the sum of predicted_length + predicted_tansfer of all tasks in list
	// exp_start is the time at which the first task of list can start
	// exp_end = exp_start + exp_end
	// Careful: those are NOT maintained by the prio_queue operations
	double exp_start, exp_end, exp_len;
};

static inline void _starpu_prio_deque_init(struct _starpu_prio_deque *pdeque)
{
	memset(pdeque,0,sizeof(*pdeque));
	starpu_task_prio_list_init(&pdeque->list);
}

static inline void _starpu_prio_deque_destroy(struct _starpu_prio_deque *pdeque)
{
	starpu_task_prio_list_deinit(&pdeque->list);
}

/** return 0 iff the struct _starpu_prio_deque is not empty */
static inline int _starpu_prio_deque_is_empty(struct _starpu_prio_deque *pdeque)
{
	return pdeque->ntasks == 0;
}

static inline void _starpu_prio_deque_erase(struct _starpu_prio_deque *pdeque, struct starpu_task *task)
{
	starpu_task_prio_list_erase(&pdeque->list, task);
}

/** push a task in O(lg(nb priorities)) */
static inline int _starpu_prio_deque_push_front_task(struct _starpu_prio_deque *pdeque, struct starpu_task *task)
{
	starpu_task_prio_list_push_front(&pdeque->list, task);
	pdeque->ntasks++;
	return 0;
}
static inline int _starpu_prio_deque_push_back_task(struct _starpu_prio_deque *pdeque, struct starpu_task *task)
{
	starpu_task_prio_list_push_back(&pdeque->list, task);
	pdeque->ntasks++;
	return 0;
}
int _starpu_prio_deque_push_back_task(struct _starpu_prio_deque *, struct starpu_task *);


static inline struct starpu_task * _starpu_prio_deque_highest_task(struct _starpu_prio_deque *pdeque)
{
	struct starpu_task *task;
	if (starpu_task_prio_list_empty(&pdeque->list))
		return NULL;
	task = starpu_task_prio_list_front_highest(&pdeque->list);
	return task;
}

/** all _starpu_prio_deque_pop/deque_task function return a task or a NULL pointer if none are available
 * in O(lg(nb priorities))
 */

static inline struct starpu_task * _starpu_prio_deque_pop_task(struct _starpu_prio_deque *pdeque)
{
	struct starpu_task *task;
	if (starpu_task_prio_list_empty(&pdeque->list))
		return NULL;
	task = starpu_task_prio_list_pop_front_highest(&pdeque->list);
	pdeque->ntasks--;
	return task;
}

static inline struct starpu_task * _starpu_prio_deque_pop_back_task(struct _starpu_prio_deque *pdeque)
{
	struct starpu_task *task;
	if (starpu_task_prio_list_empty(&pdeque->list))
		return NULL;
	task = starpu_task_prio_list_pop_back_lowest(&pdeque->list);
	pdeque->ntasks--;
	return task;
}

static inline int _starpu_prio_deque_pop_this_task(struct _starpu_prio_deque *pdeque, int workerid, struct starpu_task *task)
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

/** return a task that can be executed by workerid
 */
struct starpu_task * _starpu_prio_deque_pop_task_for_worker(struct _starpu_prio_deque *, int workerid, int *skipped);

/** return a task that can be executed by workerid
 */
struct starpu_task * _starpu_prio_deque_deque_task_for_worker(struct _starpu_prio_deque *, int workerid, int *skipped);

struct starpu_task *_starpu_prio_deque_deque_first_ready_task(struct _starpu_prio_deque *, unsigned workerid);

#endif /* __PRIO_DEQUE_H__ */
