/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Simon Archipoff
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
#include <starpu_task_list.h>


struct _starpu_prio_list
{
	int prio;
	struct starpu_task_list list;
};

struct _starpu_prio_deque
{
	struct _starpu_prio_list * array;
	int size_array;
	unsigned ntasks;
	unsigned nprocessed;
	double exp_start, exp_end, exp_len;
};

void _starpu_prio_deque_init(struct _starpu_prio_deque *);
void _starpu_prio_deque_destroy(struct _starpu_prio_deque *);

/* return 0 iff the struct _starpu_prio_deque is not empty */
int _starpu_prio_deque_is_empty(struct _starpu_prio_deque *);

/* push a task in O(nb priorities) */
int _starpu_prio_deque_push_task(struct _starpu_prio_deque *, struct starpu_task *);


/* all _starpu_prio_deque_pop/deque_task function return a task or a NULL pointer if none are available
 * in O(nb priorities)
 */

struct starpu_task * _starpu_prio_deque_pop_task(struct _starpu_prio_deque *);

/* return a task that can be executed by workerid
 */
struct starpu_task * _starpu_prio_deque_pop_task_for_worker(struct _starpu_prio_deque *, int workerid);

/* deque a task of the higher priority available */
struct starpu_task * _starpu_prio_deque_deque_task(struct _starpu_prio_deque *);
/* return a task that can be executed by workerid
 */
struct starpu_task * _starpu_prio_deque_deque_task_for_worker(struct _starpu_prio_deque *, int workerid);

#endif /* __PRIO_DEQUE_H__ */
