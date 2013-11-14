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

#include <core/workers.h>

#include "prio_deque.h"


void _starpu_prio_deque_init(struct _starpu_prio_deque * pdeque)
{
	STARPU_ASSERT(pdeque);
	memset(pdeque,0,sizeof(*pdeque));
}
void _starpu_prio_deque_destroy(struct _starpu_prio_deque * pdeque)
{
	STARPU_ASSERT(pdeque);
	int i;
	for(i = 0; i < pdeque->size_array; i++)
	{
		STARPU_ASSERT(starpu_task_list_empty(&pdeque->array[i].list));
	}
	free(pdeque->array);
}

int _starpu_prio_deque_is_empty(struct _starpu_prio_deque * pdeque)
{
	STARPU_ASSERT(pdeque);
	return pdeque->ntasks == 0;
}


/* return the struct starpu_prio_list * of prio,
 * create it and return it if none exist yet
 */
static struct starpu_task_list * get_prio(struct _starpu_prio_deque * pdeque, int prio)
{
	STARPU_ASSERT(pdeque);
	int i;
	for(i = 0; i < pdeque->size_array; i++)
	{
		if(pdeque->array[i].prio == prio)
		{
			return &pdeque->array[i].list;
		}
		else
			if(pdeque->array[i].prio < prio)
				break;
	}
	pdeque->size_array++;
	pdeque->array = realloc(pdeque->array, sizeof(struct _starpu_prio_list) * (pdeque->size_array));
	memmove(pdeque->array + i + 1,
		pdeque->array + i,
		(pdeque->size_array - i - 1) * sizeof(struct _starpu_prio_list));
	pdeque->array[i].prio = prio;
	starpu_task_list_init(&pdeque->array[i].list);
	return &pdeque->array[i].list;
}



int _starpu_prio_deque_push_task(struct _starpu_prio_deque * pdeque, struct starpu_task * task)
{
	STARPU_ASSERT(pdeque && task);
	struct starpu_task_list * list = get_prio(pdeque, task->priority);
	starpu_task_list_push_back(list, task);
	pdeque->ntasks++;
	return 0;
}

int _starpu_prio_deque_push_back_task(struct _starpu_prio_deque * pdeque, struct starpu_task * task)
{
	STARPU_ASSERT(pdeque && task);
	struct starpu_task_list * list = get_prio(pdeque, task->priority);
	starpu_task_list_push_front(list, task);
	pdeque->ntasks++;
	return 0;
}




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
			return 1;
	return 0;
}

#define REMOVE_TASK(pdeque, first_task_field, next_task_field, predicate, parg)	\
	{								\
		int i;							\
		struct starpu_task * t = NULL;				\
		for(i = 0; i < pdeque->size_array; i++)			\
		{							\
			t = pdeque->array[i].list.first_task_field;	\
			while(t && !predicate(t,parg))			\
				t = t->next_task_field;			\
			if(t)						\
			{						\
				starpu_task_list_erase(&pdeque->array[i].list, t); \
				pdeque->ntasks--;			\
				return t;				\
			}						\
		}							\
		return NULL;						\
	}

struct starpu_task * _starpu_prio_deque_pop_task(struct _starpu_prio_deque * pdeque)
{
	REMOVE_TASK(pdeque, head, prev, pred_true, STARPU_POISON_PTR);
}
struct starpu_task * _starpu_prio_deque_pop_task_for_worker(struct _starpu_prio_deque * pdeque, int workerid)
{
	STARPU_ASSERT(pdeque);
	STARPU_ASSERT(0 <= workerid && (unsigned) workerid < starpu_worker_get_count());
	REMOVE_TASK(pdeque, head, prev, pred_can_execute, &workerid);
}

/* deque a task of the higher priority available */
struct starpu_task * _starpu_prio_deque_deque_task(struct _starpu_prio_deque * pdeque)
{
	STARPU_ASSERT(pdeque);
	REMOVE_TASK(pdeque, tail, next, pred_true, STARPU_POISON_PTR);
}

struct starpu_task * _starpu_prio_deque_deque_task_for_worker(struct _starpu_prio_deque * pdeque, int workerid)
{
	STARPU_ASSERT(pdeque);
	STARPU_ASSERT(0 <= workerid && (unsigned) workerid < starpu_worker_get_count());
	REMOVE_TASK(pdeque, tail, next, pred_can_execute, &workerid);
}
