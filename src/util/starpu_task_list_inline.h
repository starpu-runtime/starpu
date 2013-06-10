/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
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

#ifndef __STARPU_TASK_LIST_INLINE_H
#define __STARPU_TASK_LIST_INLINE_H

#include <starpu_task.h>

#ifndef STARPU_INLINE
#ifdef __GNUC_GNU_INLINE__
#define STARPU_INLINE extern inline
#else
#define STARPU_INLINE static inline
#endif
#endif

STARPU_INLINE
void starpu_task_list_init(struct starpu_task_list *list)
{
	list->head = NULL;
	list->tail = NULL;
}

STARPU_INLINE
void starpu_task_list_push_front(struct starpu_task_list *list,
				struct starpu_task *task)
{
	if (list->tail == NULL)
	{
		list->tail = task;
	}
	else
	{
		list->head->prev = task;
	}

	task->prev = NULL;
	task->next = list->head;
	list->head = task;
}

STARPU_INLINE
void starpu_task_list_push_back(struct starpu_task_list *list,
				struct starpu_task *task)
{
	if (list->head == NULL)
	{
		list->head = task;
	}
	else
	{
		list->tail->next = task;
	}

	task->next = NULL;
	task->prev = list->tail;
	list->tail = task;
}

STARPU_INLINE
struct starpu_task *starpu_task_list_front(struct starpu_task_list *list)
{
	return list->head;
}

STARPU_INLINE
struct starpu_task *starpu_task_list_back(struct starpu_task_list *list)
{
	return list->tail;
}

STARPU_INLINE
int starpu_task_list_empty(struct starpu_task_list *list)
{
	return (list->head == NULL);
}

STARPU_INLINE
void starpu_task_list_erase(struct starpu_task_list *list,
				struct starpu_task *task)
{
	struct starpu_task *p = task->prev;

	if (p)
	{
		p->next = task->next;
	}
	else
	{
		list->head = task->next;
	}

	if (task->next)
	{
		task->next->prev = p;
	}
	else
	{
		list->tail = p;
	}

	task->prev = NULL;
	task->next = NULL;
}

STARPU_INLINE
struct starpu_task *starpu_task_list_pop_front(struct starpu_task_list *list)
{
	struct starpu_task *task = list->head;

	if (task)
		starpu_task_list_erase(list, task);

	return task;
}

STARPU_INLINE
struct starpu_task *starpu_task_list_pop_back(struct starpu_task_list *list)
{
	struct starpu_task *task = list->tail;

	if (task)
		starpu_task_list_erase(list, task);

	return task;
}

STARPU_INLINE
struct starpu_task *starpu_task_list_begin(struct starpu_task_list *list)
{
	return list->head;
}

STARPU_INLINE
struct starpu_task *starpu_task_list_end(struct starpu_task_list *list STARPU_ATTRIBUTE_UNUSED)
{
	return NULL;
}

STARPU_INLINE
struct starpu_task *starpu_task_list_next(struct starpu_task *task)
{
	return task->next;
}
#endif /* __STARPU_TASK_LIST_INLINE_H */
