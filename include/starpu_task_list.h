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

#ifndef __STARPU_TASK_LIST_H__
#define __STARPU_TASK_LIST_H__

#include <starpu_task.h>

struct starpu_task_list {
	struct starpu_task *head;
	struct starpu_task *tail;
};

/* Initialize a list structure */
void starpu_task_list_init(struct starpu_task_list *list);

/* Push a task at the front of a list */
void starpu_task_list_push_front(struct starpu_task_list *list,	struct starpu_task *task);

/* Push a task at the back of a list */
void starpu_task_list_push_back(struct starpu_task_list *list, struct starpu_task *task);

/* Get the front of the list (without removing it) */
struct starpu_task *starpu_task_list_front(struct starpu_task_list *list);

/* Get the back of the list (without removing it) */
struct starpu_task *starpu_task_list_back(struct starpu_task_list *list);

/* Test if a list is empty */
int starpu_task_list_empty(struct starpu_task_list *list);

/* Remove an element from the list */
void starpu_task_list_erase(struct starpu_task_list *list, struct starpu_task *task);

/* Remove the element at the front of the list */
struct starpu_task *starpu_task_list_pop_front(struct starpu_task_list *list);

/* Remove the element at the back of the list */
struct starpu_task *starpu_task_list_pop_back(struct starpu_task_list *list);
						
#endif // __STARPU_TASK_LIST_H__
