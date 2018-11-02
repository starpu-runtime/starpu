/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2012,2014,2016,2017                 Université de Bordeaux
 * Copyright (C) 2011-2014,2017,2018                      CNRS
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

#ifndef __STARPU_TASK_LIST_H__
#define __STARPU_TASK_LIST_H__

#include <starpu_task.h>
#include <starpu_util.h>

#ifdef __cplusplus
extern "C"
{
#endif

	/* NOTE: this needs to have at least the same size as lists in src/common/list.h */
#ifdef BUILDING_STARPU
#define STARPU_TASK_LIST_INLINE extern inline
#else
struct starpu_task_list
{
	struct starpu_task *head;
	struct starpu_task *tail;
};
#define STARPU_TASK_LIST_INLINE extern
#endif

STARPU_TASK_LIST_INLINE
void starpu_task_list_init(struct starpu_task_list *list);

STARPU_TASK_LIST_INLINE
void starpu_task_list_push_front(struct starpu_task_list *list, struct starpu_task *task);

STARPU_TASK_LIST_INLINE
void starpu_task_list_push_back(struct starpu_task_list *list, struct starpu_task *task);

STARPU_TASK_LIST_INLINE
struct starpu_task *starpu_task_list_front(const struct starpu_task_list *list);

STARPU_TASK_LIST_INLINE
struct starpu_task *starpu_task_list_back(const struct starpu_task_list *list);

STARPU_TASK_LIST_INLINE
int starpu_task_list_empty(const struct starpu_task_list *list);

STARPU_TASK_LIST_INLINE
void starpu_task_list_erase(struct starpu_task_list *list, struct starpu_task *task);

STARPU_TASK_LIST_INLINE
struct starpu_task *starpu_task_list_pop_front(struct starpu_task_list *list);

STARPU_TASK_LIST_INLINE
struct starpu_task *starpu_task_list_pop_back(struct starpu_task_list *list);

STARPU_TASK_LIST_INLINE
struct starpu_task *starpu_task_list_begin(const struct starpu_task_list *list);

STARPU_TASK_LIST_INLINE
struct starpu_task *starpu_task_list_end(const struct starpu_task_list *list STARPU_ATTRIBUTE_UNUSED);

STARPU_TASK_LIST_INLINE
struct starpu_task *starpu_task_list_next(const struct starpu_task *task);

STARPU_TASK_LIST_INLINE
int starpu_task_list_ismember(const struct starpu_task_list *list, const struct starpu_task *look);

STARPU_TASK_LIST_INLINE
void starpu_task_list_move(struct starpu_task_list *ldst, struct starpu_task_list *lsrc);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TASK_LIST_H__ */
