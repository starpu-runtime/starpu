/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __SCHED_CONTEXT_LIST_H__
#define __SCHED_CONTEXT_LIST_H__

/** @file */

/** Represents a non circular list of priorities and contains a list of sched context */
struct _starpu_sched_ctx_elt;
struct _starpu_sched_ctx_list
{
	struct _starpu_sched_ctx_list *prev;
	struct _starpu_sched_ctx_list *next;
	struct _starpu_sched_ctx_elt *head;
	unsigned priority;
};

/** Represents a circular list of sched context. */
struct _starpu_sched_ctx_elt
{
	struct _starpu_sched_ctx_elt *prev;
	struct _starpu_sched_ctx_elt *next;
	struct _starpu_sched_ctx_list *parent;
	unsigned sched_ctx;
	long task_number;
	unsigned last_poped;
};

struct _starpu_sched_ctx_list_iterator
{
	struct _starpu_sched_ctx_list *list_head;
	struct _starpu_sched_ctx_elt *cursor;
};

/** Element (sched_ctx) level operations */
struct _starpu_sched_ctx_elt* _starpu_sched_ctx_elt_find(struct _starpu_sched_ctx_list *list, unsigned sched_ctx);
void _starpu_sched_ctx_elt_ensure_consistency(struct _starpu_sched_ctx_list *list, unsigned sched_ctx);
void _starpu_sched_ctx_elt_init(struct _starpu_sched_ctx_elt *elt, unsigned sched_ctx);
struct _starpu_sched_ctx_elt* _starpu_sched_ctx_elt_add_after(struct _starpu_sched_ctx_list *list, unsigned sched_ctx);
struct _starpu_sched_ctx_elt* _starpu_sched_ctx_elt_add_before(struct _starpu_sched_ctx_list *list, unsigned sched_ctx);
struct _starpu_sched_ctx_elt* _starpu_sched_ctx_elt_add(struct _starpu_sched_ctx_list *list, unsigned sched_ctx);
void _starpu_sched_ctx_elt_remove(struct _starpu_sched_ctx_list *list, struct _starpu_sched_ctx_elt *elt);
int _starpu_sched_ctx_elt_exists(struct _starpu_sched_ctx_list *list, unsigned sched_ctx);
int _starpu_sched_ctx_elt_get_priority(struct _starpu_sched_ctx_list *list, unsigned sched_ctx);


/** List (priority) level operations */
struct _starpu_sched_ctx_list* _starpu_sched_ctx_list_find(struct _starpu_sched_ctx_list *list, unsigned prio);
struct _starpu_sched_ctx_elt* _starpu_sched_ctx_list_add_prio(struct _starpu_sched_ctx_list **list, unsigned prio, unsigned sched_ctx);
int _starpu_sched_ctx_list_add(struct _starpu_sched_ctx_list **list, unsigned sched_ctx);
void _starpu_sched_ctx_list_remove_elt(struct _starpu_sched_ctx_list **list, struct _starpu_sched_ctx_elt *rm);
int _starpu_sched_ctx_list_remove(struct _starpu_sched_ctx_list **list, unsigned sched_ctx);
int _starpu_sched_ctx_list_move(struct _starpu_sched_ctx_list **list, unsigned sched_ctx, unsigned prio_to);
int _starpu_sched_ctx_list_exists(struct _starpu_sched_ctx_list *list, unsigned prio);
void _starpu_sched_ctx_list_remove_all(struct _starpu_sched_ctx_list *list);
void _starpu_sched_ctx_list_delete(struct _starpu_sched_ctx_list **list);

/** Task number management */
int _starpu_sched_ctx_list_push_event(struct _starpu_sched_ctx_list *list, unsigned sched_ctx);
int _starpu_sched_ctx_list_pop_event(struct _starpu_sched_ctx_list *list, unsigned sched_ctx);
int _starpu_sched_ctx_list_pop_all_event(struct _starpu_sched_ctx_list *list, unsigned sched_ctx);

/** Iterator operations */
int _starpu_sched_ctx_list_iterator_init(struct _starpu_sched_ctx_list *list, struct _starpu_sched_ctx_list_iterator *it);
int _starpu_sched_ctx_list_iterator_has_next(struct _starpu_sched_ctx_list_iterator *it);
struct _starpu_sched_ctx_elt* _starpu_sched_ctx_list_iterator_get_next(struct _starpu_sched_ctx_list_iterator *it);

#endif // __SCHED_CONTEXT_H__
