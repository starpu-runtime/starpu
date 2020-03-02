/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include "sched_ctx_list.h"

struct _starpu_sched_ctx_elt* _starpu_sched_ctx_elt_find(struct _starpu_sched_ctx_list *list,
							 unsigned sched_ctx)
{
	struct _starpu_sched_ctx_list *l = NULL;
	struct _starpu_sched_ctx_elt *e = NULL;
	unsigned found = 0;

	for (l = list; l && !found; l=l->next)
	{
		e=l->head; //Go in a circle once before stopping
		do
		{
			if (e->sched_ctx == sched_ctx)
			{
				found = 1;
				break;
			}
			e = e->next;
		}
		while (e != l->head);
	}

	return found ? e : NULL;
}

void _starpu_sched_ctx_elt_init(struct _starpu_sched_ctx_elt *elt, unsigned sched_ctx)
{
	elt->sched_ctx = sched_ctx;
	elt->task_number = 0;
	elt->last_poped = 0;
	elt->parent = NULL;
	elt->next = NULL;
	elt->prev = NULL;
}

void _starpu_sched_ctx_elt_ensure_consistency(struct _starpu_sched_ctx_list *list,
					      unsigned sched_ctx)
{
	struct _starpu_sched_ctx_elt *elt = _starpu_sched_ctx_elt_find(list, sched_ctx);
	if (elt && elt->task_number>0)
		elt->task_number = 0;
}

/* Adds a new element after the head of the given list. */
struct _starpu_sched_ctx_elt* _starpu_sched_ctx_elt_add_after(struct _starpu_sched_ctx_list *list,
							      unsigned sched_ctx)
{
	struct _starpu_sched_ctx_elt *head, *next;
	struct _starpu_sched_ctx_elt *elt;
	_STARPU_MALLOC(elt, sizeof(struct _starpu_sched_ctx_elt));

	_starpu_sched_ctx_elt_init(elt, sched_ctx);
	elt->parent = list;

	head = list->head;
	if (head != NULL)
	{
		next = head->next;
		head->next = elt;
		elt->prev = head;

		/** We know next != NULL since it is at least head **/
		elt->next = next;
		next->prev = elt;
	}
	else
	{
		elt->next = elt;
		elt->prev = elt;
		list->head = elt;
	}

	return elt;
}

/* Adds a new element before the head of the given list. */
struct _starpu_sched_ctx_elt* _starpu_sched_ctx_elt_add_before(struct _starpu_sched_ctx_list *list,
							       unsigned sched_ctx)
{
	struct _starpu_sched_ctx_elt *head, *prev;
	struct _starpu_sched_ctx_elt *elt;
	_STARPU_MALLOC(elt, sizeof(struct _starpu_sched_ctx_elt));

	_starpu_sched_ctx_elt_init(elt, sched_ctx);
	elt->parent = list;

	head = list->head;
	if (head != NULL)
	{
		prev = head->prev;
		head->prev = elt;
		elt->next = head;

		elt->prev = prev;
		prev->next = elt;
	}
	else
	{
		elt->next = elt;
		elt->prev = elt;
		list->head = elt;
	}
	return elt;
}

struct _starpu_sched_ctx_elt* _starpu_sched_ctx_elt_add(struct _starpu_sched_ctx_list *list,
							unsigned sched_ctx)
{
	return _starpu_sched_ctx_elt_add_after(list, sched_ctx);
}

/* Remove elt from list */
void _starpu_sched_ctx_elt_remove(struct _starpu_sched_ctx_list *list,
				 struct _starpu_sched_ctx_elt *elt)
{
	elt->prev->next = elt->next;
	elt->next->prev = elt->prev;

	if (elt->next == elt) //singleton
		list->head = NULL;
	else if (elt->next != elt && list->head == elt)
		list->head = elt->next;

	free(elt);
	return;
}

int _starpu_sched_ctx_elt_exists(struct _starpu_sched_ctx_list *list,
				 unsigned sched_ctx)
{
	struct _starpu_sched_ctx_elt *e;
	e = _starpu_sched_ctx_elt_find(list, sched_ctx);
	return (e == NULL) ? 0 : 1;
}

int _starpu_sched_ctx_elt_get_priority(struct _starpu_sched_ctx_list *list,
				       unsigned sched_ctx)
{
	struct _starpu_sched_ctx_elt *e;
	e = _starpu_sched_ctx_elt_find(list, sched_ctx);
	return (e == NULL) ? 0 : e->parent->priority;
}

struct _starpu_sched_ctx_list* _starpu_sched_ctx_list_find(struct _starpu_sched_ctx_list *list,
							   unsigned prio)
{
	struct _starpu_sched_ctx_list *l = NULL;

	for (l = list; l != NULL ; l=l->next)
	{
		if (l->priority == prio)
			break;
	}

	return l;
}

/* Adds sched_ctx in a priority list. We consider that we don't add two times
 * the same sched_ctx. Returns head of list. */
struct _starpu_sched_ctx_elt* _starpu_sched_ctx_list_add_prio(struct _starpu_sched_ctx_list **list,
							      unsigned prio, unsigned sched_ctx)
{
	struct _starpu_sched_ctx_list *parent_list = NULL, *prev = NULL, *last = NULL;
	struct _starpu_sched_ctx_list *l;

	for (l = *list; l != NULL; l=l->next)
	{
		if (l->priority <= prio)
			break;
		last = l;
	}

	if (l != NULL && l->priority == prio)
	{
		parent_list = l;
	}
	else //l's priority is inferior or inexistant, add before
	{
		_STARPU_MALLOC(parent_list, sizeof(struct _starpu_sched_ctx_list));
		parent_list->priority = prio;
		parent_list->next = l;
		parent_list->head = NULL;
		parent_list->prev = NULL;
		if (l != NULL)
		{
			prev = l->prev;
			l->prev = parent_list;
			if (prev != NULL)
			{
				prev->next = parent_list;
				parent_list->prev = prev;
			}
			else
			{
				*list = parent_list;
			}
		}
		else
		{
			if (last == NULL)
			{
				*list = parent_list;
			}
			else
			{
				last->next = parent_list;
				parent_list->prev = last;
			}
		}
	}

	return _starpu_sched_ctx_elt_add(parent_list, sched_ctx);
}

int _starpu_sched_ctx_list_add(struct _starpu_sched_ctx_list **list,
			       unsigned sched_ctx)
{
	return _starpu_sched_ctx_list_add_prio(list, 0, sched_ctx) != NULL ? 0 : -1;
}

void _starpu_sched_ctx_list_remove_elt(struct _starpu_sched_ctx_list **list,
				      struct _starpu_sched_ctx_elt *rm)
{
	struct _starpu_sched_ctx_list *parent;

	parent = rm->parent;

	_starpu_sched_ctx_elt_remove(parent, rm);

	/* Automatically clean up useless prio list */
	if (parent->head == NULL)
	{
		if (parent->prev == NULL)
		{
			*list = parent->next;
			if (parent->next != NULL)
				parent->next->prev = NULL;
		}
		else
		{
			parent->prev->next = parent->next;
			parent->next->prev = parent->prev;
		}
		free(parent);
		parent = NULL;
	}
	return;
}

/* Searches for a context and remove it */
int _starpu_sched_ctx_list_remove(struct _starpu_sched_ctx_list **list,
				  unsigned sched_ctx)
{
	struct _starpu_sched_ctx_elt *rm;
	rm = _starpu_sched_ctx_elt_find(*list, sched_ctx);

	if (rm == NULL)
		return -1;

	_starpu_sched_ctx_list_remove_elt(list, rm);
	return 0;
}

int _starpu_sched_ctx_list_move(struct _starpu_sched_ctx_list **list,
				unsigned sched_ctx, unsigned prio_to)
{
	struct _starpu_sched_ctx_elt *elt = _starpu_sched_ctx_elt_find(*list, sched_ctx);
	long task_number = 0;
	if (elt == NULL)
		return -1;

	task_number = elt->task_number;
	_starpu_sched_ctx_list_remove_elt(list, elt);
	elt = _starpu_sched_ctx_list_add_prio(list, prio_to, sched_ctx);
	elt->task_number = task_number;

	return 0;
}

int _starpu_sched_ctx_list_exists(struct _starpu_sched_ctx_list *list,
					   unsigned prio)
{
	struct _starpu_sched_ctx_list *l;
	l = _starpu_sched_ctx_list_find(list, prio);
	return ((l == NULL && list->priority == prio) || l != NULL) ? 1 : 0;
}

void _starpu_sched_ctx_list_remove_all(struct _starpu_sched_ctx_list *list)
{
	while (list->head != NULL)
		_starpu_sched_ctx_elt_remove(list, list->head);

	free(list);
}

void _starpu_sched_ctx_list_delete(struct _starpu_sched_ctx_list **list)
{
	while(*list)
	{
		struct _starpu_sched_ctx_list *next = (*list)->next;
		_starpu_sched_ctx_list_remove_all(*list);
		*list = NULL;
		if(next)
			*list = next;
	}
}

int _starpu_sched_ctx_list_iterator_init(struct _starpu_sched_ctx_list *list,
					 struct _starpu_sched_ctx_list_iterator *it)
{
	it->list_head = list;
	it->cursor = NULL;

	return 0;
}

int _starpu_sched_ctx_list_iterator_has_next(struct _starpu_sched_ctx_list_iterator *it)
{
	if (it->cursor == NULL)
	{
		if (it->list_head != NULL)
			return it->list_head->head != NULL;
		else
			return 0;
	}
	else
	{
		struct _starpu_sched_ctx_list *parent = it->cursor->parent;
		if (it->cursor->next == parent->head)
			return parent->next != NULL;
	}

	return 1;
}

struct _starpu_sched_ctx_elt* _starpu_sched_ctx_list_iterator_get_next(struct _starpu_sched_ctx_list_iterator *it)
{
	struct _starpu_sched_ctx_elt *ret=NULL, *current;
	struct _starpu_sched_ctx_list *parent;
	current = it->cursor;

	if (current != NULL)
	{
		parent = it->cursor->parent;
		if (current->next == parent->head)
		{
			if (parent->next != NULL)
			{
				it->cursor = parent->next->head;
				ret = it->cursor;
			}
			else
			{
				/* if everything fails (e.g. worker removed from ctx since related has_next call)
				   just return head, it'll save us a synchro */
				it->cursor = NULL;
				ret = it->list_head->head;
			}
		}
		else
		{
			it->cursor = current->next;
			ret = it->cursor;
		}
	}
	else
	{
		it->cursor = it->list_head->head;
		ret = it->cursor;
	}

	return ret;
}

int _starpu_sched_ctx_list_push_event(struct _starpu_sched_ctx_list *list, unsigned sched_ctx)
{
	struct _starpu_sched_ctx_elt *elt = _starpu_sched_ctx_elt_find(list, sched_ctx);
	if (elt == NULL)
		return -1;

	elt->task_number++;

	return 0;
}


int _starpu_sched_ctx_list_pop_event(struct _starpu_sched_ctx_list *list, unsigned sched_ctx)
{
	struct _starpu_sched_ctx_elt *elt = _starpu_sched_ctx_elt_find(list, sched_ctx);
	if (elt == NULL)
		return -1;

	elt->task_number--;

	/** Balance circular lists **/
	elt->parent->head = elt->next;

	return 0;
}

int _starpu_sched_ctx_list_pop_all_event(struct _starpu_sched_ctx_list *list, unsigned sched_ctx)
{
	struct _starpu_sched_ctx_elt *elt = _starpu_sched_ctx_elt_find(list, sched_ctx);
	if (elt == NULL)
		return -1;

	elt->task_number = 0;

	/** Balance circular lists **/
	elt->parent->head = elt->next;

	return 0;
}
