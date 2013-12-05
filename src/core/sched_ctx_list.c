/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2013  INRIA
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

void _starpu_sched_ctx_list_init(struct _starpu_sched_ctx_list *list)
{
	list->next = NULL;
	list->sched_ctx = STARPU_NMAX_SCHED_CTXS;
}

void _starpu_sched_ctx_list_add(struct _starpu_sched_ctx_list **list, unsigned sched_ctx)
{
	if((*list)->sched_ctx == STARPU_NMAX_SCHED_CTXS)
		(*list)->sched_ctx = sched_ctx;
	else
	{
		struct _starpu_sched_ctx_list *l = (struct _starpu_sched_ctx_list*)malloc(sizeof(struct _starpu_sched_ctx_list));
		l->sched_ctx = sched_ctx;
		l->next = *list;
		*list = l;
	}
}

void _starpu_sched_ctx_list_remove(struct _starpu_sched_ctx_list **list, unsigned sched_ctx)
{
	struct _starpu_sched_ctx_list *l = NULL;
	struct _starpu_sched_ctx_list *prev = NULL;
	for (l = (*list); l; l = l->next)
	{
		if(l->sched_ctx == sched_ctx)
			break;
		prev = l;
	}
	struct _starpu_sched_ctx_list *next = NULL;
	if(l->next)
		next = l->next;
	free(l);
	l = NULL;
	
	if(next)
	{
		if(prev)
			prev->next = next;
		else
			*list = next;
	}
}

unsigned _starpu_sched_ctx_list_get_sched_ctx(struct _starpu_sched_ctx_list *list, unsigned sched_ctx)
{
	struct _starpu_sched_ctx_list *l = NULL;
	for (l = list; l; l = l->next)
	{
		if(l->sched_ctx == sched_ctx)
			return sched_ctx;
	}
	return STARPU_NMAX_SCHED_CTXS;
}

void _starpu_sched_ctx_list_delete(struct _starpu_sched_ctx_list **list)
{
	while(*list)
	{
		struct _starpu_sched_ctx_list *next = (*list)->next;
		free(*list);
		*list = NULL;
		if(next)
			*list = next;
	}
		
}
