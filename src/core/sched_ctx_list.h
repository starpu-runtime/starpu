/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  INRIA
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

struct _starpu_sched_ctx_list
{
	struct _starpu_sched_ctx_list *next;
	unsigned sched_ctx;
};

void _starpu_sched_ctx_list_init(struct _starpu_sched_ctx_list *list);
void _starpu_sched_ctx_list_add(struct _starpu_sched_ctx_list **list, unsigned sched_ctx);
void _starpu_sched_ctx_list_remove(struct _starpu_sched_ctx_list **list, unsigned sched_ctx);
unsigned _starpu_sched_ctx_list_get_sched_ctx(struct _starpu_sched_ctx_list *list, unsigned sched_ctx);
void _starpu_sched_ctx_list_delete(struct _starpu_sched_ctx_list **list);

#endif // __SCHED_CONTEXT_H__
