/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../helper.h"
#include "../../src/core/sched_ctx_list.h"

int main(void)
{
	struct _starpu_sched_ctx_list *ctx_list = NULL, *found_list;
	struct _starpu_sched_ctx_elt *elt;
	struct _starpu_sched_ctx_list_iterator it;
	int ret=1, global=1;

	/* Check prio list addition */
	ret &= (_starpu_sched_ctx_list_add_prio(&ctx_list, 50, 0) != NULL);
	ret &= (ctx_list->priority == 50);
	ret &= (_starpu_sched_ctx_list_add_prio(&ctx_list, 999, 2) != NULL);
	ret &= (ctx_list->priority == 999);
	ret &= (ctx_list->next->priority == 50);
	ret &= !_starpu_sched_ctx_list_add(&ctx_list, 1);
	ret &= (ctx_list->next->next->priority == 0);

	/* Check elements added */
	ret &= (ctx_list->head->sched_ctx == 2);
	ret &= (ctx_list->next->head->sched_ctx == 0);
	ret &= (ctx_list->next->next->head->sched_ctx == 1);

	/* Check singleton status */
	ret &= (ctx_list->next->head->prev->sched_ctx == 0);
	ret &= (ctx_list->next->head->next->sched_ctx == 0);
	global &= ret;
	STARPU_CHECK_RETURN_VALUE_IS(ret, 1, "_starpu_sched_ctx_list_add");

	/* Check addition to existing list */
	ret = 1;
	_starpu_sched_ctx_elt_add(ctx_list->next, 3);
	ret &= (ctx_list->next->head->next->sched_ctx == 3);
	ret &= (ctx_list->next->head->prev->sched_ctx == 3);
	global &= ret;
	STARPU_CHECK_RETURN_VALUE_IS(ret, 1, "_starpu_sched_ctx_elt_add");

	/* Find element */
	ret = 1;
	elt = _starpu_sched_ctx_elt_find(ctx_list, 3);
	ret &= (elt != NULL && elt->sched_ctx == 3);
	elt = _starpu_sched_ctx_elt_find(ctx_list, 5);
	ret &= (elt == NULL);
	global &= ret;
	STARPU_CHECK_RETURN_VALUE_IS(ret, 1, "_starpu_sched_ctx_elt_find");

	/* Find list */
	ret = 1;
	found_list = _starpu_sched_ctx_list_find(ctx_list, 0);
	ret &= (found_list->priority == 0);
	ret &= (found_list->prev->priority == 50);
	found_list = _starpu_sched_ctx_list_find(ctx_list, 999);
	ret &= (found_list->priority==999);
	found_list = _starpu_sched_ctx_list_find(ctx_list, 42);
	ret &= (found_list == NULL);
	global &= ret;
	STARPU_CHECK_RETURN_VALUE_IS(ret, 1, "_starpu_sched_ctx_list_find");

	/* List exists */
	ret = 1;
	ret &= _starpu_sched_ctx_list_exists(ctx_list, 999);
	ret &= _starpu_sched_ctx_list_exists(ctx_list, 50);
	ret &= _starpu_sched_ctx_list_exists(ctx_list, 0);
	ret &= !_starpu_sched_ctx_list_exists(ctx_list, 42);
	global &= ret;
	STARPU_CHECK_RETURN_VALUE_IS(ret, 1, "_starpu_sched_ctx_list_exists");

	/* Iterator */
	ret = 1;
	ret &= !_starpu_sched_ctx_list_iterator_init(ctx_list, &it);
	ret &= _starpu_sched_ctx_list_iterator_has_next(&it);
	elt = _starpu_sched_ctx_list_iterator_get_next(&it);
	ret &= (elt->sched_ctx == 2);
	ret &= _starpu_sched_ctx_list_iterator_has_next(&it);
	elt = _starpu_sched_ctx_list_iterator_get_next(&it);
	ret &= (elt->sched_ctx == 0);
	ret &= _starpu_sched_ctx_list_iterator_has_next(&it);
	elt = _starpu_sched_ctx_list_iterator_get_next(&it);
	ret &= (elt->sched_ctx == 3);
	ret &= _starpu_sched_ctx_list_iterator_has_next(&it);
	elt = _starpu_sched_ctx_list_iterator_get_next(&it);
	ret &= (elt->sched_ctx == 1);
	ret &= !_starpu_sched_ctx_list_iterator_has_next(&it);
	global &= ret;
	STARPU_CHECK_RETURN_VALUE_IS(ret, 1, "_starpu_sched_ctx_list_iterator");

	/* Add element before head */
	ret = 1;
	_starpu_sched_ctx_elt_add_before(ctx_list->next, 4);
	ret &= (ctx_list->next->head->prev->sched_ctx == 4);
	ret &= (ctx_list->next->head->next->next->sched_ctx == 4);
	global &= ret;
	STARPU_CHECK_RETURN_VALUE_IS(ret, 1, "_starpu_sched_ctx_elt_add_before");

	/* Let's move it */
	ret = 1;
	ret &= !_starpu_sched_ctx_list_move(&ctx_list, 4, 1002);
	ret &= (ctx_list->priority == 1002);
	ret &= (ctx_list->head->sched_ctx == 4);
	ret &= (ctx_list->head->next->sched_ctx == 4);
	ret &= (ctx_list->next->next->head->prev->sched_ctx != 4);
	STARPU_CHECK_RETURN_VALUE_IS(ret, 1, "_starpu_sched_ctx_list_move");

	/* Let's remove it */
	ret = 1;
	elt = _starpu_sched_ctx_elt_find(ctx_list, 4);
	_starpu_sched_ctx_list_remove_elt(&ctx_list, elt);
	//ret &= (elt == NULL);
	ret &= (_starpu_sched_ctx_elt_find(ctx_list, 4) == NULL);
	ret &= (ctx_list->next->head->next->sched_ctx == 3);
	ret &= (ctx_list->next->head->prev->sched_ctx == 3);
	global &= ret;
	STARPU_CHECK_RETURN_VALUE_IS(ret, 1, "_starpu_sched_ctx_elt_remove");

	/* Let's remove head of that same ctx */
	ret = 1;
	ret &= !_starpu_sched_ctx_list_remove(&ctx_list, 0);
	ret &= (_starpu_sched_ctx_elt_find(ctx_list, 0) == NULL);
	ret &= (ctx_list->next->head->sched_ctx == 3);
	ret &= (ctx_list->next->head->next->sched_ctx == 3);
	ret &= (ctx_list->next->head->prev->sched_ctx == 3);
	global &= ret;
	STARPU_CHECK_RETURN_VALUE_IS(ret, 1, "_starpu_sched_ctx_list_remove");

	/* Remove the last one of this list, we get an empty ctx */
	ret = 1;
	ret &= !_starpu_sched_ctx_list_remove(&ctx_list, 3);
	ret &= (_starpu_sched_ctx_elt_find(ctx_list, 3) == NULL);
	found_list = _starpu_sched_ctx_list_find(ctx_list, 50);
	ret &= (found_list == NULL && ctx_list->priority != 50);
	ret &= (ctx_list->next->priority == 0);
	global &= ret;
	STARPU_CHECK_RETURN_VALUE_IS(ret, 1, "_starpu_sched_ctx_list_remove");

	/* Add an element to a new prio then remove it to ensure prio list is cleaned correctly */
	ret = 1;
	ret &= (_starpu_sched_ctx_list_add_prio(&ctx_list, 100000, 75) != NULL);
	ret &= (ctx_list->priority == 100000);
	ret &= (_starpu_sched_ctx_elt_find(ctx_list, 75) != NULL);
	ret &= (ctx_list->head->sched_ctx == 75);
	ret &= !_starpu_sched_ctx_list_remove(&ctx_list, 75);
	ret &= (_starpu_sched_ctx_elt_find(ctx_list, 75) == NULL);
	found_list = _starpu_sched_ctx_list_find(ctx_list, 100000);
	ret &= (found_list == NULL && ctx_list->priority != 100000);
	ret &= (ctx_list->priority == 999);
	global &= ret;
	STARPU_CHECK_RETURN_VALUE_IS(ret, 1, "_starpu_sched_ctx_list_remove");

	/* Delete this list, the function is internal only so we need to modify the list pointers too */
	ret = 1;
	found_list = ctx_list->next;
	found_list->prev = ctx_list->prev;
	_starpu_sched_ctx_list_remove_all(ctx_list);
	ctx_list = found_list;
	found_list = _starpu_sched_ctx_list_find(ctx_list, 999);
	ret &= (found_list == NULL && ctx_list->priority != 999);
	ret &= (_starpu_sched_ctx_elt_find(ctx_list, 2) == NULL);
	ret &= (ctx_list->priority == 0);
	ret &= (ctx_list->head->sched_ctx == 1); //as before
	ret &= (ctx_list->head->next->sched_ctx == 1);
	ret &= (ctx_list->head->prev->sched_ctx == 1);
	global &= ret;
	STARPU_CHECK_RETURN_VALUE_IS(ret, 1, "_starpu_sched_ctx_list_remove_all");

	/* Let's add some things again then clean everything */
	ret = 1;
	ret &= (_starpu_sched_ctx_list_add_prio(&ctx_list, 1000, 42) != NULL);
	ret &= (_starpu_sched_ctx_list_add_prio(&ctx_list, 1000, 43) != NULL);
	_starpu_sched_ctx_list_delete(&ctx_list);
	ret &= (ctx_list == NULL);
	global &= ret;
	STARPU_CHECK_RETURN_VALUE_IS(ret, 1, "_starpu_sched_ctx_list_delete");

	STARPU_CHECK_RETURN_VALUE_IS(global, 1, "_starpu_sched_ctx_(list|elt) global status");

	return 0;
}
