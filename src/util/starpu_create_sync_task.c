/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <common/config.h>
#include <core/task.h>

void starpu_create_sync_task(starpu_tag_t sync_tag, unsigned ndeps, starpu_tag_t *deps, void (*callback)(void *), void *callback_arg)
{
	starpu_tag_declare_deps_array(sync_tag, ndeps, deps);

	/* We create an empty task */
	struct starpu_task *sync_task = starpu_task_create();
	sync_task->name = "create_sync_task";

	sync_task->use_tag = 1;
	sync_task->tag_id = sync_tag;

	sync_task->callback_func = callback;
	sync_task->callback_arg = callback_arg;

	/* This task does nothing */
	sync_task->cl = NULL;

	int sync_ret = _starpu_task_submit_internally(sync_task);
	STARPU_ASSERT(!sync_ret);
}

void starpu_create_callback_task(void (*callback)(void *), void *callback_arg)
{
	/* We create an empty task */
	struct starpu_task *empty_task = starpu_task_create();
	empty_task->name = "empty_task";
	empty_task->callback_func = callback;
	empty_task->callback_arg = callback_arg;

	/* This task does nothing */
	empty_task->cl = NULL;

	int ret = _starpu_task_submit_internally(empty_task);
	STARPU_ASSERT(!ret);
}
