/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <common/utils.h>
#include <core/jobs.h>

const char *_starpu_codelet_get_model_name(struct starpu_codelet *cl)
{
	if (!cl)
		return NULL;

	if (cl->model && cl->model->symbol && cl->model->symbol[0])
		return cl->model->symbol;
	else
		return cl->name;
}

const char *_starpu_job_get_model_name(struct _starpu_job *j)
{
	if (!j)
		return NULL;

	struct starpu_task *task = j->task;
	if (!task)
		return NULL;

	return _starpu_codelet_get_model_name(task->cl);
}

const char *_starpu_job_get_task_name(struct _starpu_job *j)
{
	if (!j)
		return NULL;

	struct starpu_task *task = j->task;
	if (!task)
		return NULL;

	if (task->name)
		return task->name;
	else
		return _starpu_job_get_model_name(j);
}

const char *starpu_task_get_model_name(struct starpu_task *task)
{
	if (!task)
		return NULL;

	return _starpu_codelet_get_model_name(task->cl);
}

const char *starpu_task_get_name(struct starpu_task *task)
{
	if (!task)
		return NULL;
	if (task->name)
		return task->name;
	else
		return starpu_task_get_model_name(task);
}
