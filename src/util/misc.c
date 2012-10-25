/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Université de Bordeaux 1
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

const char *_starpu_get_cl_model_name(struct starpu_codelet *cl)
{
	if (!cl)
		return NULL;

	if (cl->model && cl->model->symbol && cl->model->symbol[0])
		return cl->model->symbol;
	else
		return cl->name;
}

const char *_starpu_get_job_model_name(struct _starpu_job *j)
{
	const char *ret = NULL;

	if (!j)
		return NULL;

	struct starpu_task *task = j->task;
	if (task)
		ret = _starpu_get_cl_model_name(task->cl);

#ifdef STARPU_USE_FXT
	if (!ret)
                ret = j->model_name;
#endif
        return ret;
}
