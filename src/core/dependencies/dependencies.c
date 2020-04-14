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
#include <common/utils.h>
#include <core/dependencies/tags.h>
#include <core/jobs.h>
#include <core/sched_policy.h>
#include <core/dependencies/data_concurrency.h>

/* We assume that the job will not disappear under our hands */
void _starpu_notify_dependencies(struct _starpu_job *j)
{
	STARPU_ASSERT(j);
	STARPU_ASSERT(j->task);

	/* unlock tasks depending on that task */
	_starpu_notify_task_dependencies(j);

	/* unlock tags depending on that task */
	if (j->task->use_tag)
		_starpu_notify_tag_dependencies(j->tag);

}

/* TODO: make this a hashtable indexed by func+data and pass that through data. */
static starpu_notify_ready_soon_func notify_ready_soon_func;
static void *notify_ready_soon_func_data;

struct _starpu_notify_job_start_data
{
	double delay;
};

void starpu_task_notify_ready_soon_register(starpu_notify_ready_soon_func f, void *data)
{
	STARPU_ASSERT(!notify_ready_soon_func);
	notify_ready_soon_func = f;
	notify_ready_soon_func_data = data;
}

/* Called when a job has just started, so we can notify tasks which were waiting
 * only for this one when they can expect to start */
static void __starpu_job_notify_start(struct _starpu_job *j, double delay);
void _starpu_job_notify_start(struct _starpu_job *j, struct starpu_perfmodel_arch* perf_arch)
{
	double delay;

	if (!notify_ready_soon_func)
		return;

	delay = starpu_task_expected_length(j->task, perf_arch, j->nimpl);
	if (isnan(delay) || _STARPU_IS_ZERO(delay))
		return;

	__starpu_job_notify_start(j, delay);
}

static void __starpu_job_notify_start(struct _starpu_job *j, double delay)
{
	_starpu_notify_job_start_data data = { .delay = delay };

	_starpu_notify_job_start_tasks(j, &data);

	if (j->task->use_tag)
		_starpu_notify_job_start_tag_dependencies(j->tag, &data);

	/* TODO: check data notification */
}

/* Called when the last dependency of this job has just started, so we know that
 * this job will be released after the given delay. */
void _starpu_job_notify_ready_soon(struct _starpu_job *j, _starpu_notify_job_start_data *data)
{
	struct starpu_task *task = j->task;

	/* Notify that this task will start after the given delay */
	notify_ready_soon_func(notify_ready_soon_func_data, task, data->delay);


	/* Notify some known transitions as well */

	if (!task->cl || task->cl->where == STARPU_NOWHERE || task->where == STARPU_NOWHERE)
		/* This task will immediately terminate, so transition this */
		__starpu_job_notify_start(_starpu_get_job_associated_to_task(task), data->delay);
	if (j->quick_next)
		/* This job is actually a pre_sync job with a post_sync job to be released right after */
		_starpu_job_notify_ready_soon(j->quick_next, data);
}
