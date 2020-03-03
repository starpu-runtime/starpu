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
#include <core/jobs.h>
#include <core/task.h>
#include <common/utils.h>
#include <core/workers.h>
#include <common/barrier.h>

struct starpu_task *starpu_task_dup(struct starpu_task *task)
{
	struct starpu_task *task_dup;
	_STARPU_MALLOC(task_dup, sizeof(struct starpu_task));

	/* TODO perhaps this is a bit too much overhead and we should only copy
	 * part of the structure ? */
	memcpy(task_dup, task, sizeof(struct starpu_task));

	return task_dup;
}

void starpu_parallel_task_barrier_init_n(struct starpu_task* task, int worker_size)
{
	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
	j->task_size = worker_size;
	j->combined_workerid = -1;
	j->active_task_alias_count = 0;

	//fprintf(stderr, "POP -> size %d best_size %d\n", worker_size, best_size);

	STARPU_PTHREAD_BARRIER_INIT(&j->before_work_barrier, NULL, worker_size);
	STARPU_PTHREAD_BARRIER_INIT(&j->after_work_barrier, NULL, worker_size);
	j->after_work_busy_barrier = worker_size;

	return;
}

void starpu_parallel_task_barrier_init(struct starpu_task* task, int workerid)
{
	/* The master needs to dispatch the task between the
	 * different combined workers */
	struct _starpu_combined_worker *combined_worker =  _starpu_get_combined_worker_struct(workerid);
	int worker_size = combined_worker->worker_size;
	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	starpu_parallel_task_barrier_init_n(task, worker_size);

	j->combined_workerid = workerid;
}

