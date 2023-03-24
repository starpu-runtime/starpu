/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2021, 2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <unistd.h>
#include <starpu.h>
#include "../helper.h"

/*
 * Test that job creation is threadsafe
 */

#define N 1000

static struct starpu_task *tasks[N];

void dummy_func(void *arg)
{
	unsigned worker, i;
	int worker0;
	(void) arg;

	starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, &worker0, 1);
	if ((int) starpu_worker_get_id_check() == worker0)
		/* One worker creates the tasks */
		for (i = 0; i < N; i++)
		{
			struct starpu_task *task = starpu_task_create();
			task->destroy = 0;
			STARPU_WMB();
			tasks[i] = task;
		}
	else
		/* While others eagerly wait for it before trying to get their id */
		for (i = 0; i < N; i++)
		{
			struct starpu_task *task;
			while (!(task = tasks[i]))
			{
				STARPU_UYIELD();
				STARPU_SYNCHRONIZE();
			}
			STARPU_RMB();
			starpu_task_get_job_id(task);
		}
}

int main(void)
{
	int ret;
	unsigned i;
	struct starpu_conf conf;

	starpu_conf_init(&conf);
	conf.precedence_over_environment_variables = 1;
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	STARPU_HG_DISABLE_CHECKING(tasks);
	starpu_execute_on_each_worker(dummy_func, NULL, STARPU_CPU);

	for (i = 0; i < N; i++)
	{
		starpu_task_destroy(tasks[i]);
	}

	struct starpu_task *task = starpu_task_create();
	unsigned long id;
	task->destroy = 0;
	id = starpu_task_get_job_id(task);
	starpu_task_destroy(task);

	FPRINTF(stderr, "jobid %lu for %u tasks and %u workers\n",
			id, N, starpu_worker_get_count());

	/* We are not supposed to have created more than one jobid for each
	 * worker (for execute_on_each) and for each of the N user tasks. */
	ret = id > starpu_worker_get_count() + N + 1;

	starpu_shutdown();
	return ret;
}
