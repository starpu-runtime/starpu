/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#include <starpu.h>

#define NLOOPS	128

static void callback(void *arg)
{
	struct starpu_task *taskA, *taskB;

	taskA = starpu_get_current_task();
	taskB = arg;

	starpu_task_declare_deps_array(taskB, 1, &taskA);
	starpu_submit_task(taskB);
}

static void dummy_func(void *descr[] __attribute__ ((unused)), void *arg __attribute__ ((unused)))
{
}

static starpu_codelet dummy_codelet = 
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = dummy_func,
	.cuda_func = dummy_func,
	.model = NULL,
	.nbuffers = 0
};

static struct starpu_task *create_dummy_task(void)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &dummy_codelet;
	task->cl_arg = NULL;

	return task;
}

int main(int argc, char **argv)
{
	int ret;
	unsigned loop;

	starpu_init(NULL);

	struct starpu_task *taskA, *taskB;

	for (loop = 0; loop < NLOOPS; loop++)
	{
		taskA = create_dummy_task();
		taskB = create_dummy_task();

		taskA->callback_func = callback;
		taskA->callback_arg = taskB;

		starpu_submit_task(taskA);
	}

	starpu_wait_all_tasks();

	starpu_shutdown();

	return 0;
}
