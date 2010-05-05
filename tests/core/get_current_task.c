/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#include <stdio.h>
#include <unistd.h>
#include <starpu.h>

static unsigned ntasks = 65536;

static void check_task_func(void *descr[], void *arg)
{
	/* We check that the returned task is valid from the codelet */
	struct starpu_task *task = arg;
	STARPU_ASSERT(task == starpu_get_current_task());
}

static void check_task_callback(void *arg)
{
	/* We check that the returned task is valid from the callback */
	struct starpu_task *task = arg;
	STARPU_ASSERT(task == starpu_get_current_task());
}

static struct starpu_codelet_t dummy_cl = {
	.where = STARPU_CUDA|STARPU_CPU,
	.cuda_func = check_task_func,
	.cpu_func = check_task_func,
	.model = NULL,
	.nbuffers = 0
};

int main(int argc, char **argv)
{
	double timing;
	struct timeval start;
	struct timeval end;

	starpu_init(NULL);

	fprintf(stderr, "#tasks : %d\n", ntasks);

	int i;
	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();

		/* We check if the function is valid from the codelet or from
		 * the callback */
		task->cl = &dummy_cl;
		task->cl_arg = task;

		task->callback_func = check_task_callback;
		task->callback_arg = task;

		int ret = starpu_task_submit(task);
		STARPU_ASSERT(!ret);
	}

	starpu_task_wait_for_all();
	
	fprintf(stderr, "#empty tasks : %d\n", ntasks);

	/* We repeat the same experiment with null codelets */

	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = NULL;

		/* We check if the function is valid from the callback */
		task->callback_func = check_task_callback;
		task->callback_arg = task;

		int ret = starpu_task_submit(task);
		STARPU_ASSERT(!ret);
	}

	starpu_task_wait_for_all();

	starpu_shutdown();

	return 0;
}
