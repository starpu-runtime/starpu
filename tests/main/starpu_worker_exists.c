/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "core/workers.h"
#include "../helper.h"

/*
 * Test that _starpu_worker_exists works appropriately
 */

static int can_always_execute(unsigned workerid,
			      struct starpu_task *task,
			      unsigned nimpl)
{
	(void) workerid;
	(void) task;
	(void) nimpl;

	return 1;
}

static int can_never_execute(unsigned workerid,
			     struct starpu_task *task,
			     unsigned nimpl)
{
	(void) workerid;
	(void) task;
	(void) nimpl;

	return 0;
}

void fake(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
}

static struct starpu_codelet cl =
{
	.cpu_funcs    = { fake},
	.cuda_funcs   = { fake},
	.opencl_funcs = { fake},
	.cpu_funcs_name = { "fake"},
	.nbuffers     = 0
};

int main(int argc, char **argv)
{
	int ret;
	struct starpu_task *task;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;

	task = starpu_task_create();
	task->cl = &cl;
	task->destroy = 0;
	task->sched_ctx = 0;

	cl.can_execute = NULL;
	ret = _starpu_worker_exists(task);
	if (!ret)
	{
		FPRINTF(stderr, "failure with can_execute=NULL\n");
		return EXIT_FAILURE;
	}

	cl.can_execute = can_always_execute;
	ret = _starpu_worker_exists(task);
	if (!ret)
	{
		FPRINTF(stderr, "failure with can_always_execute\n");
		return EXIT_FAILURE;
	}

	cl.can_execute = can_never_execute;
	ret = _starpu_worker_exists(task);
	if (ret)
	{
		FPRINTF(stderr, "failure with can_never_execute\n");
		return EXIT_FAILURE;
	}

	starpu_task_destroy(task);
	starpu_shutdown();

	return EXIT_SUCCESS;
}
