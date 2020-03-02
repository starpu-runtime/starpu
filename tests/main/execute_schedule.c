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

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>
#include "../helper.h"
#include <common/thread.h>

/*
 * Test binding tasks on specific workers and in a specific order
 */

#ifdef STARPU_QUICK_CHECK
  #define K 2
#else
  #define K 16
#endif

#define N 64

static unsigned current = 1;

void codelet(void *descr[], void *_args)
{
	(void)descr;
	uintptr_t me = (uintptr_t) _args;
	STARPU_ASSERT(current == me);
	current++;
}

static double cost_function(struct starpu_task *task, unsigned nimpl)
{
	(void) task;
	(void) nimpl;
	return 1000;
}

static struct starpu_perfmodel model =
{
	.type = STARPU_COMMON,
	.cost_function = cost_function,
	.symbol = "cost"
};

static struct starpu_codelet cl =
{
	.cpu_funcs = {codelet},
	.cuda_funcs = {codelet},
	.opencl_funcs = {codelet},
	.nbuffers = 1,
	.modes = {STARPU_R},
	.model = &model,
};

int main(int argc, char **argv)
{
	int ret;
	struct starpu_task *dep_task[N];
	int *t[N];
	starpu_data_handle_t h[N];

	unsigned n, i, k;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	for (n = 0; n < N; n++)
	{
		t[n] = malloc((1<<20) * sizeof(*(t[n])));
		starpu_variable_data_register(&h[n], STARPU_MAIN_RAM, (uintptr_t) t[n], (1<<20) * sizeof(*(t[n])));
	}

	for (k = 0; k < K; k++)
	{
		for (n = 0; n < N; n++)
		{
			struct starpu_task *task;

			dep_task[n] = starpu_task_create();

			dep_task[n]->cl = NULL;

			task = starpu_task_create();

			task->cl = &cl;

			task->execute_on_a_specific_worker = 1;
			task->workerid = 0;
			/* We request for running the tasks in the opposite order of the submission order */
			task->workerorder = k*N + (N-n);
			task->cl_arg = (void*) (uintptr_t) (k*N + (N-n));
			task->handles[0] = h[n];

			starpu_task_declare_deps_array(task, 1, &dep_task[n]);

			ret = starpu_task_submit(task);
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		for (n = 0; n < N; n++)
		{
			i = (int)starpu_drand48()%(N-n);
			ret = starpu_task_submit(dep_task[i]);
			memmove(&dep_task[i], &dep_task[i+1], (N-i-1)*sizeof(dep_task[i]));
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}
	}

	starpu_task_wait_for_all();
	for (n = 0; n < N; n++)
	{
		starpu_data_unregister(h[n]);
		free(t[n]);
	}

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_shutdown();
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return STARPU_TEST_SKIPPED;
}
