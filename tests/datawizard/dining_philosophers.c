/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../helper.h"

/*
 * Test Dijkstra's Dining Philosophers problem
 */

/* TODO: try to use an arbiter and check improved concurrency */

/* number of philosophers */
#define N	16

starpu_data_handle_t fork_handles[N];
unsigned forks[N];

void eat_kernel(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
}

static struct starpu_codelet eating_cl =
{
	.modes = { STARPU_RW, STARPU_RW },
	.cuda_funcs = {eat_kernel},
	.cpu_funcs = {eat_kernel},
        .opencl_funcs = {eat_kernel},
	.cpu_funcs_name = {"eat_kernel"},
	.nbuffers = 2
};

static
int submit_one_task(unsigned p)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &eating_cl;

	unsigned left = p;
	unsigned right = (p+1)%N;

	task->handles[0] = fork_handles[left];
	task->handles[1] = fork_handles[right];

	int ret = starpu_task_submit(task);
	return ret;
}

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* initialize the forks */
	unsigned f;
	for (f = 0; f < N; f++)
	{
		forks[f] = 0;

		starpu_vector_data_register(&fork_handles[f], STARPU_MAIN_RAM, (uintptr_t)&forks[f], 1, sizeof(unsigned));
		starpu_data_set_sequential_consistency_flag(fork_handles[f], 0);
	}

	unsigned ntasks = 1024;

	unsigned t;
	for (t = 0; t < ntasks; t++)
	{
		/* select one philosopher randomly */
		unsigned philosopher = rand() % N;
		ret = submit_one_task(philosopher);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	FPRINTF(stderr, "waiting done\n");
	for (f = 0; f < N; f++)
	{
		starpu_data_unregister(fork_handles[f]);
	}

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	for (f = 0; f < N; f++)
	{
		starpu_data_unregister(fork_handles[f]);
	}
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
