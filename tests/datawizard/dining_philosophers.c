/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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
#include "../common/helper.h"

/* number of philosophers */
#define N	16

starpu_data_handle fork_handles[N];
unsigned forks[N];

#define FPRINTF(ofile, fmt, args ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ##args); }} while(0)

static void eat_kernel(void *descr[], void *arg)
{
}

static starpu_codelet eating_cl = {
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL,
	.cuda_func = eat_kernel,
	.cpu_func = eat_kernel,
        .opencl_func = eat_kernel,
	.nbuffers = 2
};

int submit_one_task(unsigned p)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &eating_cl;

	unsigned left = p;
	unsigned right = (p+1)%N;

	task->buffers[0].handle = fork_handles[left];
	task->buffers[0].mode = STARPU_RW;
	task->buffers[1].handle = fork_handles[right];
	task->buffers[1].mode = STARPU_RW;

	int ret = starpu_task_submit(task);
	return ret;
}

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* initialize the forks */
	unsigned f;
	for (f = 0; f < N; f++)
	{
		forks[f] = 0;

		starpu_vector_data_register(&fork_handles[f], 0, (uintptr_t)&forks[f], 1, sizeof(unsigned));
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

	return 0;

enodev:
	for (f = 0; f < N; f++)
	{
		starpu_data_unregister(fork_handles[f]);
	}
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return 77;
}
