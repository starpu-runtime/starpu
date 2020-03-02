/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Check that concurrency does happen when using multi-stream CUDA.
 */

#ifdef STARPU_QUICK_CHECK
#define NITERS 100000
#else
#define NITERS 1000000
#endif
#define NTASKS 64
#define SYNC 16

#ifdef STARPU_USE_CUDA
extern void long_kernel_cuda(unsigned long niters);

void codelet_long_kernel_async(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
	long_kernel_cuda(NITERS);
}

void codelet_long_kernel_sync(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
	long_kernel_cuda(NITERS);
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

static struct starpu_perfmodel model_async =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "long_kernel_async",
};

static struct starpu_perfmodel model_sync =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "long_kernel_sync",
};

static struct starpu_codelet cl_async =
{
	.cuda_funcs = {codelet_long_kernel_async},
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 0,
	.model =  &model_async,
};

static struct starpu_codelet cl =
{
	.cuda_funcs = {codelet_long_kernel_sync},
	.nbuffers = 0,
	.model =  &model_sync,
};
#endif

int main(int argc, char **argv)
{
#ifndef STARPU_USE_CUDA
	return STARPU_TEST_SKIPPED;
#else
	setenv("STARPU_NWORKER_PER_CUDA", "4", 1);
	int ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	if (starpu_cuda_worker_get_count() == 0)
	{
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	unsigned iter;
	for (iter = 0; iter < NTASKS; iter++)
	{
		struct starpu_task *task = starpu_task_create();

		if (!(iter % SYNC))
			/* Insert a synchronous task, just for fun */
			task->cl = &cl;
		else
			task->cl = &cl_async;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_shutdown();

	STARPU_RETURN(EXIT_SUCCESS);

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	STARPU_RETURN(STARPU_TEST_SKIPPED);
#endif
}
