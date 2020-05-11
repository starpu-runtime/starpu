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
#include "../helper.h"

/*
 * Check that we detect that with only a CPU we can't submit a GPU-only task
 */

#if !defined(STARPU_USE_CPU)
#warning no cpu are available. Skipping test
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else

void dummy_func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
}

static struct starpu_codelet gpu_only_cl =
{
	.cuda_funcs = {dummy_func},
	.opencl_funcs = {dummy_func},
	.model = NULL,
	.nbuffers = 0
};

int main(void)
{
	int ret;

	/* We force StarPU to use 1 CPU only */
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.precedence_over_environment_variables = 1;
	conf.ncpus = 1;
	conf.nopencl = 0;
	conf.ncuda = 0;

	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	struct starpu_task *task = starpu_task_create();
	task->cl = &gpu_only_cl;

	/* Only a GPU device could execute that task ! */
	ret = starpu_task_submit(task);
	STARPU_ASSERT(ret == -ENODEV);

	task->destroy = 0;
	starpu_task_destroy(task);

	struct starpu_task *task_specific = starpu_task_create();
	task_specific->cl = &gpu_only_cl;
	task_specific->execute_on_a_specific_worker = 1;
	task_specific->workerid = 0;

	/* Only a CUDA device could execute that task ! */
	ret = starpu_task_submit(task_specific);
	STARPU_ASSERT(ret == -ENODEV);

	task_specific->destroy = 0;
	starpu_task_destroy(task_specific);

	starpu_shutdown();

	return EXIT_SUCCESS;
}
#endif
