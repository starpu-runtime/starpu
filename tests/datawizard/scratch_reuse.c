/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#if !defined(STARPU_HAVE_SETENV) || !defined(STARPU_USE_CPU) || !defined(STARPU_HAVE_HWLOC)
#warning setenv is not defined or no cpu are available. Skipping test
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else

#ifdef STARPU_QUICK_CHECK
#define ITER 32
#else
#define ITER 128
#endif

static void kernel(void *buffers[], void *cl_args)
{
	STARPU_ASSERT(STARPU_MATRIX_GET_PTR(buffers[0]) != 0);
}

static struct starpu_codelet codelet =
{
	.name = "codelet",
	.cuda_funcs = { kernel },
	.nbuffers = 1,
	.modes = { STARPU_SCRATCH },
};

int main(int argc, char *argv[])
{
	setenv("STARPU_LIMIT_CUDA_MEM", "50", 1);

	int ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_cuda_worker_get_count() == 0)
	{
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	starpu_data_handle_t handle[ITER];

	int i;
	for (i = 0; i < ITER; i++)
	{
		starpu_matrix_data_register(&handle[i], -1, 0, 1024, 1024, 1024, sizeof(float));
		ret = starpu_task_insert(&codelet, STARPU_SCRATCH, handle[i], 0);
		if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	starpu_task_wait_for_all();

	for (i = 0; i < ITER; i++)
		starpu_data_unregister(handle[i]);

	starpu_shutdown();

	return 0;
}
#endif
