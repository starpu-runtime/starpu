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
 * Trigger re-using a buffer allocation on GPUs
 */

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
static void codelet(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
	FPRINTF(stderr, "%lx\n", (unsigned long) STARPU_VARIABLE_GET_PTR(descr[0]));
	FPRINTF(stderr, "codelet\n");
}
#endif

#ifdef STARPU_USE_CUDA
static struct starpu_codelet cuda_cl =
{
	.cuda_funcs = {codelet},
	.nbuffers = 1,
	.modes = {STARPU_R}
};
#endif

#ifdef STARPU_USE_OPENCL
static struct starpu_codelet opencl_cl =
{
	.opencl_funcs = {codelet},
	.nbuffers = 1,
	.modes = {STARPU_R}
};
#endif

void dotest(struct starpu_codelet *cl)
{
	int ret;
	int var = 42;
	starpu_data_handle_t handle;

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&var, sizeof(var));

	ret = starpu_task_insert(cl, STARPU_R, handle, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();

	starpu_data_unregister(handle);

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&var, sizeof(var));

	ret = starpu_task_insert(cl, STARPU_R, handle, 0);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();

enodev:
	starpu_data_unregister(handle);
}

int main()
{
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_CUDA
	dotest(&cuda_cl);
#endif
#ifdef STARPU_USE_OPENCL
	dotest(&opencl_cl);
#endif

	starpu_shutdown();

	return 0;
}
