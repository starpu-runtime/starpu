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
 * Try to use data_acquire_try in parallel with tasks
 */

void func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
	starpu_sleep(0.01);
}

static struct starpu_codelet cl =
{
	.modes = { STARPU_RW },
	.cpu_funcs = {func},
	.cuda_funcs = {func},
	.opencl_funcs = {func},
	.cpu_funcs_name = {"func"},
	.nbuffers = 1
};

unsigned token = 0;
starpu_data_handle_t token_handle;

static
void callback(void *arg)
{
	(void)arg;
        starpu_data_release(token_handle);
}

int main(int argc, char **argv)
{
	unsigned i;
	int ret;
	int nacquired;

        ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&token_handle, STARPU_MAIN_RAM, (uintptr_t)&token, sizeof(unsigned));

	ret = starpu_task_insert(&cl, STARPU_RW, token_handle, 0);
	if (ret == -ENODEV)
		goto enodev;
	ret = starpu_data_acquire_try(token_handle, STARPU_R);
	STARPU_ASSERT(ret != 0);

	starpu_do_schedule();
	while ((ret = starpu_data_acquire_try(token_handle, STARPU_R)) != 0)
	{
		starpu_sleep(0.001);
	}

	ret = starpu_task_insert(&cl, STARPU_RW, token_handle, 0);
	if (ret == -ENODEV)
		goto enodev;

	starpu_data_release(token_handle);

	starpu_task_wait_for_all();

	ret = starpu_data_acquire_try(token_handle, STARPU_R);
	STARPU_ASSERT(ret == 0);
	starpu_data_release(token_handle);

	starpu_data_unregister(token_handle);

	starpu_shutdown();

	return 0;

enodev:
	starpu_data_unregister(token_handle);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
