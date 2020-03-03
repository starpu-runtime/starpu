/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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
 * Check that we catch calling tag_wait, i.e. a blocking call, from the
 * codelet function, which is invalid. This test is thus expected to fail.
 */

/* mpirun may not exit if it fails, skip the test for master-slave */
#if defined(STARPU_NO_ASSERT) || defined(STARPU_USE_MPI_MASTER_SLAVE)
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else

#define TAG	0x42

static starpu_data_handle_t handle;
static unsigned *data;

void wrong_func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
	/* The function is expected to fail. This is indicated in tests/Makefile.am */
	/* try to fetch data in the RAM while we are in a codelet, such a
	 * blocking call is forbidden */
	starpu_data_acquire(handle, STARPU_RW);
	starpu_tag_wait(TAG);
}

static struct starpu_codelet wrong_codelet =
{
	.modes = { STARPU_RW },
	.cpu_funcs = {wrong_func},
	.cuda_funcs = {wrong_func},
        .opencl_funcs = {wrong_func},
	.model = NULL,
	.nbuffers = 0
};

static void wrong_callback(void *arg)
{
	(void)arg;
	/* The function is expected to fail. This is indicated in tests/Makefile.am */
	starpu_data_acquire(handle, STARPU_RW);
	starpu_tag_wait(TAG);
}

int main(int argc, char **argv)
{
	int ret;

	if (RUNNING_ON_VALGRIND)
		return STARPU_TEST_SKIPPED;

	disable_coredump();

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_malloc((void**)&data, sizeof(*data));
	*data = 42;

	/* register a piece of data */
	starpu_vector_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)data,
						1, sizeof(unsigned));

	struct starpu_task *task = starpu_task_create();

	task->cl = &wrong_codelet;

	task->handles[0] = handle;

	task->use_tag = 1;
	task->tag_id = TAG;

	task->callback_func = wrong_callback;
	task->detach = 0;

	ret = starpu_task_submit(task);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	ret = starpu_tag_wait(TAG);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_tag_wait");

	/* This call is valid as it is done by the application outside a
	 * callback */
	ret = starpu_data_acquire(handle, STARPU_RW);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_acquire");

	starpu_data_release(handle);

	ret = starpu_task_wait(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait");
	starpu_data_unregister(handle);

	starpu_free(data);

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
#endif
