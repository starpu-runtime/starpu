/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2015  Université de Bordeaux
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

/*
 * This test stress the memory allocation system and should force StarPU to
 * reclaim memory from time to time.
 */

#include <assert.h>
#include <starpu.h>
#include "../helper.h"

#if !defined(STARPU_HAVE_SETENV)
#warning setenv is not defined. Skipping test
int main(int argc, char **argv)
{
	return STARPU_TEST_SKIPPED;
}
#else

void dummy_func(void *descr[], STARPU_ATTRIBUTE_UNUSED void *_args)
{
}

static struct starpu_codelet dummy_cl =
{
	.cpu_funcs = {dummy_func},
	.cpu_funcs_name = {"dummy_func"},
	.nbuffers = 1,
	.modes = {STARPU_W}
};

static void emit_task(starpu_data_handle_t handle)
{
	struct starpu_task *task = starpu_task_create();
	int ret;
	task->cl = &dummy_cl;
	task->handles[0] = handle;
	ret = starpu_task_submit(task);
	STARPU_ASSERT(ret == 0);
}

static struct starpu_codelet empty_cl =
{
	.cpu_funcs = {dummy_func},
	.cpu_funcs_name = {"dummy_func"},
	.nbuffers = 0,
};

static void emit_empty_task(void)
{
	struct starpu_task *task = starpu_task_create();
	int ret;
	task->cl = &empty_cl;
	ret = starpu_task_submit(task);
	STARPU_ASSERT(ret == 0);
}

#define TOTAL "100"
#define FILL (99*1024*1024)

int main(int argc, char **argv)
{
	int i, ret;
	struct starpu_conf conf;
	starpu_data_handle_t handle;
	void *allocated;

	setenv("STARPU_LIMIT_CPU_MEM", TOTAL, 1);

	starpu_conf_init(&conf);
	conf.ncpus = 1;
	conf.ncuda = 0;
	conf.nopencl = 0;
	conf.nmic = 0;
	conf.nscc = 0;

        ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&handle, -1, 0, FILL);

	/* This makes the data allocated */
	emit_task(handle);
	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	ret = starpu_malloc_flags(&allocated, FILL, STARPU_MALLOC_COUNT);
	/* Room should be busy due to data */
	STARPU_ASSERT(ret == -ENOMEM);

	ret = starpu_malloc_flags(&allocated, FILL, STARPU_MALLOC_COUNT|STARPU_MALLOC_NORECLAIM);
	/* But we should be able to tell we don't care */
	STARPU_ASSERT(ret == 0);
	((char*)allocated)[FILL-1] = 0;
	starpu_free_flags(allocated, FILL, STARPU_MALLOC_COUNT);

	/* Release the automatically allocated data */
	starpu_data_unregister(handle);

	/* Memory may not be available immediately, make sure the driver has
	 * the opportunity to release it */
	emit_empty_task();
	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");
	emit_empty_task();
	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	ret = starpu_malloc_flags(&allocated, FILL, STARPU_MALLOC_COUNT);
	/* Room should now be available */
	STARPU_ASSERT(ret == 0);
	starpu_free_flags(allocated, FILL, STARPU_MALLOC_COUNT);

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
