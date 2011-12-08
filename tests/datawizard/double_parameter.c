/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Université de Bordeaux 1
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

static void dummy_func(void *descr[] __attribute__ ((unused)), void *arg __attribute__ ((unused)))
{
}

static struct starpu_codelet codelet =
{
        .where = STARPU_CPU,
        .cpu_funcs = { dummy_func, NULL },
        .model = NULL,
        .nbuffers = 2
};


int main(int argc, char **argv)
{
	float foo = 0.0f;
	starpu_data_handle_t handle;
	int ret;
	struct starpu_task *task;

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&handle, 0, (uintptr_t)&foo, sizeof(foo));

#define SUBMIT(mode1, mode2) \
	task = starpu_task_create(); \
\
	task->cl = &codelet; \
	task->buffers[0].handle = handle; \
	task->buffers[0].mode = STARPU_##mode1; \
	task->buffers[1].handle = handle; \
	task->buffers[1].mode = STARPU_##mode2; \
\
	ret = starpu_task_submit(task); \
	if (ret == -ENODEV) goto enodev; \
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	SUBMIT(R,R);
	/* Not possible yet */
#if 0
	SUBMIT(R,W);
	SUBMIT(R,RW);
	SUBMIT(W,R);
	SUBMIT(W,W);
	SUBMIT(W,RW);
	SUBMIT(RW,R);
	SUBMIT(RW,W);
	SUBMIT(RW,RW);
#endif

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");
	starpu_data_unregister(handle);
	starpu_shutdown();

        return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(handle);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return 77;
}
