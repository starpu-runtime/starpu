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

#define TAG	0x42

static starpu_data_handle handle;
static unsigned data = 42;

static void wrong_func(void *descr[], void *arg)
{
	int ret;

	/* try to fetch data in the RAM while we are in a codelet, such a
	 * blocking call is forbidden */
	ret = starpu_data_acquire(handle, STARPU_RW);
	if (ret != -EDEADLK)
		exit(-1);

	ret = starpu_tag_wait(TAG);
	if (ret != -EDEADLK)
		exit(-1);
}

static starpu_codelet wrong_codelet = 
{
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL,
	.cpu_func = wrong_func,
	.cuda_func = wrong_func,
        .opencl_func = wrong_func,
	.model = NULL,
	.nbuffers = 0
};

static void wrong_callback(void *arg)
{
	int ret;

	ret  = starpu_data_acquire(handle, STARPU_RW);
	if (ret != -EDEADLK)
		exit(-1);

	ret = starpu_tag_wait(TAG);
	if (ret != -EDEADLK)
		exit(-1);
}

int main(int argc, char **argv)
{
	int ret;

	starpu_init(NULL);

	/* register a piece of data */
	starpu_vector_data_register(&handle, 0, (uintptr_t)&data,
						1, sizeof(unsigned));

	struct starpu_task *task = starpu_task_create();

	task->cl = &wrong_codelet;

	task->buffers[0].handle = handle;
	task->buffers[0].mode = STARPU_RW;

	task->use_tag = 1;
	task->tag_id = TAG;

	task->callback_func = wrong_callback;

	ret = starpu_task_submit(task);
	if (ret == -ENODEV)
		goto enodev;

	ret = starpu_tag_wait(TAG);
	if (ret)
		return -1;

	/* This call is valid as it is done by the application outside a
	 * callback */
	ret = starpu_data_acquire(handle, STARPU_RW);
	if (ret)
		return -1;

	starpu_data_release(handle);

	starpu_shutdown();

	return 0;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
	 * could perform the kernel, so this is not an error from StarPU */
	return 77;
}
