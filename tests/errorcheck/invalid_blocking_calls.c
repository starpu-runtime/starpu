/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
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

	/* try to fetch data in the STARPU_RAM while we are in a codelet, such a
	 * blocking call is forbidden */
	ret = starpu_sync_data_with_mem(handle, STARPU_RW);
	if (ret != -EDEADLK)
		exit(-1);

	ret = starpu_tag_wait(TAG);
	if (ret != -EDEADLK)
		exit(-1);
}

static starpu_codelet wrong_codelet = 
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = wrong_func,
	.cuda_func = wrong_func,
	.model = NULL,
	.nbuffers = 0
};

static void wrong_callback(void *arg)
{
	int ret;

	ret  = starpu_sync_data_with_mem(handle, STARPU_RW);
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
	starpu_register_vector_data(&handle, 0, (uintptr_t)&data,
						1, sizeof(unsigned));

	struct starpu_task *task = starpu_task_create();

	task->cl = &wrong_codelet;

	task->buffers[0].handle = handle;
	task->buffers[0].mode = STARPU_RW;

	task->use_tag = 1;
	task->tag_id = TAG;

	task->callback_func = wrong_callback;

	ret = starpu_submit_task(task);
	if (ret == -ENODEV)
		goto enodev;

	ret = starpu_tag_wait(TAG);
	if (ret)
		return -1;

	/* This call is valid as it is done by the application outside a
	 * callback */
	ret = starpu_sync_data_with_mem(handle, STARPU_RW);
	if (ret)
		return -1;

	starpu_release_data_from_mem(handle);

	starpu_shutdown();

	return 0;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
	 * could perform the kernel, so this is not an error from StarPU */
	return 0;
}
