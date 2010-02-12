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

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>

#define VECTORSIZE	1024

starpu_data_handle v_handle;

#ifdef USE_CUDA
static void cuda_codelet_null(void *descr[], __attribute__ ((unused)) void *_args)
{
	char *buf = (char *)STARPU_GET_VECTOR_PTR(descr[0]);

	cudaMemset(buf, 42, 1);
}
#endif

static void core_codelet_null(void *descr[], __attribute__ ((unused)) void *_args)
{
	char *buf = (char *)STARPU_GET_VECTOR_PTR(descr[0]);

	*buf = 42;
}

static void display_var(void *descr[], __attribute__ ((unused)) void *_args)
{
	char *buf = (char *)STARPU_GET_VECTOR_PTR(descr[0]);
	if (*buf != 42)
	{
		fprintf(stderr, "Value = %c (should be %c)\n", *buf, 42);
		exit(-1);
	}
}

static starpu_codelet cl = {
	.where = STARPU_CORE|STARPU_CUDA,
	.core_func = core_codelet_null,
#ifdef USE_CUDA
	.cuda_func = cuda_codelet_null,
#endif
	.nbuffers = 1
};

static starpu_codelet display_cl = {
	.where = STARPU_CORE,
	.core_func = display_var,
	.nbuffers = 1
};


int main(int argc, char **argv)
{
	int ret;

	starpu_init(NULL);

	/* The buffer should never be explicitely allocated */
	starpu_register_vector_data(&v_handle, (uint32_t)-1, (uintptr_t)NULL, VECTORSIZE, sizeof(char));

	struct starpu_task *task = starpu_task_create();
		task->cl = &cl;
		task->buffers[0].handle = v_handle;
		task->buffers[0].mode = STARPU_W;
		task->detach = 0;

	ret = starpu_submit_task(task);
	if (ret == -ENODEV)
			goto enodev;

	ret = starpu_wait_task(task);
	if (ret)
		exit(-1);

	task = starpu_task_create();
		task->cl = &display_cl;
		task->buffers[0].handle = v_handle;
		task->buffers[0].mode = STARPU_R;
		task->detach = 0;

	ret = starpu_submit_task(task);
	if (ret == -ENODEV)
			goto enodev;

	ret = starpu_wait_task(task);
	if (ret)
		exit(-1);

	/* this should get rid of automatically allocated buffers */
	starpu_delete_data(v_handle);

	starpu_shutdown();

	return 0;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return 0;
}
