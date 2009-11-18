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

static void cuda_codelet_null(starpu_data_interface_t *buffers, __attribute__ ((unused)) void *_args)
{
	fprintf(stderr, "POUET CUDA\n");
	int *buf = (int *)buffers[0].vector.ptr;

	cudaMemset(buf, 42, sizeof(int));
}

static void core_codelet_null(starpu_data_interface_t *buffers, __attribute__ ((unused)) void *_args)
{
	fprintf(stderr, "POUET CUDA\n");
	int *buf = (int *)buffers[0].vector.ptr;

	*buf = 42;
}

static void display_var(starpu_data_interface_t *buffers, __attribute__ ((unused)) void *_args)
{
	int *buf = (int *)buffers[0].vector.ptr;

	fprintf(stderr, "Value = %d (should be %d)\n", *buf, 42);
	fflush(stderr);

	if (*buf != 42)
		exit(-1);
}

static starpu_codelet cl = {
	.where = CORE|CUDA,
	.core_func = core_codelet_null,
	.cuda_func = cuda_codelet_null,
	.nbuffers = 1
};

static starpu_codelet display_cl = {
	.where = CORE,
	.core_func = display_var,
	.nbuffers = 1
};


int main(int argc, char **argv)
{
	starpu_init(NULL);

	/* The buffer should never be explicitely allocated */
	starpu_register_vector_data(&v_handle, (uint32_t)-1, (uintptr_t)NULL, VECTORSIZE, sizeof(int));

	struct starpu_task *task = starpu_task_create();
		task->cl = &cl;
		task->buffers[0].handle = v_handle;
		task->buffers[0].mode = STARPU_W;

		task->synchronous = 1;

	int ret = starpu_submit_task(task);
	if (ret == -ENODEV)
			goto enodev;

//	starpu_wait_task(task);

	task = starpu_task_create();
		task->cl = &display_cl;
		task->buffers[0].handle = v_handle;
		task->buffers[0].mode = STARPU_R;

		task->synchronous = 1;

	ret = starpu_submit_task(task);
	if (ret == -ENODEV)
			goto enodev;

//	starpu_wait_task(task);

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
