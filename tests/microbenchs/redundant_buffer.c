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

#define N	10000

#define VECTORSIZE	1024

starpu_data_handle v_handle;
static unsigned *v;

static void opencl_codelet_null(void *descr[], __attribute__ ((unused)) void *_args)
{
}

static void cuda_codelet_null(void *descr[], __attribute__ ((unused)) void *_args)
{
}

static void cpu_codelet_null(void *descr[], __attribute__ ((unused)) void *_args)
{
}

static starpu_codelet cl = {
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL,
	.cpu_func = cpu_codelet_null,
	.cuda_func = cuda_codelet_null,
	.opencl_func = opencl_codelet_null,
        .nbuffers = 2
};


int main(int argc, char **argv)
{
	starpu_init(NULL);

	starpu_data_malloc_pinned_if_possible((void **)&v, VECTORSIZE*sizeof(unsigned));

	starpu_vector_data_register(&v_handle, 0, (uintptr_t)v, VECTORSIZE, sizeof(unsigned));

	unsigned iter;
	for (iter = 0; iter < N; iter++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &cl;

		task->buffers[0].handle = v_handle;
		task->buffers[0].mode = STARPU_R;

		task->buffers[1].handle = v_handle;
		task->buffers[1].mode = STARPU_R;

		int ret = starpu_task_submit(task);
		if (ret == -ENODEV)
			goto enodev;
	}

	starpu_task_wait_for_all();

	starpu_shutdown();

	return 0;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return 0;
}
