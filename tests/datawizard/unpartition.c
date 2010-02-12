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

#define NITER		1000
#define VECTORSIZE	1024

float *buffer;

starpu_data_handle v_handle;

static void dummy_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
}

static starpu_codelet cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = dummy_codelet,
#ifdef USE_CUDA
	.cuda_func = dummy_codelet,
#endif
	.nbuffers = 1
};

int use_handle(starpu_data_handle handle)
{
	int ret;
	struct starpu_task *task;

	task = starpu_task_create();
		task->cl = &cl;
		task->buffers[0].handle = handle;
		task->buffers[0].mode = STARPU_RW;
		task->detach = 0;

	ret = starpu_submit_task(task);

	return ret;
}

int main(int argc, char **argv)
{
	int ret;

	starpu_init(NULL);

	starpu_malloc_pinned_if_possible((void **)&buffer, VECTORSIZE);

	starpu_register_vector_data(&v_handle, 0, (uintptr_t)buffer, VECTORSIZE, sizeof(char));

	starpu_filter f = {
		.filter_func = starpu_divide_in_2_filter_func_vector,
		.filter_arg = VECTORSIZE/2
	};

	unsigned iter;
	for (iter = 0; iter < NITER; iter++)
	{
		starpu_map_filters(v_handle, 1, &f);
	
		ret = use_handle(starpu_get_sub_data(v_handle, 1, 0));
		if (ret == -ENODEV)
			goto enodev;
	
		ret = use_handle(starpu_get_sub_data(v_handle, 1, 1));
		if (ret == -ENODEV)
			goto enodev;
	
		starpu_wait_all_tasks();
	
		starpu_unpartition_data(v_handle, 0);
	
		ret = use_handle(v_handle);
		if (ret == -ENODEV)
			goto enodev;
	
		starpu_wait_all_tasks();
	}

	starpu_delete_data(v_handle);

	starpu_shutdown();

	return 0;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	return 0;
}
