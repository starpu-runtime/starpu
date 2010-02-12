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

static unsigned book = 0;
static starpu_data_handle book_handle;

static void dummy_kernel(void *descr[], void *arg)
{
}

static starpu_codelet rw_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cuda_func = dummy_kernel,
	.cpu_func = dummy_kernel,
	.nbuffers = 1
};

int main(int argc, char **argv)
{
	starpu_init(NULL);

	/* initialize the resource */
	starpu_register_vector_data(&book_handle, 0, (uintptr_t)&book, 1, sizeof(unsigned));

	unsigned ntasks = 16*1024;

	unsigned t;
	for (t = 0; t < ntasks; t++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &rw_cl;

		/* we randomly select either a reader or a writer (give 10
		 * times more chances to be a reader) */
		task->buffers[0].mode = ((rand() % 10)==0)?STARPU_W:STARPU_R;
		task->buffers[0].handle = book_handle;

		int ret = starpu_submit_task(task);
		STARPU_ASSERT(!ret);
	}

	starpu_wait_all_tasks();

	starpu_shutdown();

	return 0;
}
