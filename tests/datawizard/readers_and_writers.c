/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Try mixing readers and writers on the same variable
 */

static unsigned book = 0;
static starpu_data_handle_t book_handle;

void dummy_kernel(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
}

static struct starpu_codelet r_cl =
{
	.cuda_funcs = {dummy_kernel},
	.cpu_funcs = {dummy_kernel},
	.opencl_funcs = {dummy_kernel},
	.cpu_funcs_name = {"dummy_kernel"},
	.nbuffers = 1,
	.modes = {STARPU_R}
};

static struct starpu_codelet w_cl =
{
	.cuda_funcs = {dummy_kernel},
	.cpu_funcs = {dummy_kernel},
	.opencl_funcs = {dummy_kernel},
	.cpu_funcs_name = {"dummy_kernel"},
	.nbuffers = 1,
	.modes = {STARPU_W}
};

int main(int argc, char **argv)
{
	int ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* initialize the resource */
	starpu_vector_data_register(&book_handle, STARPU_MAIN_RAM, (uintptr_t)&book, 1, sizeof(unsigned));

#ifdef STARPU_QUICK_CHECK
	unsigned ntasks = 16;
#else
	unsigned ntasks = 16*1024;
#endif

	unsigned t;
	for (t = 0; t < ntasks; t++)
	{
		struct starpu_task *task = starpu_task_create();

		task->handles[0] = book_handle;

		/* we randomly select either a reader or a writer (give 10
		 * times more chances to be a reader) */
		enum starpu_data_access_mode mode = ((rand() % 10)==0)?STARPU_W:STARPU_R;
		if (mode == STARPU_W)
			task->cl = &w_cl;
		else
			task->cl = &r_cl;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	starpu_data_unregister(book_handle);
	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_data_unregister(book_handle);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
