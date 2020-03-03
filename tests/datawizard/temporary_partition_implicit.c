/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define SIZE (1<<20)
#define NPARTS 16

/*
 * Test asynchronous partitioning on a temporary data without submitting explit
 * partitioning/unpartitioning.
 */

static void codelet(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;
}

static struct starpu_codelet clw =
{
	.where = STARPU_CPU,
	.cpu_funcs = {codelet},
	.nbuffers = 1,
	.modes = {STARPU_W}
};

static struct starpu_codelet clr =
{
	.where = STARPU_CPU,
	.cpu_funcs = {codelet},
	.nbuffers = 1,
	.modes = {STARPU_R}
};

int main(void)
{
	int ret;
	starpu_data_handle_t handle, handles[NPARTS];
	int i;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_vector_data_register(&handle, -1, 0, SIZE, sizeof(char));

	/* Fork */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = NPARTS
	};
	starpu_data_partition_plan(handle, &f, handles);

	/* Process in parallel */
	for (i = 0; i < NPARTS; i++)
	{
		ret = starpu_task_insert(&clw,
					 STARPU_W, handles[i],
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	/* Invalidate one random piece we don't care coherency about */
	starpu_data_invalidate_submit(handles[NPARTS/2]);

	/* Clean */
	starpu_data_unpartition_submit(handle, NPARTS, handles, -1);
	starpu_data_partition_clean(handle, NPARTS, handles);

	/* Read result */
	starpu_task_insert(&clr, STARPU_R, handle, 0);

	starpu_data_unregister(handle);

	starpu_shutdown();

	return 0;

enodev:
	starpu_data_unpartition_submit(handle, NPARTS, handles, -1);
	starpu_data_partition_clean(handle, NPARTS, handles);
	starpu_data_unregister(handle);
	starpu_shutdown();
	/* yes, we do not perform the computation but we did detect that no one
	 * could perform the kernel, so this is not an error from StarPU */
	fprintf(stderr, "WARNING: No one can execute this task\n");
	return STARPU_TEST_SKIPPED;
}
