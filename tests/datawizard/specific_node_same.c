/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>
#include "../helper.h"

/*
 * Test using the specific_nodes field with loading the same data several
 * times on different nodes.
 */

unsigned data;

void specific_ro_kernel(void *descr[], void *arg)
{
	(void)arg;
	int node = starpu_task_get_current_data_node(0);
	STARPU_ASSERT(node >= 0);
	STARPU_ASSERT(starpu_node_get_kind(node) == STARPU_CPU_RAM);
	unsigned *dataptr = (unsigned*) STARPU_VARIABLE_GET_PTR(descr[0]);

	if (node == STARPU_MAIN_RAM)
		STARPU_ASSERT(dataptr == &data);

	node = starpu_task_get_current_data_node(1);
	STARPU_ASSERT((unsigned) node == starpu_worker_get_local_memory_node());
}

static struct starpu_codelet specific_cl_ro =
{
	.cpu_funcs = {specific_ro_kernel},
	.cuda_funcs = {specific_ro_kernel},
	.opencl_funcs = {specific_ro_kernel},
	.hip_funcs = {specific_ro_kernel},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_R},
	.specific_nodes = 1,
	.nodes = {STARPU_SPECIFIC_NODE_CPU, STARPU_SPECIFIC_NODE_LOCAL},
};

int main(void)
{
	starpu_data_handle_t data_handle;

#ifdef STARPU_QUICK_CHECK
	unsigned ntasks = 16;
#else
	unsigned ntasks = 1024;
#endif

	int ret;

	/* Disable prefetching, it makes the test work just by luck */
#ifdef STARPU_HAVE_SETENV
	setenv("STARPU_PREFETCH", "0", 1);
#endif

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	data = 0;

	/* Create a void data which will be used as an exclusion mechanism. */
	starpu_variable_data_register(&data_handle, STARPU_MAIN_RAM, (uintptr_t) &data, sizeof(data));

	unsigned i;
	for (i = 0 ; i < starpu_worker_get_count(); i++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &specific_cl_ro;
		task->execute_on_a_specific_worker = 1;
		task->workerid = i;

		task->handles[0] = data_handle;
		task->handles[1] = data_handle;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV)
		{
			task->destroy = 0;
			starpu_task_destroy(task);
			goto enodev;
		}
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_data_unregister(data_handle);

	starpu_shutdown();

	return ret;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_data_unregister(data_handle);
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
