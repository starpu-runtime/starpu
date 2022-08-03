/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../variable/increment.h"

/*
 * Test using the specific_nodes field by forcing the data to main memory
 * even if the task is run on a GPU (and actually doing the computation from
 * the CPU driving the GPU). It mixes such accesses and normal accesses from
 * the GPU
 */

unsigned data, data2;

void specific3_kernel(void *descr[] STARPU_ATTRIBUTE_UNUSED, void *arg STARPU_ATTRIBUTE_UNUSED)
{
	(void)arg;
}

static struct starpu_codelet specific3_cl =
{
	.cpu_funcs = {specific3_kernel},
	.cuda_funcs = {specific3_kernel},
	.opencl_funcs = {specific3_kernel},
	.hip_funcs = {specific3_kernel},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_RW},
	.specific_nodes = 1,
	.nodes = {STARPU_SPECIFIC_NODE_NONE, STARPU_SPECIFIC_NODE_NONE},
};

void specific2_kernel(void *descr[], void *arg)
{
	(void)arg;
	int node = starpu_task_get_current_data_node(0);
	STARPU_ASSERT(node >= 0);
	STARPU_ASSERT(starpu_node_get_kind(node) == STARPU_CPU_RAM);
	unsigned *dataptr = (unsigned*) STARPU_VARIABLE_GET_PTR(descr[0]);

	if (node == STARPU_MAIN_RAM)
		STARPU_ASSERT(dataptr == &data);

	(*dataptr)++;

	node = starpu_task_get_current_data_node(1);
	STARPU_ASSERT(node >= 0);
	STARPU_ASSERT(starpu_node_get_kind(node) == STARPU_CPU_RAM
			|| (unsigned) node == starpu_worker_get_local_memory_node());
	dataptr = (unsigned*) STARPU_VARIABLE_GET_PTR(descr[1]);

	if (node == STARPU_MAIN_RAM)
		STARPU_ASSERT(dataptr == &data2);
}

static struct starpu_codelet specific2_cl =
{
	.cpu_funcs = {specific2_kernel},
	.cuda_funcs = {specific2_kernel},
	.opencl_funcs = {specific2_kernel},
	.hip_funcs = {specific2_kernel},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_RW},
	.specific_nodes = 1,
	.nodes = {STARPU_SPECIFIC_NODE_CPU, STARPU_SPECIFIC_NODE_LOCAL_OR_CPU},
};

void specific_kernel(void *descr[], void *arg)
{
	(void)arg;
	int node = starpu_task_get_current_data_node(0);
	STARPU_ASSERT(node >= 0);
	STARPU_ASSERT(starpu_node_get_kind(node) == STARPU_CPU_RAM);
	unsigned *dataptr = (unsigned*) STARPU_VARIABLE_GET_PTR(descr[0]);

	if (node == STARPU_MAIN_RAM)
		STARPU_ASSERT(dataptr == &data);

	(*dataptr)++;


	node = starpu_task_get_current_data_node(1);
	STARPU_ASSERT((unsigned) node == starpu_worker_get_local_memory_node());
}

static struct starpu_codelet specific_cl =
{
	.cpu_funcs = {specific_kernel},
	.cuda_funcs = {specific_kernel},
	.opencl_funcs = {specific_kernel},
	.hip_funcs = {specific_kernel},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_RW},
	.specific_nodes = 1,
	.nodes = {STARPU_SPECIFIC_NODE_CPU, STARPU_SPECIFIC_NODE_LOCAL},
};

int main(void)
{
	starpu_data_handle_t data_handle, data_handle2;

#ifdef STARPU_QUICK_CHECK
	unsigned ntasks = 12;
#else
	unsigned ntasks = 1000;
#endif

	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	increment_load_opencl();

	data = 0;
	data2 = 0;

	/* Create a void data which will be used as an exclusion mechanism. */
	starpu_variable_data_register(&data_handle, STARPU_MAIN_RAM, (uintptr_t) &data, sizeof(data));
	starpu_variable_data_register(&data_handle2, STARPU_MAIN_RAM, (uintptr_t) &data2, sizeof(data2));

	unsigned i;
	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();
		if (i%4 == 0)
			task->cl = &specific_cl;
		else if (i%4 == 1)
			task->cl = &specific2_cl;
		else if (i%4 == 2)
			task->cl = &specific3_cl;
		else
			task->cl = &increment_cl;
		task->handles[0] = data_handle;
		task->handles[1] = data_handle2;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_data_unregister(data_handle);
	starpu_data_unregister(data_handle2);

	ret = (data == (ntasks*3) / 4) ? EXIT_SUCCESS : EXIT_FAILURE;

	increment_unload_opencl();
	starpu_shutdown();

	return ret;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_data_unregister(data_handle);
	starpu_data_unregister(data_handle2);
	increment_unload_opencl();
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
