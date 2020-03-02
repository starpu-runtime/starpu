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

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>
#include "../helper.h"

/*
 * Test using the specific_nodes field by forcing the data to main memory
 * even if the task is run on a GPU (and actually doing the computation from
 * the CPU driving the GPU). It mixes such accesses and normal accesses from
 * the GPU
 */

unsigned data, data2;

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
}

static struct starpu_codelet specific_cl =
{
	.cpu_funcs = {specific_kernel},
	.cuda_funcs = {specific_kernel},
	.opencl_funcs = {specific_kernel},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_RW},
	.specific_nodes = 1,
	.nodes = {STARPU_SPECIFIC_NODE_CPU, STARPU_SPECIFIC_NODE_LOCAL},
};

void cpu_codelet_unsigned_inc(void *descr[], void *arg)
{
	(void)arg;
	unsigned *dataptr = (unsigned*) STARPU_VARIABLE_GET_PTR(descr[0]);
	(*dataptr)++;
}

#ifdef STARPU_USE_CUDA
void cuda_codelet_unsigned_inc(void *descr[], void *cl_arg);
#endif
#ifdef STARPU_USE_OPENCL
void opencl_codelet_unsigned_inc(void *buffers[], void *args);
#endif

static struct starpu_codelet cl =
{
	.cpu_funcs = {cpu_codelet_unsigned_inc},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cuda_codelet_unsigned_inc},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {opencl_codelet_unsigned_inc},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},
};

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program opencl_program;
#endif

int main(void)
{
	starpu_data_handle_t data_handle, data_handle2;

#ifdef STARPU_QUICK_CHECK
	unsigned ntasks = 10;
#else
	unsigned ntasks = 1000;
#endif

	int ret, ret2;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_load_opencl_from_file("tests/datawizard/opencl_codelet_unsigned_inc_kernel.cl",
						  &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif

	data = 0;
	data2 = 0;

	/* Create a void data which will be used as an exclusion mechanism. */
	starpu_variable_data_register(&data_handle, STARPU_MAIN_RAM, (uintptr_t) &data, sizeof(data));

	starpu_variable_data_register(&data_handle2, STARPU_MAIN_RAM, (uintptr_t) &data2, sizeof(data2));

	unsigned i;
	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();
		if (i%2)
			task->cl = &specific_cl;
		else
			task->cl = &cl;
		task->handles[0] = data_handle;
		task->handles[1] = data_handle2;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_data_unregister(data_handle);
	starpu_data_unregister(data_handle2);

	ret = (data == ntasks) ? EXIT_SUCCESS : EXIT_FAILURE;

#ifdef STARPU_USE_OPENCL
        ret2 = starpu_opencl_unload_opencl(&opencl_program);
        STARPU_CHECK_RETURN_VALUE(ret2, "starpu_opencl_unload_opencl");
#endif

	starpu_shutdown();

	return ret;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
