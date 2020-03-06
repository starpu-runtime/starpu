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

#include <config.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>
#include "../helper.h"

starpu_data_handle_t data_handle;

unsigned data;

void specific_kernel(STARPU_ATTRIBUTE_UNUSED void *descr[], STARPU_ATTRIBUTE_UNUSED void *_args)
{
	/* We do not protect this variable because it is only accessed when the
	 * "data_handle" piece of data is accessed. */
	unsigned *dataptr = (unsigned*) STARPU_VARIABLE_GET_PTR(descr[0]);

	STARPU_ASSERT(dataptr == &data);
	(*dataptr)++;
}

static struct starpu_codelet specific_cl =
{
	.cpu_funcs = {specific_kernel},
	.cuda_funcs = {specific_kernel},
	.opencl_funcs = {specific_kernel},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.specific_nodes = 1,
	.nodes = {0},
};

#ifdef STARPU_USE_CUDA
void cuda_codelet_unsigned_inc(void *descr[], STARPU_ATTRIBUTE_UNUSED void *cl_arg);
#endif
#ifdef STARPU_USE_OPENCL
void opencl_codelet_unsigned_inc(void *buffers[], void *args);
#endif

static struct starpu_codelet cl =
{
	.cpu_funcs = {specific_kernel},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cuda_codelet_unsigned_inc},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {opencl_codelet_unsigned_inc},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW},
};

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program opencl_program;
#endif

int main(STARPU_ATTRIBUTE_UNUSED int argc, STARPU_ATTRIBUTE_UNUSED char **argv)
{
#ifdef STARPU_QUICK_CHECK
	unsigned ntasks = 10;
#else
	unsigned ntasks = 1000;
#endif

	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_load_opencl_from_file("tests/datawizard/opencl_codelet_unsigned_inc_kernel.cl",
						  &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif

	data = 0;

	/* Create a void data which will be used as an exclusion mechanism. */
	starpu_variable_data_register(&data_handle, 0, (uintptr_t) &data, sizeof(data));

	unsigned i;
	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();
		if (i%2)
			task->cl = &specific_cl;
		else
			task->cl = &cl;
		task->handles[0] = data_handle;

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_data_unregister(data_handle);

	ret = (data == ntasks) ? EXIT_SUCCESS : EXIT_FAILURE;

#ifdef STARPU_USE_OPENCL
        ret = starpu_opencl_unload_opencl(&opencl_program);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
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
