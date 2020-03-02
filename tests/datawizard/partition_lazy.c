/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "scal.h"

/*
 * Test partitioning an uninitialized vector
 */

struct starpu_codelet mycodelet =
{
	.cpu_funcs = { scal_func_cpu },
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = { scal_func_opencl },
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
#ifdef STARPU_USE_CUDA
	.cuda_funcs = { scal_func_cuda },
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	.cpu_funcs_name = {"scal_func_cpu"},
	.modes = { STARPU_W },
        .model = NULL,
        .nbuffers = 1
};

int main(int argc, char **argv)
{
	unsigned *foo;
	starpu_data_handle_t handle;
	int ret;
	int n, size;
	unsigned i;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_load_opencl_from_file("tests/datawizard/scal_opencl.cl", &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif

	n = starpu_worker_get_count();
	size = 10 * n;

	starpu_vector_data_register(&handle, -1, (uintptr_t)NULL, size, sizeof(*foo));

	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = n > 1 ? n : 2,
	};

	starpu_data_partition(handle, &f);

	for (i = 0; i < f.nchildren; i++)
	{
		ret = starpu_task_insert(&mycodelet,
					 STARPU_W,
					 starpu_data_get_sub_data(handle, 1, i),
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	starpu_data_unregister(handle);
	starpu_shutdown();

        return 0;

enodev:
	starpu_data_unregister(handle);
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
