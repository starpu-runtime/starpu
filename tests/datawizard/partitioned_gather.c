/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../vector/memset.h"

#define SIZE (1<<20)
#define NPARTS 16

/*
 * Test gathering data on the non-home node with asynchronous partitioning.
 */

int main(void)
{
	int ret;
	starpu_data_handle_t handle, handles[NPARTS];
	int i;
	char d[SIZE];
	struct starpu_conf conf;

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.ncuda = 1;
	conf.nopencl = 1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	if (starpu_cuda_worker_get_count() == 0 && starpu_opencl_worker_get_count() == 0)
	{
		FPRINTF(stderr, "This application requires a GPU\n");
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	memset(d, 0, SIZE*sizeof(char));
	starpu_vector_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t) &d, SIZE, sizeof(char));

	/* Fork */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = NPARTS
	};
	starpu_data_partition_plan(handle, &f, handles);

	/* We want to operate on a GPU */
	memset_cl.cpu_funcs[0] = NULL;
	memset_check_content_cl.cpu_funcs[0] = NULL;

	/* We will work with this GPU */
	int gpu;
	int node;

	if (starpu_cuda_worker_get_count() > 0)
	{
		/* We have a CUDA GPU, ignore any OpenCL GPU */
		memset_cl.opencl_funcs[0] = NULL;
		memset_check_content_cl.opencl_funcs[0] = NULL;
		gpu = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
	}
	else
	{
		gpu = starpu_worker_get_by_type(STARPU_OPENCL_WORKER, 0);
	}

	node = starpu_worker_get_memory_node(gpu);

	/* Prefetch the whole piece on the GPU, we should be able to use this to gather */
	starpu_data_acquire_on_node(handle, node, STARPU_R);
	starpu_data_release_on_node(handle, node);

	/* Memset on the pieces on the GPU */
	for (i = 0; i < NPARTS; i++)
	{
		starpu_task_insert(&memset_cl, STARPU_W, handles[i], 0);
	}

	/* Check that the whole data on the GPU is correct */
	starpu_task_insert(&memset_check_content_cl, STARPU_R, handle, 0);

	/* Clean */
	starpu_data_partition_clean(handle, NPARTS, handles);

	starpu_data_unregister(handle);

	starpu_shutdown();

	return 0;
}
