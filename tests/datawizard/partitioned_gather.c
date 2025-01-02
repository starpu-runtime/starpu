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

#include <starpu.h>
#include "../helper.h"
#include "../vector/memset.h"

#define SIZE (1<<20)
#define NPARTS 16

/*
 * Test gathering data without home node with asynchronous partitioning.
 */

int main(void)
{
	int ret;
	starpu_data_handle_t handle, handles[NPARTS];
	int i;
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

	starpu_vector_data_register(&handle, -1, 0, SIZE, sizeof(char));

	/* We will work with this GPU */
	int gpu;
	int node;

	if (starpu_cuda_worker_get_count() > 0)
		gpu = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
	else
		gpu = starpu_worker_get_by_type(STARPU_OPENCL_WORKER, 0);

	node = starpu_worker_get_memory_node(gpu);

	/* Fork */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = NPARTS
	};
	starpu_data_partition_plan(handle, &f, handles);

	/* Memset on the pieces on the GPU */
	for (i = 0; i < NPARTS; i++)
		starpu_task_insert(&memset_cl, STARPU_EXECUTE_ON_WORKER, gpu, STARPU_W, handles[i], 0);

	/* Check that the whole data on the GPU is correct */
	starpu_task_insert(&memset_check_content_cl, STARPU_EXECUTE_ON_WORKER, gpu, STARPU_R, handle, 0);

	/* That's because the pointers in the gathering node (by default, the main ram) are the same */
	struct starpu_vector_interface *handle_interface = starpu_data_get_interface_on_node(handle, STARPU_MAIN_RAM);
	struct starpu_vector_interface *subhandle0_interface = starpu_data_get_interface_on_node(handles[0], STARPU_MAIN_RAM);

	starpu_data_acquire_on_node(handle, STARPU_MAIN_RAM, STARPU_R);
	starpu_data_acquire_on_node(handles[0], STARPU_MAIN_RAM, STARPU_R);
	STARPU_ASSERT(handle_interface->ptr == subhandle0_interface->ptr);
	STARPU_ASSERT(handle_interface->dev_handle == subhandle0_interface->dev_handle);
	STARPU_ASSERT(handle_interface->offset == subhandle0_interface->offset);
	starpu_data_release_on_node(handles[0], STARPU_MAIN_RAM);
	starpu_data_release_on_node(handle, STARPU_MAIN_RAM);

	/* Clean */
	starpu_data_partition_clean(handle, NPARTS, handles);

	starpu_data_unregister(handle);

	starpu_shutdown();

	return 0;
}
