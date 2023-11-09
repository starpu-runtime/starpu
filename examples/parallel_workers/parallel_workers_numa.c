/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <omp.h>

#if !defined(STARPU_PARALLEL_WORKER)
int main(void)
{
	return 77;
}
#else

static void display_cpu(void * descr[], void *cl_arg)
{
	(void)cl_arg;
	int *val = (int *) STARPU_VARIABLE_GET_PTR(descr[0]);
#pragma omp parallel
	fprintf(stderr, "hello %d from the task %d\n", *val, omp_get_thread_num());
}

static struct starpu_codelet display_cl =
{
	.cpu_funcs = {display_cpu, NULL},
	.nbuffers = 1,
	.modes={STARPU_RW,STARPU_R, STARPU_R}
};

int main(void)
{
	setenv("STARPU_NMPI_MS","0",1);

	int ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	struct starpu_parallel_worker_config *parallel_workers = starpu_parallel_worker_init(HWLOC_OBJ_NUMANODE,
											     0);
	if (parallel_workers == NULL)
		goto enodev;
	starpu_parallel_worker_print(parallel_workers);

	starpu_data_handle_t handle;
	int val=42;
	starpu_variable_data_register(&handle, 0, (uintptr_t)&val, sizeof(val));

	ret = starpu_task_insert(&display_cl,
				 STARPU_RW, handle,
				 0);
	if (ret != -ENODEV)
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

out:
	starpu_task_wait_for_all();
	starpu_data_unregister(handle);

	starpu_parallel_worker_shutdown(parallel_workers);
	starpu_shutdown();
	return (ret == -ENODEV) ? 77 : 0 ;

enodev:
	starpu_shutdown();
	return 77;
}
#endif
