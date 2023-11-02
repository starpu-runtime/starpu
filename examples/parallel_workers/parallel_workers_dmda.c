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

static void display_cpu(void *descr[], void *cl_arg)
{
	(void)descr;
	(void)cl_arg;
#pragma omp parallel
	{
#ifdef __linux__
		fprintf(stderr, "thread %d on cpu %d\n", omp_get_thread_num(), sched_getcpu());
#endif
	}
}

static struct starpu_perfmodel display_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "display"
};

static struct starpu_codelet display_cl =
{
	.cpu_funcs = {display_cpu, NULL},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.model = &display_model
};

int main(void)
{
	int ret;
	struct starpu_parallel_worker_config *parallel_workers;
	starpu_data_handle_t handle;

	setenv("STARPU_NMPI_MS","0",1);
	setenv("STARPU_CALIBRATE_MINIMUM", "2", 1);
	setenv("STARPU_CALIBRATE", "0", 1);
	setenv("STARPU_SCHED", "dmda", 1);

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	parallel_workers = starpu_parallel_worker_init(HWLOC_OBJ_SOCKET,
						       STARPU_PARALLEL_WORKER_POLICY_NAME, "dmda",
						       0);
	if (parallel_workers == NULL)
		goto enodev;
	starpu_parallel_worker_print(parallel_workers);

	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&ret, sizeof(ret));
	ret = starpu_task_insert(&display_cl,
				 STARPU_RW, handle,
				 STARPU_NAME, "task_pw",
				 0);
	if (ret == -ENODEV)
		goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* wait for all tasks at the end*/
	starpu_task_wait_for_all();
	starpu_data_unregister(handle);
	starpu_parallel_worker_shutdown(parallel_workers);
	starpu_shutdown();
	return 0;

enodev:
	starpu_shutdown();
	return 77;
}
#endif
