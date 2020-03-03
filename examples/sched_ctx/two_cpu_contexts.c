/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <stdlib.h>

/* This example case follows the same pattern its native Fortran version nf_sched_ctx.f90 */
static void sched_ctx_cpu_func(void *descr[], void *cl_args)
{
	(void)descr;
	int task_id;
	starpu_codelet_unpack_args(cl_args, &task_id);
	printf("task: %d, workerid: %d\n", task_id, starpu_worker_get_id());
}

static struct starpu_codelet sched_ctx_codelet =
{
	.cpu_funcs = {sched_ctx_cpu_func},
	.model = NULL,
	.nbuffers = 0,
	.name = "sched_ctx"
};

int main(void)
{
	int ncpu;
	int nprocs1;
	int nprocs2;
	int *procs = NULL;
	int *procs1 = NULL;
	int *procs2 = NULL;
	int i;
	int n = 20;
	int ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ncpu = starpu_cpu_worker_get_count();

	/* actually we really need at least 2 CPU workers such to allocate 2
	 * non overlapping contexts */
	if (ncpu < 2)
		return 77;
	procs = calloc(ncpu, sizeof(int));
	starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, procs, ncpu);

	nprocs1 = ncpu / 2;
	procs1 = calloc(nprocs1, sizeof(int));

	for (i=0; i<nprocs1; i++)
	{
		procs1[i] = procs[i];
	}

	nprocs2 = ncpu - nprocs1;
	procs2 = calloc(nprocs2, sizeof(int));
	for (i=0; i<nprocs2; i++)
	{
		procs2[i] = procs[i+nprocs1];
	}

        /* create sched context 1 with default policy, by giving a empty policy name */
	unsigned sched_ctx1 = starpu_sched_ctx_create(procs1, nprocs1, "ctx1", STARPU_SCHED_CTX_POLICY_NAME, "", 0);
        /* create sched context 2 with a user selected policy name */
	unsigned sched_ctx2 = starpu_sched_ctx_create(procs2, nprocs2, "ctx2", STARPU_SCHED_CTX_POLICY_NAME, "eager", 0);

	starpu_sched_ctx_set_inheritor(sched_ctx2, sched_ctx1);

	starpu_sched_ctx_display_workers(sched_ctx1, stderr);
	starpu_sched_ctx_display_workers(sched_ctx2, stderr);

	for (i=0; i < n; i++)
	{
		int arg_id = 1*1000 + i;
		ret = starpu_task_insert(&sched_ctx_codelet, STARPU_VALUE, &arg_id, sizeof(int), STARPU_SCHED_CTX, sched_ctx1, 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	for (i=0; i < n; i++)
	{
		int arg_id = 2*1000 + i;
		ret = starpu_task_insert(&sched_ctx_codelet, STARPU_VALUE, &arg_id, sizeof(int), STARPU_SCHED_CTX, sched_ctx2, 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_sched_ctx_finished_submit(sched_ctx2);

	for (i=0; i < n; i++)
	{
		int arg_id = 1*10000 + i;
		ret = starpu_task_insert(&sched_ctx_codelet, STARPU_VALUE, &arg_id, sizeof(int), STARPU_SCHED_CTX, sched_ctx1, 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_sched_ctx_finished_submit(sched_ctx1);
	starpu_task_wait_for_all();

	starpu_sched_ctx_add_workers(procs1, nprocs1, sched_ctx2);
	starpu_sched_ctx_delete(sched_ctx2);
	starpu_sched_ctx_delete(sched_ctx1);
	starpu_shutdown();
	free(procs);
	free(procs1);
	free(procs2);
	return 0;
}
