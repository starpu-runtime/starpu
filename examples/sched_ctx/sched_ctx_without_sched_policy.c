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

#include <starpu.h>
#include <omp.h>

#ifndef STARPU_QUICK_CHECK
#define NTASKS 64
#else
#define NTASKS 10
#endif

int tasks_executed[2];

int parallel_code(int sched_ctx)
{
	int i;
	int t = 0;
	int *cpuids = NULL;
	int ncpuids = 0;
	starpu_sched_ctx_get_available_cpuids(sched_ctx, &cpuids, &ncpuids);

//	printf("execute task of %d threads \n", ncpuids);
#pragma omp parallel num_threads(ncpuids) reduction(+:t)
	{
		starpu_sched_ctx_bind_current_thread_to_cpuid(cpuids[omp_get_thread_num()]);
// 			printf("cpu = %d ctx%d nth = %d\n", sched_getcpu(), sched_ctx, omp_get_num_threads());
#pragma omp for
		for(i = 0; i < NTASKS; i++)
			t++;
	}

	free(cpuids);
	return t;
}

static void sched_ctx_func(void *descr[], void *arg)
{
	(void)descr;
	unsigned sched_ctx = (uintptr_t)arg;
	tasks_executed[sched_ctx-1] += parallel_code(sched_ctx);
}


static struct starpu_codelet sched_ctx_codelet =
{
	.cpu_funcs = {sched_ctx_func},
	.model = NULL,
	.nbuffers = 0,
	.name = "sched_ctx"
};


int main(void)
{
	tasks_executed[0] = 0;
	tasks_executed[1] = 0;
	int ntasks = NTASKS;
	int ret, j, k;
	unsigned ncpus = 0;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	int nprocs1 = 1;
	int nprocs2 = 1;
	int ncuda = 0;
	int *procs1, *procs2, *procscuda;

#ifdef STARPU_USE_CUDA
	ncuda = starpu_cuda_worker_get_count();
	procscuda = (int*)malloc(ncuda*sizeof(int));
	starpu_worker_get_ids_by_type(STARPU_CUDA_WORKER, procscuda, ncuda);
#endif
#ifdef STARPU_USE_CPU
	ncpus = starpu_cpu_worker_get_count();
	procs1 = (int*)malloc(ncpus*sizeof(int));
	starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, procs1, ncpus);

	if(ncpus > 1)
	{
		nprocs1 = ncpus/2;
		nprocs2 =  ncpus-nprocs1;
		k = 0;
		procs2 = (int*)malloc(nprocs2*sizeof(int));
		for(j = nprocs1; j < nprocs1+nprocs2; j++)
			procs2[k++] = procs1[j];
	}
	else
	{
		procs2 = (int*)malloc(nprocs2*sizeof(int));
		procs2[0] = procs1[0];

	}
#endif

	if (ncpus == 0) goto enodev;
#ifdef STARPU_USE_CUDA
	if (ncuda > 0 && nprocs1 > 1)
	{
		procs1[nprocs1-1] = procscuda[0];
	}
#endif

	/*create contexts however you want*/
	unsigned sched_ctx1 = starpu_sched_ctx_create(procs1, nprocs1, "ctx1", 0);
	unsigned sched_ctx2 = starpu_sched_ctx_create(procs2, nprocs2, "ctx2", 0);
	starpu_sched_ctx_display_workers(sched_ctx1, stderr);
	starpu_sched_ctx_display_workers(sched_ctx2, stderr);
	int i;
	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &sched_ctx_codelet;
		task->cl_arg = (void*)(uintptr_t) sched_ctx1;

		/*submit tasks to context*/
		ret = starpu_task_submit_to_ctx(task,sched_ctx1);

		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &sched_ctx_codelet;
		task->cl_arg = (void*)(uintptr_t) sched_ctx2;

		/*submit tasks to context*/
		ret = starpu_task_submit_to_ctx(task,sched_ctx2);

		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}


	/* tell starpu when you finished submitting tasks to this context
	   in order to allow moving resources from this context to the inheritor one
	   when its corresponding tasks finished executing */



	/* wait for all tasks at the end*/
	starpu_task_wait_for_all();

	starpu_sched_ctx_delete(sched_ctx1);
	starpu_sched_ctx_delete(sched_ctx2);
	printf("ctx%u: tasks starpu executed %d out of %d\n", sched_ctx1, tasks_executed[0], NTASKS*NTASKS);
	printf("ctx%u: tasks starpu executed %d out of %d\n", sched_ctx2, tasks_executed[1], NTASKS*NTASKS);

enodev:
#ifdef STARPU_USE_CPU
	free(procs1);
	free(procs2);
#endif
	starpu_shutdown();
	return ncpus == 0 ? 77 : 0;
}
