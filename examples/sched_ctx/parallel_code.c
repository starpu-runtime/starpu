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
#ifdef STARPU_USE_CPU
#include <omp.h>

#ifdef STARPU_QUICK_CHECK
#define NTASKS 4
#else
#define NTASKS 10
#endif

int parallel_code(unsigned *sched_ctx)
{
	int i;
	int t = 0;
	int *cpuids = NULL;
	int ncpuids = 0;
	starpu_sched_ctx_get_available_cpuids(*sched_ctx, &cpuids, &ncpuids);

	/* printf("execute task of %d threads \n", ncpuids); */
	omp_set_num_threads(ncpuids);
#pragma omp parallel
	{
		starpu_sched_ctx_bind_current_thread_to_cpuid(cpuids[omp_get_thread_num()]);
			/* printf("cpu = %d ctx%d nth = %d\n", sched_getcpu(), *sched_ctx, omp_get_num_threads()); */
#pragma omp for
		for(i = 0; i < NTASKS; i++)
		{
#pragma omp atomic
				t++;
		}
	}

	free(cpuids);
	return t;
}

void *th(void* p)
{
	unsigned* sched_ctx = (unsigned*)p;
	void* ret;
	ret = starpu_sched_ctx_exec_parallel_code((void*)parallel_code, p, *sched_ctx);
	pthread_exit(ret);
}

int main(void)
{
	int ret;
	void* tasks_executed;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	int nprocs1;
	int *procs1;

	unsigned ncpus =  starpu_cpu_worker_get_count();
	procs1 = (int*)malloc(ncpus*sizeof(int));
	starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, procs1, ncpus);
	nprocs1 = ncpus;

	unsigned sched_ctx1 = starpu_sched_ctx_create(procs1, nprocs1, "ctx1", STARPU_SCHED_CTX_POLICY_NAME, "dmda", 0);

	/* This is the interesting part, we can launch a code to hijack the context and
		 use its cores to do something else entirely thanks to this */
	pthread_t mp;
	pthread_create(&mp, NULL, th, &sched_ctx1);

	pthread_join(mp, &tasks_executed);

	/* Finished, delete the context and print the amount of executed tasks */
	starpu_sched_ctx_delete(sched_ctx1);
	printf("ctx%u: tasks starpu executed %ld out of %d\n", sched_ctx1, (intptr_t)tasks_executed, NTASKS);
	starpu_shutdown();

	free(procs1);

	return 0;
}
#else /* STARPU_USE_CPU */
int main(int argc, char **argv)
{
	/* starpu_sched_ctx_exec_parallel_code() requires a CPU worker has parallel region master */
	return 77; /* STARPU_TEST_SKIPPED */
}
#endif /* STARPU_USE_CPU */
