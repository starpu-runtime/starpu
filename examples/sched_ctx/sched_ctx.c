/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2010-2013  Centre National de la Recherche Scientifique
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

#ifdef STARPU_QUICK_CHECK
#define NTASKS 64
#else
#define NTASKS 1000
#endif

int tasks_executed = 0;
starpu_pthread_mutex_t mut;

static void sched_ctx_func(void *descr[] __attribute__ ((unused)), void *arg __attribute__ ((unused)))
{
	starpu_pthread_mutex_lock(&mut);
	tasks_executed++;
	starpu_pthread_mutex_unlock(&mut);
}

static struct starpu_codelet sched_ctx_codelet =
{
	.cpu_funcs = {sched_ctx_func, NULL},
	.cuda_funcs = {sched_ctx_func, NULL},
	.opencl_funcs = {sched_ctx_func, NULL},
	.model = NULL,
	.nbuffers = 0
};


int main(int argc, char **argv)
{
	int ntasks = NTASKS;
	int ret;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_pthread_mutex_init(&mut, NULL);
	int nprocs1 = 1;
	int nprocs2 = 1;
	int procs1[20], procs2[20];
	procs1[0] = 0;
	procs2[0] = 0;

#ifdef STARPU_USE_CPU
	unsigned ncpus =  starpu_cpu_worker_get_count();
	starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, procs1, ncpus);

	nprocs1 = ncpus;
#endif

#ifdef STARPU_USE_CUDA
	unsigned ncuda = starpu_cuda_worker_get_count();
	starpu_worker_get_ids_by_type(STARPU_CUDA_WORKER, procs2, ncuda);

	nprocs2 = ncuda == 0 ? 1 : ncuda;
#endif

	/*create contexts however you want*/
	unsigned sched_ctx1 = starpu_sched_ctx_create("dmda", procs1, nprocs1, "ctx1");
	unsigned sched_ctx2 = starpu_sched_ctx_create("dmda", procs2, nprocs2, "ctx2");

	/*indicate what to do with the resources when context 2 finishes (it depends on your application)*/
	starpu_sched_ctx_set_inheritor(sched_ctx2, sched_ctx1);

	int i;
	for (i = 0; i < ntasks/2; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &sched_ctx_codelet;
		task->cl_arg = NULL;

		/*submit tasks to context*/
		ret = starpu_task_submit_to_ctx(task,sched_ctx1);

		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/* tell starpu when you finished submitting tasks to this context
	   in order to allow moving resources from this context to the inheritor one
	   when its corresponding tasks finished executing */

	starpu_sched_ctx_finished_submit(sched_ctx1);

	for (i = 0; i < ntasks/2; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &sched_ctx_codelet;
		task->cl_arg = NULL;

		ret = starpu_task_submit_to_ctx(task,sched_ctx2);

		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_sched_ctx_finished_submit(sched_ctx2);

	/* wait for all tasks at the end*/
	starpu_task_wait_for_all();

	starpu_sched_ctx_delete(sched_ctx1);
	starpu_sched_ctx_delete(sched_ctx2);
	printf("tasks executed %d out of %d\n", tasks_executed, ntasks);
	starpu_shutdown();

	return 0;
}
