/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2012  Université de Bordeaux 1
 * Copyright (C) 2010-2012  Centre National de la Recherche Scientifique
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

#include<starpu.h>
#include<pthread.h>

#define NTASKS 1000
int tasks_executed = 0;
pthread_mutex_t mut;
static void sched_ctx_func(void *descr[] __attribute__ ((unused)), void *arg __attribute__ ((unused)))
{
	pthread_mutex_lock(&mut);
	tasks_executed++;
	pthread_mutex_unlock(&mut);
}

static struct starpu_codelet sched_ctx_codelet =
{
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL,
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
	struct starpu_conf conf;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_QUICK_CHECK
	ntasks /= 100;
#endif

	pthread_mutex_init(&mut, NULL);
	unsigned ncuda = 0;
	unsigned ncpus = 1;
#ifdef STARPU_USE_CPU
	ncpus =  starpu_cpu_worker_get_count();

    int cpus[ncpus];
    starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, cpus, ncpus);

#endif
#ifdef STARPU_USE_CUDA
	ncuda = starpu_cuda_worker_get_count();
    int cudadevs[ncuda];
    starpu_worker_get_ids_by_type(STARPU_CUDA_WORKER, cudadevs, ncuda);
#endif


	int nprocs1 = ncpus;
	int nprocs2 = ncuda;

	int procs1[nprocs1];
	int procs2[nprocs2];

	int k;
	for(k = 0; k < nprocs1; k++)
	{
		if(k < ncpus)
			procs1[k] = cpus[k];
	}

	for(k = 0; k < nprocs2; k++)
	{
		procs2[k] = cudadevs[k];
	}


    /*create contexts however you want*/
	unsigned sched_ctx1 = starpu_create_sched_ctx("heft", procs1, nprocs1, "ctx1");
	unsigned sched_ctx2 = starpu_create_sched_ctx("heft", procs2, nprocs2, "ctx2");

	/*indicate what to do with the resources when context 2 finishes (it depends on your application)*/
	starpu_sched_ctx_set_inheritor(sched_ctx2, sched_ctx1);

	unsigned i;
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

	printf("tasks executed %d out of %d\n", tasks_executed, ntasks);
	starpu_shutdown();

	return 0;
}
