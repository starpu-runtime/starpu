/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifdef STARPU_HAVE_VALGRIND_H
#include <valgrind/valgrind.h>
#endif

#ifdef STARPU_QUICK_CHECK
#define NTASKS 64
#else
#define NTASKS 1000
#endif

int tasks_executed = 0;
starpu_pthread_mutex_t mut;

static void sched_ctx_cpu_func(void *descr[] STARPU_ATTRIBUTE_UNUSED, void *arg STARPU_ATTRIBUTE_UNUSED)
{
	STARPU_PTHREAD_MUTEX_LOCK(&mut);
	tasks_executed++;
	STARPU_PTHREAD_MUTEX_UNLOCK(&mut);
}

static void sched_ctx_cuda_func(void *descr[] STARPU_ATTRIBUTE_UNUSED, void *arg STARPU_ATTRIBUTE_UNUSED)
{
}

static struct starpu_codelet sched_ctx_codelet1 =
{
	.cpu_funcs = {sched_ctx_cpu_func},
	.model = NULL,
	.nbuffers = 0,
	.name = "sched_ctx"
};

static struct starpu_codelet sched_ctx_codelet2 =
{
	.cpu_funcs = {sched_ctx_cpu_func},
	.cuda_funcs = {sched_ctx_cuda_func},
	.model = NULL,
	.nbuffers = 0,
	.name = "sched_ctx"
};


int main(int argc, char **argv)
{
	int ntasks = NTASKS;
	int ret;
	unsigned ncuda = 0;
	int nprocs1 = 0;
	int nprocs2 = 0;
	int procs1[STARPU_NMAXWORKERS], procs2[STARPU_NMAXWORKERS];
	char *sched;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_HAVE_VALGRIND_H
       if (RUNNING_ON_VALGRIND)
               ntasks = 8;
#endif

	STARPU_PTHREAD_MUTEX_INIT(&mut, NULL);

#ifdef STARPU_USE_CPU
	nprocs1 = starpu_cpu_worker_get_count();
	starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, procs1, nprocs1);
#endif
	// if there is no cpu, skip
	if (nprocs1 == 0) goto enodev;

#ifdef STARPU_USE_CUDA
	ncuda = nprocs2 = starpu_cuda_worker_get_count();
	starpu_worker_get_ids_by_type(STARPU_CUDA_WORKER, procs2, nprocs2);
#endif
	if (nprocs2 == 0)
	{
	     nprocs2 = 1;
	     procs2[0] = procs1[0];
	}

	/*create contexts however you want*/
	sched = getenv("STARPU_SCHED");
	unsigned sched_ctx1 = starpu_sched_ctx_create(procs1, nprocs1, "ctx1", STARPU_SCHED_CTX_POLICY_NAME, sched?sched:"eager", 0);
	unsigned sched_ctx2 = starpu_sched_ctx_create(procs2, nprocs2, "ctx2", STARPU_SCHED_CTX_POLICY_NAME, sched?sched:"eager", 0);

	/*indicate what to do with the resources when context 2 finishes (it depends on your application)*/
	starpu_sched_ctx_set_inheritor(sched_ctx2, sched_ctx1);

	starpu_sched_ctx_display_workers(sched_ctx2, stderr);

	int i;
	for (i = 0; i < ntasks/2; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &sched_ctx_codelet1;
		task->cl_arg = NULL;

		/*submit tasks to context*/
		ret = starpu_task_submit_to_ctx(task,sched_ctx1);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/* tell starpu when you finished submitting tasks to this context
	   in order to allow moving resources from this context to the inheritor one
	   when its corresponding tasks finished executing */

	starpu_sched_ctx_finished_submit(sched_ctx1);

	/* task with no cuda impl submitted to a ctx with gpus only */
	struct starpu_task *task2 = starpu_task_create();
	task2->cl = &sched_ctx_codelet1;
	task2->cl_arg = NULL;

	/*submit tasks to context*/
	ret = starpu_task_submit_to_ctx(task2,sched_ctx2);
	if (ncuda == 0)
	{
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	else
	{
		STARPU_ASSERT_MSG(ret == -ENODEV, "submit task should ret enodev when the ctx does not have the PUs needed by the task");
	}

	for (i = 0; i < ntasks/2; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &sched_ctx_codelet2;
		task->cl_arg = NULL;

		ret = starpu_task_submit_to_ctx(task,sched_ctx2);

		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_sched_ctx_finished_submit(sched_ctx2);

	/* wait for all tasks at the end*/
	starpu_task_wait_for_all();

	starpu_sched_ctx_add_workers(procs1, nprocs1, sched_ctx2);
	starpu_sched_ctx_delete(sched_ctx1);
	starpu_sched_ctx_delete(sched_ctx2);
	printf("tasks executed %d out of %d\n", tasks_executed, ntasks/2);

enodev:
	starpu_shutdown();
	return nprocs1 == 0 ? 77 : 0;
}
