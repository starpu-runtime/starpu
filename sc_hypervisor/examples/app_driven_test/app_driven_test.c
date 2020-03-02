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

#include <stdio.h>
#include <stdint.h>
#include <starpu.h>
#include <sc_hypervisor.h>

#define NTASKS 1000
#define NINCR 10
#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

struct params
{
	unsigned sched_ctx;
	int task_tag;
};

unsigned val[2];
starpu_pthread_mutex_t mut[2];

/* Every implementation of a codelet must have this prototype, the first                                                                                                                                             * argument (buffers) describes the buffers/streams that are managed by the
 * DSM; the second arguments references read-only data that is passed as an
 * argument of the codelet (task->cl_arg). Here, "buffers" is unused as there
 * are no data input/output managed by the DSM (cl.nbuffers = 0) */

void cpu_func(__attribute__((unused))void *buffers[], void *cl_arg)
{
	struct params *params = (struct params *) cl_arg;

	int i;
	for(i = 0; i < NINCR; i++)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&mut[params->sched_ctx - 1]);
		val[params->sched_ctx - 1]++;
		STARPU_PTHREAD_MUTEX_UNLOCK(&mut[params->sched_ctx - 1]);
	}
	if(params->task_tag != 0)
		FPRINTF(stdout, "Task with tag %d executed in ctx = %u %u counter_tests\n", params->task_tag, params->sched_ctx, val[params->sched_ctx - 1]);
}

struct starpu_codelet cl = {0};

/* the management of the tags is done by the user */
/* who will take care that the tags will be unique */
int tag = 1;
void* submit_tasks_thread(void *arg)
{
	unsigned sched_ctx = *((unsigned*)arg);
	starpu_sched_ctx_set_context(&sched_ctx);

	struct starpu_task *task[NTASKS];
	struct params params[NTASKS];
	int i;
	for(i = 0; i < NTASKS; i++)
	{
		task[i] = starpu_task_create();
//		usleep(5000);
		cl.cpu_funcs[0] = cpu_func;
		cl.nbuffers = 0;

		task[i]->cl = &cl;

		if(sched_ctx == 1 && i == 5)
		{
			/* tag the tasks whose execution will start the resizing process */
			task[i]->hypervisor_tag = tag;
			/* indicate particular settings the context should have when the 
			   resizing will be done */
			sc_hypervisor_ctl(sched_ctx,
						   SC_HYPERVISOR_TIME_TO_APPLY, tag,
						   SC_HYPERVISOR_MIN_WORKERS, 2,
						   SC_HYPERVISOR_MAX_WORKERS, 12,
						   SC_HYPERVISOR_NULL);
			printf("require resize for sched_ctx %u at tag %d\n", sched_ctx, tag);
			/* specify that the contexts should be resized when the task having this
			   particular tag will finish executing */
			sc_hypervisor_post_resize_request(sched_ctx, tag);
		}

		params[i].sched_ctx = sched_ctx;
		params[i].task_tag = task[i]->hypervisor_tag;

		task[i]->cl_arg = &params[i];
		task[i]->cl_arg_size = sizeof(params);

		int ret = starpu_task_submit(task[i]);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_task_wait_for_all();
	return NULL;
}

int main()
{
	int ret = starpu_init(NULL);

	if (ret == -ENODEV)
        return 77;

	int num_workers = starpu_worker_get_count();
	int nres1 = num_workers;
	int nres2 = num_workers;
	int ressources1[nres1];
	int ressources2[nres2];
	int i;
	for(i = 0; i < nres1; i++)
		ressources1[i] = i;

	for(i = 0; i < nres2; i++)
		ressources2[i] = i;

	/* create contexts */
	unsigned sched_ctx1 = starpu_sched_ctx_create(ressources1, nres1, "sched_ctx1", STARPU_SCHED_CTX_POLICY_NAME, "dmda", 0);
	unsigned sched_ctx2 = starpu_sched_ctx_create(ressources2, nres2, "sched_ctx2", STARPU_SCHED_CTX_POLICY_NAME, "dmda", 0);

	/* initialize the hypervisor */
	struct sc_hypervisor_policy policy = {};

	policy.custom = 0;
	/* indicate which strategy to use
	   in this particular case we use app_driven which allows the user to resize 
	   the ctxs dynamically at particular moments of the execution of the application */
	policy.name = "app_driven";
	void *perf_counters = sc_hypervisor_init(&policy);

	/* let starpu know which performance counters should use 
	   to inform the hypervisor how the application and the resources are executing */
	starpu_sched_ctx_set_perf_counters(sched_ctx1, perf_counters);
	starpu_sched_ctx_set_perf_counters(sched_ctx2, perf_counters);

	/* register the contexts that should be managed by the hypervisor
	   and indicate an approximate amount of workload if known;
	   in this case we don't know it and we put 0 */
	sc_hypervisor_register_ctx(sched_ctx1, 0.0);
	sc_hypervisor_register_ctx(sched_ctx2, 0.0);

	starpu_pthread_t tid[2];

	val[0] = 0;
	val[1] = 0;
	STARPU_PTHREAD_MUTEX_INIT(&mut[0], NULL);
	STARPU_PTHREAD_MUTEX_INIT(&mut[1], NULL);

	/* we create two threads to simulate simultaneous submission of tasks */
	STARPU_PTHREAD_CREATE(&tid[0], NULL, submit_tasks_thread, (void*)&sched_ctx1);
	STARPU_PTHREAD_CREATE(&tid[1], NULL, submit_tasks_thread, (void*)&sched_ctx2);

	STARPU_PTHREAD_JOIN(tid[0], NULL);
	STARPU_PTHREAD_JOIN(tid[1], NULL);

	/* free starpu and hypervisor data */
	starpu_shutdown();
	sc_hypervisor_shutdown();

	FPRINTF(stdout, "ctx = %u executed %u counter_tests out of %d \n", sched_ctx1, val[0], NTASKS*NINCR);
	FPRINTF(stdout, "ctx = %u executed %u counter_tests out of %d \n", sched_ctx2, val[1], NTASKS*NINCR);
	return 0;
}
