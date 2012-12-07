/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2012  INRIA
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
#include <sched_ctx_hypervisor.h>

#include <pthread.h>

#define FPRINTF(ofile, fmt, args ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ##args); }} while(0)

/* Every implementation of a codelet must have this prototype, the first                                                                                                                                             * argument (buffers) describes the buffers/streams that are managed by the
 * DSM; the second arguments references read-only data that is passed as an
 * argument of the codelet (task->cl_arg). Here, "buffers" is unused as there
 * are no data input/output managed by the DSM (cl.nbuffers = 0) */
struct params
{
	unsigned sched_ctx;
    int task_tag;
};

void cpu_func(void *buffers[], void *cl_arg)
{
	struct params *params = (struct params *) cl_arg;

	int i;
	for(i = 0; i < 1000; i++);
	FPRINTF(stdout, "Hello world sched_ctx = %d task_tag = %d \n", params->sched_ctx, params->task_tag);
}

struct starpu_codelet cl = {};

int tag = 1;
void* start_thread(void *arg)
{
	unsigned sched_ctx = *((unsigned*)arg);
	starpu_set_sched_ctx(&sched_ctx);

	struct starpu_task *task[10];
	struct params params[10];
	int i;
	for(i = 0; i < 10; i++)
	{
		int j;
		for(j = 0; j < 1000; j++);
		task[i] = starpu_task_create();

		cl.where = STARPU_CPU;
		cl.cpu_funcs[0] = cpu_func;
		cl.nbuffers = 0;

		task[i]->cl = &cl;

		if(sched_ctx == 1 && i == 5)
		{
			task[i]->hypervisor_tag = tag;
			sched_ctx_hypervisor_ioctl(sched_ctx,
						   HYPERVISOR_TIME_TO_APPLY, tag,
						   HYPERVISOR_MIN_WORKERS, 2,
						   HYPERVISOR_MAX_WORKERS, 12,
						   HYPERVISOR_NULL);
			printf("require resize for sched_ctx %d at tag %d\n", sched_ctx, tag);
			sched_ctx_hypervisor_resize(sched_ctx, tag);
		}

		params[i].sched_ctx = sched_ctx;
		params[i].task_tag = task[i]->hypervisor_tag;

		task[i]->cl_arg = &params[i];
		task[i]->cl_arg_size = sizeof(params);

		starpu_task_submit(task[i]);
	}

	starpu_task_wait_for_all();
}

int main()
{
	int ret = starpu_init(NULL);

	if (ret == -ENODEV)
        return 77;

	int nres1 = 6;
	int nres2 = 6;
	int ressources1[nres1];
	int ressources2[nres2];
	int i;
	for(i = 0; i < nres1; i++)
		ressources1[i] = i;

	for(i = 0; i < nres2; i++)
		ressources2[i] = nres1+i;

	unsigned sched_ctx1 = starpu_create_sched_ctx("heft", ressources1, nres1, "sched_ctx1");
	unsigned sched_ctx2 = starpu_create_sched_ctx("heft", ressources2, nres2, "sched_ctx2");

	struct hypervisor_policy policy;
	policy.custom = 0;
	policy.name = "app_driven";
	void *perf_counters = sched_ctx_hypervisor_init(&policy);

	starpu_set_perf_counters(sched_ctx1, (struct starpu_performance_counters*)perf_counters);
	starpu_set_perf_counters(sched_ctx2, (struct starpu_performance_counters*)perf_counters);
	sched_ctx_hypervisor_register_ctx(sched_ctx1, 0.0);
	sched_ctx_hypervisor_register_ctx(sched_ctx2, 0.0);

	pthread_t tid[2];

	pthread_create(&tid[0], NULL, start_thread, (void*)&sched_ctx1);
	pthread_create(&tid[1], NULL, start_thread, (void*)&sched_ctx2);

	pthread_join(tid[0], NULL);
	pthread_join(tid[1], NULL);

	starpu_shutdown();
	sched_ctx_hypervisor_shutdown();
}
