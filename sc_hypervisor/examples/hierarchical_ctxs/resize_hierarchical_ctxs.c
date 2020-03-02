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


unsigned val[3];
starpu_pthread_mutex_t mut[3];

/* Every implementation of a codelet must have this prototype, the first                                                                                                                                             * argument (buffers) describes the buffers/streams that are managed by the
 * DSM; the second arguments references read-only data that is passed as an
 * argument of the codelet (task->cl_arg). Here, "buffers" is unused as there
 * are no data input/output managed by the DSM (cl.nbuffers = 0) */

void cpu_func(__attribute__((unused))void *buffers[], void *cl_arg)
{
	unsigned sched_ctx = *((unsigned *) cl_arg);

	int i;
	for(i = 0; i < NINCR; i++)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&mut[sched_ctx - 1]);
		val[sched_ctx - 1]++;
		STARPU_PTHREAD_MUTEX_UNLOCK(&mut[sched_ctx - 1]);
	}
}

struct starpu_codelet cl = {0};

void* submit_tasks_thread(void *arg)
{
	unsigned sched_ctx = *((unsigned*)arg);
	starpu_sched_ctx_set_context(&sched_ctx);

	struct starpu_task *task[NTASKS];
	int i;
	for(i = 0; i < NTASKS; i++)
	{
		task[i] = starpu_task_create();
		cl.cpu_funcs[0] = cpu_func;
		cl.nbuffers = 0;

		task[i]->cl = &cl;

		task[i]->cl_arg = &sched_ctx;
		task[i]->cl_arg_size = sizeof(unsigned);

		task[i]->flops = NINCR*1000000000.0;
		int ret = starpu_task_submit(task[i]);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		if(i == NTASKS/2)
			sc_hypervisor_resize_ctxs(NULL, -1, NULL, -1);
	}

	starpu_task_wait_for_all();
	return NULL;
}

int main()
{
	int ret = starpu_init(NULL);

	if (ret == -ENODEV)
        return 77;


	/* create contexts */
	unsigned sched_ctx1 = starpu_sched_ctx_create(NULL, 0, "sched_ctx1", STARPU_SCHED_CTX_POLICY_NAME, "dmda", STARPU_SCHED_CTX_HIERARCHY_LEVEL, 0, 0);
	unsigned sched_ctx2 = starpu_sched_ctx_create(NULL, 0, "sched_ctx2", STARPU_SCHED_CTX_POLICY_NAME, "dmda", STARPU_SCHED_CTX_HIERARCHY_LEVEL, 1, 0);
	unsigned sched_ctx3 = starpu_sched_ctx_create(NULL, 0, "sched_ctx3", STARPU_SCHED_CTX_POLICY_NAME, "dmda", STARPU_SCHED_CTX_HIERARCHY_LEVEL, 1, 0);
	starpu_sched_ctx_set_inheritor(sched_ctx2, sched_ctx1);
	starpu_sched_ctx_set_inheritor(sched_ctx3, sched_ctx1);

	/* initialize the hypervisor */
	struct sc_hypervisor_policy policy;
	policy.custom = 0;
	/* indicate which strategy to use
	   in this particular case we use app_driven which allows the user to resize 
	   the ctxs dynamically at particular moments of the execution of the application */
	policy.name = "feft_lp";
	void *perf_counters = sc_hypervisor_init(&policy);

	/* let starpu know which performance counters should use 
	   to inform the hypervisor how the application and the resources are executing */
	starpu_sched_ctx_set_perf_counters(sched_ctx1, perf_counters);
	starpu_sched_ctx_set_perf_counters(sched_ctx2, perf_counters);
	starpu_sched_ctx_set_perf_counters(sched_ctx3, perf_counters);

	double flops1 = NTASKS*NINCR*1000000000.0;
	double flops2 = NTASKS*NINCR*1000000000.0;
	double flops3 = NTASKS*NINCR*1000000000.0;
	/* register the contexts that should be managed by the hypervisor
	   and indicate an approximate amount of workload if known;
	   in this case we don't know it and we put 0 */
	sc_hypervisor_register_ctx(sched_ctx1, flops1);
	sc_hypervisor_register_ctx(sched_ctx2, flops2);
	sc_hypervisor_register_ctx(sched_ctx3, flops3);

	unsigned ncpus =  starpu_cpu_worker_get_count();

	sc_hypervisor_ctl(sched_ctx1,
			  SC_HYPERVISOR_MAX_WORKERS, ncpus,
			  SC_HYPERVISOR_NULL);

	sc_hypervisor_ctl(sched_ctx2,
			  SC_HYPERVISOR_MAX_WORKERS, ncpus,
			  SC_HYPERVISOR_NULL);

	sc_hypervisor_ctl(sched_ctx3,
			  SC_HYPERVISOR_MAX_WORKERS, ncpus,
			  SC_HYPERVISOR_NULL);

        /* lp strategy allows sizing the contexts because we know the total number of flops
	   to be executed */
	sc_hypervisor_size_ctxs(NULL, -1, NULL, -1);

	starpu_pthread_t tid[3];

	val[0] = 0;
	val[1] = 0;
	val[2] = 0;
	STARPU_PTHREAD_MUTEX_INIT(&mut[0], NULL);
	STARPU_PTHREAD_MUTEX_INIT(&mut[1], NULL);
	STARPU_PTHREAD_MUTEX_INIT(&mut[2], NULL);

	/* we create two threads to simulate simultaneous submission of tasks */
	STARPU_PTHREAD_CREATE(&tid[0], NULL, submit_tasks_thread, (void*)&sched_ctx1);
	STARPU_PTHREAD_CREATE(&tid[1], NULL, submit_tasks_thread, (void*)&sched_ctx2);
	STARPU_PTHREAD_CREATE(&tid[2], NULL, submit_tasks_thread, (void*)&sched_ctx3);

	STARPU_PTHREAD_JOIN(tid[0], NULL);
	STARPU_PTHREAD_JOIN(tid[1], NULL);
	STARPU_PTHREAD_JOIN(tid[2], NULL);

	/* free starpu and hypervisor data */
	starpu_shutdown();
	sc_hypervisor_shutdown();

	FPRINTF(stdout, "ctx = %u executed %u counter_tests out of %d \n", sched_ctx1, val[0], NTASKS*NINCR);
	FPRINTF(stdout, "ctx = %u executed %u counter_tests out of %d \n", sched_ctx2, val[1], NTASKS*NINCR);
	FPRINTF(stdout, "ctx = %u executed %u counter_tests out of %d \n", sched_ctx3, val[2], NTASKS*NINCR);
	return 0;
}
