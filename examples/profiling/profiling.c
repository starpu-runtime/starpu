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

/*
 * This examplifies how to get task execution profiling from the application.
 */

#include <starpu.h>
#include <assert.h>
#include <unistd.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#ifdef STARPU_QUICK_CHECK
static unsigned niter = 50;
#else
static unsigned niter = 500;
#endif

void sleep_codelet(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
	usleep(1000);
}

int main(int argc, char **argv)
{
	int ret;

	if (argc == 2)
		niter = atoi(argv[1]);

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Enable profiling */
	starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

#ifdef STARPU_QUICK_CHECK
	/* We should observe at least 50ms in the sleep time reported by every
	 * worker. */
	usleep(50000);
#else
	/* We should observe at least 500ms in the sleep time reported by every
	 * worker. */
	usleep(500000);
#endif

	struct starpu_codelet cl =
	{
		.cpu_funcs = {sleep_codelet},
		.cpu_funcs_name = {"sleep_codelet"},
		.cuda_funcs = {sleep_codelet},
		.opencl_funcs = {sleep_codelet},
		.nbuffers = 0,
		.name = "sleep"
	};

	struct starpu_task **tasks = (struct starpu_task **) malloc(niter*sizeof(struct starpu_task *));
	assert(tasks);

	unsigned i;
	for (i = 0; i < niter; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &cl;

		/* We will destroy the task structure by hand so that we can
		 * query the profiling info before the task is destroyed. */
		task->destroy = 0;

		tasks[i] = task;

		ret = starpu_task_submit(task);
		if (STARPU_UNLIKELY(ret == -ENODEV))
		{
			FPRINTF(stderr, "No worker may execute this task\n");
			exit(0);
		}
	}

	starpu_task_wait_for_all();

	double delay_sum = 0.0;
	double length_sum = 0.0;

	for (i = 0; i < niter; i++)
	{
		struct starpu_task *task = tasks[i];
		struct starpu_profiling_task_info *info = task->profiling_info;

		/* How much time did it take before the task started ? */
		delay_sum += starpu_timing_timespec_delay_us(&info->submit_time, &info->start_time);

		/* How long was the task execution ? */
		length_sum += starpu_timing_timespec_delay_us(&info->start_time, &info->end_time);

		/* We don't need the task structure anymore */
		starpu_task_destroy(task);
	}

	free(tasks);

	if (niter)
	{
		FPRINTF(stderr, "Avg. delay : %2.2lf us\n", (delay_sum)/niter);
		FPRINTF(stderr, "Avg. length : %2.2lf us\n", (length_sum)/niter);
	}

	/* Display the occupancy of all workers during the test */
	unsigned worker;
	for (worker = 0; worker < starpu_worker_get_count(); worker++)
	{
		struct starpu_profiling_worker_info worker_info;
		ret = starpu_profiling_worker_get_info(worker, &worker_info);
		STARPU_ASSERT(!ret);

		double total_time = starpu_timing_timespec_to_us(&worker_info.total_time);
		double executing_time = starpu_timing_timespec_to_us(&worker_info.executing_time);
		double sleeping_time = starpu_timing_timespec_to_us(&worker_info.sleeping_time);
		double overhead_time = total_time - executing_time - sleeping_time;

		float executing_ratio = 100.0*executing_time/total_time;
		float sleeping_ratio = 100.0*sleeping_time/total_time;
		float overhead_ratio = 100.0 - executing_ratio - sleeping_ratio;

		char workername[128];
		starpu_worker_get_name(worker, workername, 128);
		FPRINTF(stderr, "Worker %s:\n", workername);
		FPRINTF(stderr, "\t%d task(s)\n", worker_info.executed_tasks);
		FPRINTF(stderr, "\ttotal time : %.2lf ms\n", total_time*1e-3);
		FPRINTF(stderr, "\texec time  : %.2lf ms (%.2f %%)\n", executing_time*1e-3, executing_ratio);
		FPRINTF(stderr, "\tblocked time  : %.2lf ms (%.2f %%)\n", sleeping_time*1e-3, sleeping_ratio);
		FPRINTF(stderr, "\toverhead time: %.2lf ms (%.2f %%)\n", overhead_time*1e-3, overhead_ratio);
	}

	starpu_shutdown();

	return 0;
}
