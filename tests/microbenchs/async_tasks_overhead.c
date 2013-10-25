/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2012  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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

#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <starpu.h>
#include "../helper.h"

static unsigned ntasks = 65536;

//static unsigned finished = 0;

static double cumulated = 0.0;
static double cumulated_push = 0.0;
static double cumulated_pop = 0.0;

static
void dummy_func(void *descr[] STARPU_ATTRIBUTE_UNUSED, void *arg STARPU_ATTRIBUTE_UNUSED)
{
}

static struct starpu_codelet dummy_codelet =
{
	.cpu_funcs = {dummy_func, NULL},
	.cuda_funcs = {dummy_func, NULL},
        .opencl_funcs = {dummy_func, NULL},
	.cpu_funcs_name = {"dummy_func", NULL},
	.model = NULL,
	.nbuffers = 0
};

//static void inject_one_task(void)
//{
//	struct starpu_task *task = starpu_task_create();
//
//	task->cl = &dummy_codelet;
//	task->cl_arg = NULL;
//	task->detach = 0;
//
//	int ret = starpu_task_submit(task);
//	STARPU_ASSERT(!ret);
//}

static void usage(char **argv)
{
	fprintf(stderr, "%s [-i ntasks] [-p sched_policy] [-h]\n", argv[0]);
	exit(-1);
}

static void parse_args(int argc, char **argv, struct starpu_conf *conf)
{
	int c;
	while ((c = getopt(argc, argv, "i:p:h")) != -1)
	switch(c)
	{
		case 'i':
			ntasks = atoi(optarg);
			break;
		case 'p':
			conf->sched_policy_name = optarg;
			break;
		case 'h':
			usage(argv);
			break;
	}
}

int main(int argc, char **argv)
{
	int ret;
	unsigned i;
	double timing;
	struct timeval start;
	struct timeval end;

	struct starpu_conf conf;
	starpu_conf_init(&conf);

	parse_args(argc, argv, &conf);

	ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

	fprintf(stderr, "#tasks : %u\n", ntasks);

	/* Create an array of tasks */
	struct starpu_task **tasks = (struct starpu_task **) malloc(ntasks*sizeof(struct starpu_task *));

	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &dummy_codelet;
		task->cl_arg = NULL;
		task->detach = 0;
		tasks[i] = task;
	}

	gettimeofday(&start, NULL);
	for (i = 0; i < ntasks; i++)
	{
		ret = starpu_task_submit(tasks[i]);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	gettimeofday(&end, NULL);

	/* Read profiling feedback */
	for (i = 0; i < ntasks; i++)
	{
		struct starpu_profiling_task_info *info;
		info = tasks[i]->profiling_info;

		double queued = starpu_timing_timespec_delay_us(&info->push_end_time, &info->pop_end_time);
		double length = starpu_timing_timespec_delay_us(&info->submit_time, &info->end_time);
		double push_duration = starpu_timing_timespec_delay_us(&info->push_start_time, &info->push_end_time);
		double pop_duration = starpu_timing_timespec_delay_us(&info->pop_start_time, &info->pop_end_time);
		starpu_task_destroy(tasks[i]);
		cumulated += (length - queued);
		cumulated_push += push_duration;
		cumulated_pop += pop_duration;
	}

	timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	fprintf(stderr, "Total: %f secs\n", timing/1000000);
	fprintf(stderr, "Per task: %f usecs\n", timing/ntasks);
	fprintf(stderr, "Per task (except scheduler): %f usecs\n", cumulated/ntasks);
	fprintf(stderr, "Per task (push): %f usecs\n", cumulated_push/ntasks);
	fprintf(stderr, "Per task (pop): %f usecs\n", cumulated_pop/ntasks);

        {
                char *output_dir = getenv("STARPU_BENCH_DIR");
                char *bench_id = getenv("STARPU_BENCH_ID");

                if (output_dir && bench_id)
		{
                        char file[1024];
                        FILE *f;

                        sprintf(file, "%s/async_tasks_overhead_total.dat", output_dir);
                        f = fopen(file, "a");
                        fprintf(f, "%s\t%f\n", bench_id, timing/1000000);
                        fclose(f);

                        sprintf(file, "%s/async_tasks_overhead_per_task.dat", output_dir);
                        f = fopen(file, "a");
                        fprintf(f, "%s\t%f\n", bench_id, timing/ntasks);
                        fclose(f);
                }
        }

	starpu_shutdown();
	free(tasks);

	return EXIT_SUCCESS;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
