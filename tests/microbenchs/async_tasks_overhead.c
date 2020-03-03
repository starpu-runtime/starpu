/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <unistd.h>
#include <starpu.h>
#include "../helper.h"

/*
 * Measure the cost of submitting asynchronous tasks
 */

starpu_data_handle_t data_handles[8];
float *buffers[8];

#ifdef STARPU_QUICK_CHECK
static unsigned ntasks = 128;
#else
static unsigned ntasks = 65536;
#endif
static unsigned nbuffers = 0;

#define BUFFERSIZE 16

//static unsigned finished = 0;

static double cumulated = 0.0;
static double cumulated_push = 0.0;
static double cumulated_pop = 0.0;

void dummy_func(void *descr[], void *arg)
{
	(void)descr;
	(void)arg;
}

static struct starpu_codelet dummy_codelet =
{
	.cpu_funcs = {dummy_func},
	.cuda_funcs = {dummy_func},
        .opencl_funcs = {dummy_func},
	.cpu_funcs_name = {"dummy_func"},
	.model = NULL,
	.nbuffers = 0,
	.modes = {STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW, STARPU_RW}
};

static void usage(char **argv)
{
	fprintf(stderr, "Usage: %s [-i ntasks] [-p sched_policy] [-b nbuffers] [-h]\n", argv[0]);
	exit(EXIT_FAILURE);
}

static void parse_args(int argc, char **argv, struct starpu_conf *conf)
{
	int c;
	while ((c = getopt(argc, argv, "i:b:p:h")) != -1)
	switch(c)
	{
		case 'i':
			ntasks = atoi(optarg);
			break;
		case 'b':
			nbuffers = atoi(optarg);
			dummy_codelet.nbuffers = nbuffers;
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
	double start;
	double end;

	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.ncpus = 2;

	parse_args(argc, argv, &conf);

	ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	unsigned buffer;
	for (buffer = 0; buffer < nbuffers; buffer++)
	{
		starpu_malloc((void**)&buffers[buffer], BUFFERSIZE*sizeof(float));
		starpu_vector_data_register(&data_handles[buffer], STARPU_MAIN_RAM, (uintptr_t)buffers[buffer], BUFFERSIZE, sizeof(float));
	}

	starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

	fprintf(stderr, "#tasks : %u\n#buffers : %u\n", ntasks, nbuffers);

	/* Create an array of tasks */
	struct starpu_task **tasks = (struct starpu_task **) malloc(ntasks*sizeof(struct starpu_task *));

	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &dummy_codelet;
		task->detach = 0;

		/* we have 8 buffers at most */
		for (buffer = 0; buffer < nbuffers; buffer++)
		{
			task->handles[buffer] = data_handles[buffer];
		}

		tasks[i] = task;
	}

	start = starpu_timing_now();
	for (i = 0; i < ntasks; i++)
	{
		ret = starpu_task_submit(tasks[i]);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	end = starpu_timing_now();

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

	timing = end - start;

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
                        char number[1+sizeof(nbuffers)*3+1];
                        const char *numberp;
                        char file[1024];
                        FILE *f;

                        if (nbuffers)
                        {
                                snprintf(number, sizeof(number), "_%u", nbuffers);
                                numberp = number;
                        }
                        else
                                numberp = "";

                        snprintf(file, sizeof(file), "%s/async_tasks_overhead_total%s.dat", output_dir, numberp);
                        f = fopen(file, "a");
                        fprintf(f, "%s\t%f\n", bench_id, timing/1000000);
                        fclose(f);

                        snprintf(file, sizeof(file), "%s/async_tasks_overhead_per_task%s.dat", output_dir, numberp);
                        f = fopen(file, "a");
                        fprintf(f, "%s\t%f\n", bench_id, timing/ntasks);
                        fclose(f);
                }
        }

	for (buffer = 0; buffer < nbuffers; buffer++)
	{
		starpu_data_unregister(data_handles[buffer]);
		starpu_free((void*)buffers[buffer]);
	}

	starpu_shutdown();
	free(tasks);

	return EXIT_SUCCESS;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	free(tasks);
	return STARPU_TEST_SKIPPED;
}
