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
 * Measure the cost of submitting synchronous tasks
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

static int inject_one_task(void)
{
	int ret;
	struct starpu_task *task = starpu_task_create();

	task->cl = &dummy_codelet;
	task->cl_arg = NULL;
	task->callback_func = NULL;
	task->synchronous = 1;

	ret = starpu_task_submit(task);
	return ret;

}

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

	fprintf(stderr, "#tasks : %u\n#buffers : %u\n", ntasks, nbuffers);

	start = starpu_timing_now();
	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &dummy_codelet;
		task->synchronous = 1;

		/* we have 8 buffers at most */
		for (buffer = 0; buffer < nbuffers; buffer++)
		{
			task->handles[buffer] = data_handles[buffer];
		}

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	end = starpu_timing_now();

	timing = end - start;

	fprintf(stderr, "Total: %f secs\n", timing/1000000);
	fprintf(stderr, "Per task: %f usecs\n", timing/ntasks);

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

                        snprintf(file, sizeof(file), "%s/sync_tasks_overhead_total%s.dat", output_dir, numberp);
                        f = fopen(file, "a");
                        fprintf(f, "%s\t%f\n", bench_id, timing/1000000);
                        fclose(f);

                        snprintf(file, sizeof(file), "%s/sync_tasks_overhead_per_task%s.dat", output_dir, numberp);
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

	return EXIT_SUCCESS;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
