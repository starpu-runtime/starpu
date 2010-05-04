/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <sys/time.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

static unsigned ntasks = 65536;

static void dummy_func(void *descr[], void *arg)
{
}

static starpu_codelet dummy_codelet = 
{
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL|STARPU_GORDON,
	.cpu_func = dummy_func,
	.cuda_func = dummy_func,
	.opencl_func = dummy_func,
#ifdef STARPU_USE_GORDON
	.gordon_func = 0, /* this will be defined later */
#endif
	.model = NULL,
	.nbuffers = 0
};

static void init_gordon_kernel(void)
{
#ifdef STARPU_USE_GORDON
	unsigned elf_id = 
		gordon_register_elf_plugin("./microbenchs/null_kernel_gordon.spuelf");
	gordon_load_plugin_on_all_spu(elf_id);

	unsigned gordon_null_kernel =
		gordon_register_kernel(elf_id, "empty_kernel");
	gordon_load_kernel_on_all_spu(gordon_null_kernel);

	dummy_codelet.gordon_func = gordon_null_kernel;
#endif
}

static void inject_one_task(void)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &dummy_codelet;
	task->cl_arg = NULL;
	task->callback_func = NULL;
	task->callback_arg = NULL;

	int ret = starpu_task_submit(task);
	STARPU_ASSERT(!ret);
}

static struct starpu_conf conf = {
	.sched_policy_name = NULL,
	.ncpus = -1,
	.ncuda = -1,
        .nopencl = -1,
	.nspus = -1,
	.use_explicit_workers_bindid = 0,
	.use_explicit_workers_cuda_gpuid = 0,
	.use_explicit_workers_opencl_gpuid = 0,
	.calibrate = 0
};

static void usage(char **argv)
{
	fprintf(stderr, "%s [-i ntasks] [-p sched_policy] [-h]\n", argv[0]);
	exit(-1);
}

static void parse_args(int argc, char **argv)
{
	int c;
	while ((c = getopt(argc, argv, "i:p:h")) != -1)
	switch(c) {
		case 'i':
			ntasks = atoi(optarg);
			break;
		case 'p':
			conf.sched_policy_name = optarg;
			break;
		case 'h':
			usage(argv);
			break;
	}
}

int main(int argc, char **argv)
{
	unsigned i;
	double timing;
	struct timeval start;
	struct timeval end;

	parse_args(argc, argv);

	starpu_init(&conf);

	init_gordon_kernel();

	fprintf(stderr, "#tasks : %d\n", ntasks);

	gettimeofday(&start, NULL);
	for (i = 0; i < ntasks; i++)
	{
		inject_one_task();
	}

	starpu_wait_all_tasks();

	gettimeofday(&end, NULL);

	timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	fprintf(stderr, "Total: %lf secs\n", timing/1000000);
	fprintf(stderr, "Per task: %lf usecs\n", timing/ntasks);

	starpu_shutdown();

	return 0;
}
