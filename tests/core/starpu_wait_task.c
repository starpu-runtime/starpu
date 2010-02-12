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

#include <sys/time.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#include <starpu.h>

static unsigned ntasks = 65536;

static void dummy_func(void *descr[] __attribute__ ((unused)), void *arg __attribute__ ((unused)))
{
}

static starpu_codelet dummy_codelet = 
{
	.where = STARPU_CORE|STARPU_CUDA,
	.core_func = dummy_func,
	.cuda_func = dummy_func,
	.model = NULL,
	.nbuffers = 0
};

static void usage(char **argv)
{
	fprintf(stderr, "%s [-i ntasks] [-h]\n", argv[0]);
	exit(-1);
}

static void parse_args(int argc, char **argv)
{
	int c;
	while ((c = getopt(argc, argv, "i:t:h")) != -1)
	switch(c) {
		case 'i':
			ntasks = atoi(optarg);
			break;
		case 'h':
			usage(argv);
			break;
	}
}

int main(int argc, char **argv)
{
	double timing;
	struct timeval start;
	struct timeval end;

	parse_args(argc, argv);

	starpu_init(NULL);

	fprintf(stderr, "#tasks : %d\n", ntasks);

	gettimeofday(&start, NULL);

	int i;
	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &dummy_codelet;
		task->cl_arg = NULL;
		task->callback_func = NULL;
		task->callback_arg = NULL;

		task->detach = 0;
		task->destroy = 0;
		
		int ret = starpu_submit_task(task);
		STARPU_ASSERT(!ret);

		starpu_wait_task(task);

		starpu_task_destroy(task);
	}

	gettimeofday(&end, NULL);

	timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	fprintf(stderr, "Total: %lf secs\n", timing/1000000);
	fprintf(stderr, "Per task: %lf usecs\n", timing/ntasks);

	starpu_shutdown();

	return 0;
}
