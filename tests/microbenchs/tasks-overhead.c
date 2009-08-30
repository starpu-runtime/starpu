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
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

#include <starpu.h>

static unsigned ntasks = 65536;

static void *dummy_func(void *arg __attribute__ ((unused)))
{
	return NULL;
}

static starpu_codelet dummy_codelet = 
{
	.where = CORE|CUBLAS,
	.core_func = dummy_func,
	.cublas_func = dummy_func,
	.model = NULL,
	.nbuffers = 0
};

void inject_one_task(void)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &dummy_codelet;
	task->cl_arg = NULL;
	task->callback_func = NULL;
	task->synchronous = 1;

	starpu_submit_task(task);
}

static void parse_args(int argc, char **argv)
{
	int c;
	while ((c = getopt(argc, argv, "i:")) != -1)
	switch(c) {
		case 'i':
			ntasks = atoi(optarg);
			break;
	}
}

int main(int argc, char **argv)
{
	unsigned i;

	double timing_submit;
	struct timeval start_submit;
	struct timeval end_submit;

	double timing_exec;
	struct timeval start_exec;
	struct timeval end_exec;

	parse_args(argc, argv);

	starpu_init(NULL);

	fprintf(stderr, "#tasks : %d\n", ntasks);

	/* submit tasks (but don't execute them yet !) */

	gettimeofday(&start_submit, NULL);
	for (i = 1; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();
			task->callback_func = NULL;
			task->cl = &dummy_codelet;
			task->cl_arg = NULL;
			task->synchronous = 0;
			task->use_tag = 1;
			task->tag_id = (starpu_tag_t)i;

		starpu_tag_declare_deps((starpu_tag_t)i, 1, (starpu_tag_t)(i-1));

		starpu_submit_task(task);
	}

	/* submit the first task */
	struct starpu_task *task = starpu_task_create();
		task->cl = &dummy_codelet;
		task->cl_arg = NULL;
		task->callback_func = NULL;
		task->synchronous = 0;
		task->use_tag = 1;
		task->tag_id = (starpu_tag_t)0;

	starpu_submit_task(task);

	gettimeofday(&end_submit, NULL);

	/* wait for the execution of the tasks */
	gettimeofday(&start_exec, NULL);
	starpu_tag_wait((starpu_tag_t)(ntasks - 1));
	gettimeofday(&end_exec, NULL);

	timing_submit = (double)((end_submit.tv_sec - start_submit.tv_sec)*1000000 + (end_submit.tv_usec - start_submit.tv_usec));
	timing_exec = (double)((end_exec.tv_sec - start_exec.tv_sec)*1000000 + (end_exec.tv_usec - start_exec.tv_usec));

	fprintf(stderr, "Total submit: %lf secs\n", timing_submit/1000000);
	fprintf(stderr, "Per task submit: %lf usecs\n", timing_submit/ntasks);
	fprintf(stderr, "Total execution: %lf secs\n", timing_exec/1000000);
	fprintf(stderr, "Per task execution: %lf usecs\n", timing_exec/ntasks);

	fprintf(stderr, "Total: %lf secs\n", (timing_submit+timing_exec)/1000000);
	fprintf(stderr, "Per task: %lf usecs\n", (timing_submit+timing_exec)/ntasks);

	starpu_shutdown();

	return 0;
}
