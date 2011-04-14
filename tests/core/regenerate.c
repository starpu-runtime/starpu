/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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
#include <pthread.h>
#include <starpu.h>

#define FPRINTF(ofile, fmt, args ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ##args); }} while(0)

static unsigned ntasks = 65536;
static unsigned cnt = 0;

static unsigned completed = 0;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

static void callback(void *arg __attribute__ ((unused)))
{
	struct starpu_task *task = starpu_get_current_task();

	cnt++;

	if (cnt == ntasks)
	{
		task->regenerate = 0;
		FPRINTF(stderr, "Stop !\n");

		pthread_mutex_lock(&mutex);
		completed = 1;
		pthread_cond_signal(&cond);
		pthread_mutex_unlock(&mutex);
	}
}

static void dummy_func(void *descr[] __attribute__ ((unused)), void *arg __attribute__ ((unused)))
{
}

static starpu_codelet dummy_codelet = 
{
	.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL,
	.cpu_func = dummy_func,
	.cuda_func = dummy_func,
	.opencl_func = dummy_func,
	.model = NULL,
	.nbuffers = 0
};

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
	//	unsigned i;
	double timing;
	struct timeval start;
	struct timeval end;

	parse_args(argc, argv);

	starpu_init(NULL);

	struct starpu_task task;

	starpu_task_init(&task);

	task.cl = &dummy_codelet;
	task.regenerate = 1;
	task.detach = 1;

	task.callback_func = callback;

	FPRINTF(stderr, "#tasks : %u\n", ntasks);

	gettimeofday(&start, NULL);

	starpu_task_submit(&task);

	pthread_mutex_lock(&mutex);
	if (!completed)
		pthread_cond_wait(&cond, &mutex);
	pthread_mutex_unlock(&mutex);

	gettimeofday(&end, NULL);

	timing = (double)((end.tv_sec - start.tv_sec)*1000000
				+ (end.tv_usec - start.tv_usec));

	FPRINTF(stderr, "Total: %lf secs\n", timing/1000000);
	FPRINTF(stderr, "Per task: %lf usecs\n", timing/ntasks);

	starpu_shutdown();

	return 0;
}
