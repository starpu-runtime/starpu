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

static pthread_mutex_t mutex;
static pthread_cond_t cond;

static unsigned ntasks = 65536;
static unsigned cnt;

static unsigned finished = 0;

static void *dummy_func(void *arg __attribute__ ((unused)))
{
	return NULL;
}

static starpu_codelet dummy_codelet = 
{
	.where = ANY,
	.core_func = dummy_func,
	.cublas_func = dummy_func,
	.model = NULL,
	.nbuffers = 0
};

void callback(void *arg)
{
	unsigned res = STARPU_ATOMIC_ADD(&cnt, -1);

	if (res == 0)
	{
		pthread_mutex_lock(&mutex);
		finished = 1;
		pthread_cond_signal(&cond);
		pthread_mutex_unlock(&mutex);
	}
}

static void inject_one_task(void)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &dummy_codelet;
	task->cl_arg = NULL;
	task->callback_func = callback;
	task->callback_arg = NULL;

	starpu_submit_task(task);
}

static struct starpu_conf conf = {
	.sched_policy = NULL,
	.ncpus = -1,
	.ncuda = -1,
	.nspus = -1,
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
			conf.sched_policy = optarg;
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

	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&cond, NULL);

	parse_args(argc, argv);

	cnt = ntasks;

	starpu_init(&conf);

	fprintf(stderr, "#tasks : %d\n", ntasks);

	gettimeofday(&start, NULL);
	for (i = 0; i < ntasks; i++)
	{
		inject_one_task();
	}

	pthread_mutex_lock(&mutex);
	if (!finished)
		pthread_cond_wait(&cond, &mutex);
	pthread_mutex_unlock(&mutex);

	gettimeofday(&end, NULL);

	timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	fprintf(stderr, "Total: %lf secs\n", timing/1000000);
	fprintf(stderr, "Per task: %lf usecs\n", timing/ntasks);

	starpu_shutdown();

	return 0;
}
