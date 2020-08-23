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
 * Measure the memory bandwidth available to kernels depending on the number of
 * kernels and number of idle workers.
 */

#ifdef STARPU_QUICK_CHECK
static size_t size = 1024;
static unsigned cpustep = 4;
#else
/* Must be bigger than available cache size per core, 64MiB should be enough */
static size_t size = 64UL << 20;
static unsigned cpustep = 1;
#endif

static unsigned noalone = 0;
static unsigned iter = 30;
static unsigned ncpus;
static starpu_pthread_barrier_t barrier;
static float *result;

void bw_func(void *descr[], void *arg)
{
	void *src;
	void *dst;
	unsigned i;
	double start, stop;
	int ret;

	ret = posix_memalign(&src, getpagesize(), size);
	STARPU_ASSERT(ret == 0);
	ret = posix_memalign(&dst, getpagesize(), size);
	STARPU_ASSERT(ret == 0);
	memset(src, 0, size);

	STARPU_PTHREAD_BARRIER_WAIT(&barrier);
	start = starpu_timing_now();
	for (i = 0; i < iter; i++)
		memcpy(dst, src, size);
	stop = starpu_timing_now();
	STARPU_PTHREAD_BARRIER_WAIT(&barrier);

	result[starpu_worker_get_id()] = (size*iter) / (stop - start);

	free(src);
	free(dst);
}

static struct starpu_codelet bw_codelet =
{
	.cpu_funcs = {bw_func},
	.model = NULL,
	.nbuffers = 0,
};

static void usage(char **argv)
{
	fprintf(stderr, "Usage: %s [-n iter] [-s size (MB)] [-i increment] [-a]\n", argv[0]);
	fprintf(stderr, "\t-n iter\tNumber of iterations\n");
	fprintf(stderr, "\t-s size\tBuffer size in MB\n");
	fprintf(stderr, "\t-i increment\tCpu number increment\n");
	fprintf(stderr, "\t-a\tDo not run the alone test\n");
	exit(EXIT_FAILURE);
}

static void parse_args(int argc, char **argv)
{
	int c;
	while ((c = getopt(argc, argv, "n:s:c:ah")) != -1)
	switch(c)
	{
		case 'n':
			iter = atoi(optarg);
			break;
		case 's':
			size = (long)atoi(optarg) << 20;
			break;
		case 'c':
			cpustep = atoi(optarg);
			break;
		case 'a':
			noalone = 1;
			break;
		case 'h':
			usage(argv);
			break;
	}
}

static float bench(int *argc, char ***argv, unsigned nbusy, unsigned nidle)
{
	int ret;
	unsigned i;
	struct starpu_conf conf;
	float bw;

	starpu_conf_init(&conf);
	conf.ncuda = 0;
	conf.nopencl = 0;
	conf.nmic = 0;
	conf.nmpi_ms = 0;
	conf.ncpus = nbusy + nidle;

	ret = starpu_initialize(&conf, argc, argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	STARPU_PTHREAD_BARRIER_INIT(&barrier, NULL, nbusy);

	for (i = 0; i < nbusy; i++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &bw_codelet;
		task->execute_on_a_specific_worker = 1;
		task->workerid = i;
		ret = starpu_task_submit(task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_task_wait_for_all();
	starpu_shutdown();

	for (bw = 0., i = 0; i < nbusy; i++)
	{
		bw += result[i];
	}
	return bw;
}

int main(int argc, char **argv)
{
	int ret;
	unsigned n;
	struct starpu_conf conf;
	float alone, idle;

	parse_args(argc, argv);

	starpu_conf_init(&conf);
	conf.ncuda = 0;
	conf.nopencl = 0;
	conf.nmic = 0;
	conf.nmpi_ms = 0;

	ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	ncpus = starpu_cpu_worker_get_count();
	starpu_shutdown();

	result = malloc(ncpus * sizeof(result[0]));

	printf("# nw\talone\t\t+idle\t\tidle efficiency\n");
	for (n = 1; n <= ncpus; n += cpustep)
	{
		if (noalone)
			alone = 0.;
		else
			alone = bench(&argc, &argv, n, 0);
		idle = bench(&argc, &argv, n, ncpus-n);
		printf("%d\t%f\t%f\t%f\n", n, alone/1000, idle/1000, idle*100/alone);
	}

	free(result);

	return EXIT_SUCCESS;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	free(result);
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
