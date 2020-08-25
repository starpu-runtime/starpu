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
static unsigned total_ncpus;
static starpu_pthread_barrier_t barrier_begin, barrier_end;
static float *result;
static void **buffers;	/* Indexed by logical core number */
static char padding1[STARPU_CACHELINE_SIZE];
static volatile char finished;
static char padding2[STARPU_CACHELINE_SIZE];

static unsigned interleave(unsigned i);

/* Initialize the buffer locally */
void initialize_buffer(void *foo)
{
	unsigned id = starpu_worker_get_id();
#ifdef STARPU_HAVE_POSIX_MEMALIGN
	int ret = posix_memalign(&buffers[id], getpagesize(), 2*size);
	STARPU_ASSERT(ret == 0);
#else
	buffers[id] = malloc(2*size);
#endif
	memset(buffers[id], 0, 2*size);
}

/* Actual transfer codelet */
void bw_func(void *descr[], void *arg)
{
	int id = (uintptr_t) arg;
	void *src = buffers[id];
	void *dst = (void*) ((uintptr_t)src + size);
	unsigned i;
	double start, stop;

	STARPU_PTHREAD_BARRIER_WAIT(&barrier_begin);
	start = starpu_timing_now();
	for (i = 0; i < iter; i++)
	{
		memcpy(dst, src, size);
		STARPU_SYNCHRONIZE();
	}
	stop = starpu_timing_now();
	STARPU_PTHREAD_BARRIER_WAIT(&barrier_end);
	finished = 1;

	result[id] = (size*iter) / (stop - start);
}

static struct starpu_codelet bw_codelet =
{
	.cpu_funcs = {bw_func},
	.model = NULL,
	.nbuffers = 0,
};

/* Codelet that waits for completion while doing lots of cpu yields (nop). */
void nop_func(void *descr[], void *arg)
{
	STARPU_PTHREAD_BARRIER_WAIT(&barrier_begin);
	while (!finished)
	{
		unsigned i;
		for (i = 0; i < 1000000; i++)
			STARPU_UYIELD();
		STARPU_SYNCHRONIZE();
	}
}

static struct starpu_codelet nop_codelet =
{
	.cpu_funcs = {nop_func},
	.model = NULL,
	.nbuffers = 0,
};

/* Codelet that waits for completion while aggressively reading the finished variable. */
void sync_func(void *descr[], void *arg)
{
	STARPU_PTHREAD_BARRIER_WAIT(&barrier_begin);
	while (!finished)
	{
		STARPU_VALGRIND_YIELD();
		STARPU_SYNCHRONIZE();
	}
}

static struct starpu_codelet sync_codelet =
{
	.cpu_funcs = {sync_func},
	.model = NULL,
	.nbuffers = 0,
};

static void usage(char **argv)
{
	fprintf(stderr, "Usage: %s [-n niter] [-s size (MB)] [-c cpustep] [-a]\n", argv[0]);
	fprintf(stderr, "\t-n niter\tNumber of iterations\n");
	fprintf(stderr, "\t-s size\tBuffer size in MB\n");
	fprintf(stderr, "\t-c cpustep\tCpu number increment\n");
	fprintf(stderr, "\t-a Do not run the alone test\n");
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

static unsigned interleave(unsigned i)
{
	/* TODO: rather distribute over hierarchy */
	if (total_ncpus > 1)
		return (i % (total_ncpus/2))*2 + i / (total_ncpus/2);
	else
		return 0;
}

enum sleep_type {
	PAUSE,
	NOP,
	SYNC,
	SCHED,
};

static float bench(int *argc, char ***argv, unsigned nbusy, unsigned ncpus, int intl, enum sleep_type sleep)
{
	int ret;
	unsigned i;
	struct starpu_conf conf;
	float bw;

	starpu_conf_init(&conf);
	conf.precedence_over_environment_variables = 1;
	conf.ncuda = 0;
	conf.nopencl = 0;
	conf.nmic = 0;
	conf.nmpi_ms = 0;
	conf.ncpus = ncpus;

	if (intl && sleep == PAUSE)
	{
		conf.use_explicit_workers_bindid = 1;
		for (i = 0; i < ncpus; i++)
			conf.workers_bindid[i] = interleave(i);
	}

	ret = starpu_initialize(&conf, argc, argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (sleep == PAUSE || sleep == SCHED)
		/* In these cases we don't have a task on each cpu */
		STARPU_PTHREAD_BARRIER_INIT(&barrier_begin, NULL, nbusy);
	else
		STARPU_PTHREAD_BARRIER_INIT(&barrier_begin, NULL, ncpus);

	STARPU_PTHREAD_BARRIER_INIT(&barrier_end, NULL, nbusy);

	finished = 0;
	for (i = 0; i < ncpus; i++)
		result[i] = NAN;

	for (i = 0; i < nbusy; i++)
	{
		struct starpu_task *task = starpu_task_create();
		task->cl = &bw_codelet;

		if (intl)
			task->cl_arg = (void*) (uintptr_t) interleave(i);
		else
			task->cl_arg = (void*) (uintptr_t) i;

		task->execute_on_a_specific_worker = 1;
		if (intl && sleep != PAUSE) /* In the pause case we interleaved above */
			task->workerid = interleave(i);
		else
			task->workerid = i;

		ret = starpu_task_submit(task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	if (sleep != PAUSE && sleep != SCHED)
	{
		/* Add waiting tasks */
		for ( ; i < ncpus; i++)
		{
			struct starpu_task *task = starpu_task_create();
			switch (sleep)
			{
			case NOP:
				task->cl = &nop_codelet;
				break;
			case SYNC:
				task->cl = &sync_codelet;
				break;
			default:
				STARPU_ASSERT(0);
			}
			task->execute_on_a_specific_worker = 1;
			task->workerid = interleave(i);
			ret = starpu_task_submit(task);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}
	}


	starpu_task_wait_for_all();
	starpu_shutdown();

	for (bw = 0., i = 0; i < nbusy; i++)
	{
		if (intl)
			bw += result[interleave(i)];
		else
			bw += result[i];
	}
	return bw;
}

int main(int argc, char **argv)
{
	int ret;
	unsigned n;
	struct starpu_conf conf;
	float alone, alone_int, alone_int_nop, alone_int_sync, sched, sched_int;

	parse_args(argc, argv);

	starpu_conf_init(&conf);
	conf.precedence_over_environment_variables = 1;
	conf.ncuda = 0;
	conf.nopencl = 0;
	conf.nmic = 0;
	conf.nmpi_ms = 0;

	ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	total_ncpus = starpu_cpu_worker_get_count();

	buffers = malloc(total_ncpus * sizeof(*buffers));
	starpu_execute_on_each_worker_ex(initialize_buffer, NULL, STARPU_CPU, "init_buffer");
	starpu_shutdown();

	if (total_ncpus == 0)
		return STARPU_TEST_SKIPPED;

	result = malloc(total_ncpus * sizeof(result[0]));

	printf("# nw\ta comp.\t+sched\teff%%\ta scat.\t+nop\t+sync\t+sched\teff%% vs nop\n");
	for (n = cpustep; n <= total_ncpus; n += cpustep)
	{
		if (noalone)
		{
			alone = 0.;
			alone_int = 0.;
			alone_int_nop = 0.;
			alone_int_sync = 0.;
		}
		else
		{
			alone = bench(&argc, &argv, n, n, 0, PAUSE);
			alone_int = bench(&argc, &argv, n, n, 1, PAUSE);
			alone_int_nop = bench(&argc, &argv, n, total_ncpus, 1, NOP);
			alone_int_sync = bench(&argc, &argv, n, total_ncpus, 1, SYNC);
		}
		sched = bench(&argc, &argv, n, total_ncpus, 0, SCHED);
		sched_int = bench(&argc, &argv, n, total_ncpus, 1, SCHED);
		printf("%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n",
				n,
				alone/1000,
				sched/1000, sched*100/alone,
				alone_int/1000,
				alone_int_nop/1000,
				alone_int_sync/1000,
				sched_int/1000, sched_int*100/alone_int_nop);
		fflush(stdout);
	}

	free(result);

	for (n = 0; n < total_ncpus; n++)
		free(buffers[n]);
	free(buffers);

	return EXIT_SUCCESS;
}
